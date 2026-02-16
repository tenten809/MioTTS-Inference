import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from miocodec import MioCodecModel
from miocodec.util import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate preset embedding(s) from many reference audios. "
            "Default strategy uses clustering + medoid selection."
        )
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory containing reference audio files (searched recursively by --audio-glob).",
    )
    parser.add_argument(
        "--audio-glob",
        default="**/*",
        help="Glob pattern used inside --audio-dir (default: **/*).",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help="Additional audio path. Can be specified multiple times.",
    )
    parser.add_argument(
        "--extensions",
        default=".wav,.flac,.ogg,.mp3,.m4a",
        help="Comma-separated extension allowlist for discovery in --audio-dir.",
    )
    parser.add_argument(
        "--preset-id", required=True, help="Preset id (output file name without extension)."
    )
    parser.add_argument("--output-dir", default="presets", help="Output directory.")
    parser.add_argument(
        "--model-id", default="Aratako/MioCodec-25Hz-44.1kHz-v2", help="MioCodec model id."
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu).")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=12.0,
        help="Segment length in seconds for each embedding extraction.",
    )
    parser.add_argument(
        "--segment-hop-seconds",
        type=float,
        default=12.0,
        help="Segment hop size in seconds.",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=3.0,
        help="Minimum segment length in seconds to keep.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of input files to use (0 means unlimited).",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Maximum number of segments to use (0 means unlimited).",
    )
    parser.add_argument(
        "--strategy",
        choices=["cluster_medoid", "mean"],
        default="cluster_medoid",
        help="How to build final preset(s).",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.8,
        help="Top ratio to keep by cosine similarity before aggregation/clustering.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=None,
        help="Optional minimum cosine similarity to centroid before aggregation/clustering.",
    )
    parser.add_argument(
        "--disable-outlier-removal",
        action="store_true",
        help="Disable pre-filtering and use all segment embeddings.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=0,
        help="Target number of clusters (0 means auto). Used for cluster_medoid strategy.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=4,
        help="Upper bound of auto-selected cluster count.",
    )
    parser.add_argument(
        "--cluster-iters",
        type=int,
        default=20,
        help="Spherical k-means iterations.",
    )
    parser.add_argument(
        "--cluster-min-size",
        type=int,
        default=3,
        help="Minimum members for a cluster to be emitted as a preset.",
    )
    parser.add_argument(
        "--save-all-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save all accepted clusters as <preset-id>__cXX.pt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering initialization.",
    )
    parser.add_argument(
        "--save-meta",
        action="store_true",
        help="Save metadata JSON next to output .pt.",
    )
    args = parser.parse_args()

    if not args.audio_dir and not args.audio:
        parser.error("Specify at least one of --audio-dir or --audio.")
    if args.segment_seconds <= 0:
        parser.error("--segment-seconds must be > 0.")
    if args.segment_hop_seconds <= 0:
        parser.error("--segment-hop-seconds must be > 0.")
    if args.min_segment_seconds <= 0:
        parser.error("--min-segment-seconds must be > 0.")
    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        parser.error("--keep-ratio must be in (0, 1].")
    if args.num_clusters < 0:
        parser.error("--num-clusters must be >= 0.")
    if args.max_clusters <= 0:
        parser.error("--max-clusters must be > 0.")
    if args.cluster_iters <= 0:
        parser.error("--cluster-iters must be > 0.")
    if args.cluster_min_size <= 0:
        parser.error("--cluster-min-size must be > 0.")
    return args


def parse_extensions(raw: str) -> set[str]:
    result: set[str] = set()
    for ext in raw.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        result.add(ext)
    return result


def discover_audio_files(
    audio_dir: str | None,
    audio_glob: str,
    audio_list: list[str],
    extensions: set[str],
    max_files: int,
) -> list[Path]:
    paths: list[Path] = []

    if audio_dir:
        base = Path(audio_dir).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"audio_dir not found or not a directory: {base}")
        for path in sorted(base.glob(audio_glob)):
            if not path.is_file():
                continue
            if path.suffix.lower() in extensions:
                paths.append(path.resolve())

    for raw in audio_list:
        path = Path(raw).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        paths.append(path)

    deduped = sorted(set(paths))
    if max_files > 0:
        deduped = deduped[:max_files]
    return deduped


def split_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_seconds: float,
    hop_seconds: float,
    min_segment_seconds: float,
) -> list[torch.Tensor]:
    waveform = waveform.squeeze()
    if waveform.dim() != 1:
        waveform = waveform.flatten()

    segment_len = max(1, int(segment_seconds * sample_rate))
    hop_len = max(1, int(hop_seconds * sample_rate))
    min_len = max(1, int(min_segment_seconds * sample_rate))
    total_len = int(waveform.numel())

    if total_len < min_len:
        return []
    if total_len <= segment_len:
        return [waveform]

    segments: list[torch.Tensor] = []
    start = 0
    while start < total_len:
        end = min(start + segment_len, total_len)
        chunk = waveform[start:end]
        if int(chunk.numel()) >= min_len:
            segments.append(chunk)
        if end >= total_len:
            break
        start += hop_len
    return segments


def prefilter_embeddings(
    embeddings: torch.Tensor,
    keep_ratio: float,
    min_cosine: float | None,
    disable_outlier_removal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = F.normalize(embeddings, dim=1, eps=1e-12)
    n = int(embeddings.shape[0])
    selected_indices = torch.arange(n, dtype=torch.long)
    if disable_outlier_removal or n < 3:
        return embeddings, selected_indices

    centroid = F.normalize(embeddings.mean(dim=0), dim=0, eps=1e-12)
    similarities = embeddings @ centroid

    keep_k = max(1, int(math.ceil(n * keep_ratio)))
    topk_indices = torch.topk(similarities, k=keep_k, largest=True).indices
    topk_mask = torch.zeros_like(similarities, dtype=torch.bool)
    topk_mask[topk_indices] = True

    if min_cosine is not None:
        cosine_mask = similarities >= float(min_cosine)
        mask = topk_mask & cosine_mask
    else:
        mask = topk_mask

    if int(mask.sum().item()) == 0:
        mask[topk_indices[0]] = True

    selected_indices = torch.where(mask)[0]
    return embeddings[selected_indices], selected_indices


def auto_cluster_count(num_items: int, max_clusters: int) -> int:
    if num_items < 20:
        target = 1
    elif num_items < 48:
        target = 2
    elif num_items < 96:
        target = 3
    else:
        target = 4
    return max(1, min(target, max_clusters, num_items))


def init_centers_farthest(
    embeddings: torch.Tensor, k: int, seed: int
) -> torch.Tensor:
    n = int(embeddings.shape[0])
    gen = torch.Generator(device=embeddings.device)
    gen.manual_seed(seed)
    first = int(torch.randint(low=0, high=n, size=(1,), generator=gen).item())
    centers = [embeddings[first]]

    while len(centers) < k:
        stacked = torch.stack(centers, dim=0)
        sims = embeddings @ stacked.T
        best_existing = sims.max(dim=1).values
        next_idx = int(torch.argmin(best_existing).item())
        centers.append(embeddings[next_idx])
    return torch.stack(centers, dim=0)


def spherical_kmeans(
    embeddings: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = F.normalize(embeddings, dim=1, eps=1e-12)
    centers = init_centers_farthest(embeddings, k=k, seed=seed)
    assignments = torch.zeros((embeddings.shape[0],), dtype=torch.long, device=embeddings.device)

    for _ in range(iters):
        sims = embeddings @ centers.T
        new_assignments = torch.argmax(sims, dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments

        new_centers = []
        for cluster_id in range(k):
            mask = assignments == cluster_id
            if int(mask.sum().item()) == 0:
                hardest_idx = int(torch.argmin(sims.max(dim=1).values).item())
                center = embeddings[hardest_idx]
            else:
                center = F.normalize(embeddings[mask].mean(dim=0), dim=0, eps=1e-12)
            new_centers.append(center)
        centers = torch.stack(new_centers, dim=0)
    return assignments.cpu(), centers.cpu()


def choose_medoid_local_index(cluster_embeddings: torch.Tensor) -> tuple[int, float]:
    if int(cluster_embeddings.shape[0]) == 1:
        return 0, 1.0
    sims = cluster_embeddings @ cluster_embeddings.T
    mean_sim = sims.mean(dim=1)
    medoid_local = int(torch.argmax(mean_sim).item())
    return medoid_local, float(mean_sim[medoid_local].item())


def save_mean_preset(
    raw_embeddings: torch.Tensor,
    output_path: Path,
) -> torch.Tensor:
    normalized = F.normalize(raw_embeddings, dim=1, eps=1e-12)
    final_embedding = F.normalize(normalized.mean(dim=0), dim=0, eps=1e-12).cpu()
    torch.save(final_embedding, output_path)
    return final_embedding


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.preset_id}.pt"
    meta_path = output_dir / f"{args.preset_id}.json"

    extensions = parse_extensions(args.extensions)
    audio_files = discover_audio_files(
        audio_dir=args.audio_dir,
        audio_glob=args.audio_glob,
        audio_list=args.audio,
        extensions=extensions,
        max_files=args.max_files,
    )
    if not audio_files:
        raise RuntimeError("No audio files found.")

    codec = MioCodecModel.from_pretrained(args.model_id)
    codec = codec.eval().to(args.device)
    sample_rate = int(codec.config.sample_rate)

    all_embeddings: list[torch.Tensor] = []
    embedding_sources: list[str] = []
    used_files = 0
    used_segments = 0
    per_file_segments: dict[str, int] = {}

    with torch.inference_mode():
        for path in audio_files:
            waveform = load_audio(str(path), sample_rate=sample_rate)
            segments = split_waveform(
                waveform=waveform,
                sample_rate=sample_rate,
                segment_seconds=args.segment_seconds,
                hop_seconds=args.segment_hop_seconds,
                min_segment_seconds=args.min_segment_seconds,
            )
            if not segments:
                continue
            used_files += 1
            per_file_segments[str(path)] = 0

            for segment in segments:
                segment = segment.to(device=args.device, dtype=torch.float32)
                features = codec.encode(segment, return_content=False, return_global=True)
                embedding = features.global_embedding.detach().float().flatten().cpu()
                all_embeddings.append(embedding)
                embedding_sources.append(str(path))
                used_segments += 1
                per_file_segments[str(path)] += 1

                if args.max_segments > 0 and used_segments >= args.max_segments:
                    break

            print(f"[progress] {path.name}: segments={per_file_segments[str(path)]}")
            if args.max_segments > 0 and used_segments >= args.max_segments:
                break

    if not all_embeddings:
        raise RuntimeError("No usable segments found. Check audio quality and segment settings.")

    stacked_raw = torch.stack(all_embeddings, dim=0)
    stacked_filtered_norm, selected_indices = prefilter_embeddings(
        embeddings=stacked_raw,
        keep_ratio=args.keep_ratio,
        min_cosine=args.min_cosine,
        disable_outlier_removal=args.disable_outlier_removal,
    )
    filtered_raw = stacked_raw[selected_indices]
    selected_indices_list = selected_indices.tolist()
    filtered_sources = [embedding_sources[i] for i in selected_indices_list]

    cluster_outputs: list[dict[str, object]] = []

    if args.strategy == "mean":
        _ = save_mean_preset(filtered_raw, output_path)
        cluster_outputs.append(
            {
                "rank": 1,
                "cluster_id": 0,
                "size": int(filtered_raw.shape[0]),
                "path": str(output_path),
                "medoid_source": None,
                "medoid_similarity": None,
                "sources_count": len(set(filtered_sources)),
            }
        )
    else:
        if args.num_clusters > 0:
            k = min(args.num_clusters, int(stacked_filtered_norm.shape[0]))
        else:
            k = auto_cluster_count(
                num_items=int(stacked_filtered_norm.shape[0]), max_clusters=args.max_clusters
            )
        assignments, _ = spherical_kmeans(
            embeddings=stacked_filtered_norm, k=k, iters=args.cluster_iters, seed=args.seed
        )

        cluster_map: dict[int, list[int]] = {}
        for local_idx, cluster_id in enumerate(assignments.tolist()):
            cluster_map.setdefault(int(cluster_id), []).append(local_idx)

        clusters = sorted(cluster_map.items(), key=lambda item: len(item[1]), reverse=True)
        accepted_clusters: list[tuple[int, list[int]]] = [
            (cluster_id, members)
            for cluster_id, members in clusters
            if len(members) >= args.cluster_min_size
        ]
        if not accepted_clusters:
            accepted_clusters = [clusters[0]]

        for rank, (cluster_id, members_local) in enumerate(accepted_clusters, start=1):
            member_tensor_idx = torch.tensor(members_local, dtype=torch.long)
            cluster_embeddings = stacked_filtered_norm[member_tensor_idx]
            medoid_local_in_cluster, medoid_similarity = choose_medoid_local_index(cluster_embeddings)
            medoid_local_global = members_local[medoid_local_in_cluster]
            medoid_raw_global_idx = selected_indices_list[medoid_local_global]
            medoid_embedding = stacked_raw[medoid_raw_global_idx].cpu()
            medoid_source = embedding_sources[medoid_raw_global_idx]

            if rank == 1:
                primary_path = output_path
                torch.save(medoid_embedding, primary_path)
            if args.save_all_clusters:
                cluster_path = output_dir / f"{args.preset_id}__c{rank:02d}.pt"
                torch.save(medoid_embedding, cluster_path)
            else:
                cluster_path = output_path if rank == 1 else None

            cluster_sources = [filtered_sources[idx] for idx in members_local]
            cluster_outputs.append(
                {
                    "rank": rank,
                    "cluster_id": cluster_id,
                    "size": len(members_local),
                    "path": str(cluster_path) if cluster_path is not None else None,
                    "medoid_source": medoid_source,
                    "medoid_similarity": round(medoid_similarity, 6),
                    "sources_count": len(set(cluster_sources)),
                }
            )

    print(f"Saved primary preset to {output_path}")
    print(
        f"Summary: files={used_files}/{len(audio_files)} segments={used_segments} "
        f"filtered={int(filtered_raw.shape[0])} clusters={len(cluster_outputs)} strategy={args.strategy}"
    )
    for item in cluster_outputs:
        print(
            f"[cluster] rank={item['rank']} id={item['cluster_id']} size={item['size']} "
            f"sources={item['sources_count']} path={item['path']}"
        )

    if args.save_meta:
        meta = {
            "preset_id": args.preset_id,
            "model_id": args.model_id,
            "device": args.device,
            "sample_rate": sample_rate,
            "strategy": args.strategy,
            "audio_files_found": len(audio_files),
            "audio_files_used": used_files,
            "segments_used": used_segments,
            "segments_filtered": int(filtered_raw.shape[0]),
            "segment_seconds": args.segment_seconds,
            "segment_hop_seconds": args.segment_hop_seconds,
            "min_segment_seconds": args.min_segment_seconds,
            "keep_ratio": args.keep_ratio,
            "min_cosine": args.min_cosine,
            "disable_outlier_removal": args.disable_outlier_removal,
            "num_clusters": args.num_clusters,
            "max_clusters": args.max_clusters,
            "cluster_iters": args.cluster_iters,
            "cluster_min_size": args.cluster_min_size,
            "save_all_clusters": args.save_all_clusters,
            "seed": args.seed,
            "per_file_segments": per_file_segments,
            "primary_output_path": str(output_path),
            "clusters": cluster_outputs,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)
        print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
