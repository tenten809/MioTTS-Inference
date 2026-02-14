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
        description="Generate a single MioTTS preset embedding from many reference audio files."
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
        "--model-id", default="Aratako/MioCodec-25Hz-24kHz", help="MioCodec model id."
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
        "--keep-ratio",
        type=float,
        default=0.8,
        help="Top ratio to keep by cosine similarity when outlier removal is enabled.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=None,
        help="Optional minimum cosine similarity to centroid when outlier removal is enabled.",
    )
    parser.add_argument(
        "--disable-outlier-removal",
        action="store_true",
        help="Disable outlier removal and average all segment embeddings.",
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


def aggregate_embeddings(
    embeddings: torch.Tensor,
    keep_ratio: float,
    min_cosine: float | None,
    disable_outlier_removal: bool,
) -> tuple[torch.Tensor, int]:
    embeddings = F.normalize(embeddings, dim=1)
    selected = embeddings

    if not disable_outlier_removal and embeddings.shape[0] >= 3:
        centroid = F.normalize(embeddings.mean(dim=0), dim=0)
        similarities = embeddings @ centroid

        keep_k = max(1, int(math.ceil(embeddings.shape[0] * keep_ratio)))
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
        selected = embeddings[mask]

    final_embedding = F.normalize(selected.mean(dim=0), dim=0).cpu()
    return final_embedding, int(selected.shape[0])


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

    embeddings: list[torch.Tensor] = []
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
                embeddings.append(embedding)
                used_segments += 1
                per_file_segments[str(path)] += 1

                if args.max_segments > 0 and used_segments >= args.max_segments:
                    break

            print(f"[progress] {path.name}: segments={per_file_segments[str(path)]}")
            if args.max_segments > 0 and used_segments >= args.max_segments:
                break

    if not embeddings:
        raise RuntimeError("No usable segments found. Check audio quality and segment settings.")

    stacked = torch.stack(embeddings, dim=0)
    final_embedding, selected_count = aggregate_embeddings(
        embeddings=stacked,
        keep_ratio=args.keep_ratio,
        min_cosine=args.min_cosine,
        disable_outlier_removal=args.disable_outlier_removal,
    )
    torch.save(final_embedding, output_path)

    print(f"Saved preset embedding to {output_path}")
    print(
        f"Summary: files={used_files}/{len(audio_files)} "
        f"segments={used_segments} selected~={selected_count}"
    )

    if args.save_meta:
        meta = {
            "preset_id": args.preset_id,
            "model_id": args.model_id,
            "device": args.device,
            "sample_rate": sample_rate,
            "audio_files_found": len(audio_files),
            "audio_files_used": used_files,
            "segments_used": used_segments,
            "segment_seconds": args.segment_seconds,
            "segment_hop_seconds": args.segment_hop_seconds,
            "min_segment_seconds": args.min_segment_seconds,
            "keep_ratio": args.keep_ratio,
            "min_cosine": args.min_cosine,
            "disable_outlier_removal": args.disable_outlier_removal,
            "selected_segments_estimate": selected_count,
            "per_file_segments": per_file_segments,
            "output_embedding_path": str(output_path),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)
        print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
