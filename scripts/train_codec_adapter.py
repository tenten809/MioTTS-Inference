import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from miocodec import MioCodecModel
from miocodec.util import load_audio
from safetensors.torch import save_file


@dataclass
class Sample:
    audio_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MioCodec adapter for voice quality/speaker adaptation "
            "while keeping content token semantics fixed."
        )
    )
    parser.add_argument("--esd-list", required=True, help="Path to esd.list")
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Root directory containing wav files. Default: <esd-list-dir>/raw",
    )
    parser.add_argument(
        "--codec-model-id",
        default="Aratako/MioCodec-25Hz-44.1kHz-v2",
        help="Base MioCodec model id",
    )
    parser.add_argument("--output-adapter", required=True, help="Output adapter safetensors path")
    parser.add_argument("--output-meta", default=None, help="Output metadata json path")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--min-audio-sec", type=float, default=1.0)
    parser.add_argument("--max-audio-sec", type=float, default=20.0)
    parser.add_argument("--train-global-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stft-loss-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def parse_esd_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split("|")
    if len(parts) < 1:
        return None
    return parts[0].strip()


def load_samples(esd_list: Path, audio_root: Path, max_samples: int) -> list[Sample]:
    samples: list[Sample] = []
    with esd_list.open("r", encoding="utf-8") as f:
        for raw in f:
            rel = parse_esd_line(raw)
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = audio_root / path
            path = path.resolve()
            if not path.exists():
                continue
            samples.append(Sample(audio_path=path))
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples


def split_samples(samples: list[Sample], eval_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if eval_ratio <= 0 or len(samples) < 10:
        return samples, []
    idx = list(range(len(samples)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_eval = max(1, int(len(samples) * eval_ratio))
    n_eval = min(n_eval, len(samples) - 1)
    eval_idx = set(idx[:n_eval])
    train = [s for i, s in enumerate(samples) if i not in eval_idx]
    evals = [s for i, s in enumerate(samples) if i in eval_idx]
    return train, evals


def set_requires_grad(module: torch.nn.Module | None, value: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = value


def freeze_content_path(model: MioCodecModel) -> None:
    freeze_names = [
        "ssl_feature_extractor",
        "local_encoder",
        "local_quantizer",
        "feature_decoder",
        "conv_downsample",
        "conv_upsample",
    ]
    for name in freeze_names:
        module = getattr(model, name, None)
        set_requires_grad(module, False)


def resolve_train_prefixes(model: MioCodecModel, train_global: bool, train_decoder: bool) -> list[str]:
    prefixes: list[str] = []
    if train_global and hasattr(model, "global_encoder"):
        set_requires_grad(model.global_encoder, True)
        prefixes.append("global_encoder.")
    else:
        set_requires_grad(getattr(model, "global_encoder", None), False)

    decoder_candidates = [
        "wave_prenet",
        "wave_decoder",
        "wave_prior_net",
        "wave_post_net",
        "istft_head",
        "mel_prenet",
        "mel_decoder",
        "mel_postnet",
        "mel_conv_upsample",
    ]
    for name in decoder_candidates:
        module = getattr(model, name, None)
        if module is None:
            continue
        set_requires_grad(module, train_decoder)
        if train_decoder:
            prefixes.append(f"{name}.")
    return prefixes


def align_audio(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.flatten()
    target = target.flatten()
    n = min(pred.numel(), target.numel())
    return pred[:n], target[:n]


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Multi-resolution STFT loss (log-mag + mag)
    losses = []
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        win = n_fft
        window = torch.hann_window(win, device=pred.device)
        pred_spec = torch.stft(
            pred, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True
        )
        tgt_spec = torch.stft(
            target, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True
        )
        pred_mag = pred_spec.abs().clamp_min(1e-7)
        tgt_mag = tgt_spec.abs().clamp_min(1e-7)
        losses.append(F.l1_loss(pred_mag, tgt_mag) + F.l1_loss(pred_mag.log(), tgt_mag.log()))
    return torch.stack(losses).mean()


def compute_step_loss(
    model: MioCodecModel,
    waveform: torch.Tensor,
    stft_weight: float,
) -> tuple[torch.Tensor, float, float]:
    waveform = waveform.squeeze()
    if waveform.dim() != 1:
        waveform = waveform.flatten()
    waveform = waveform.float()
    target_len = int(waveform.numel())

    padding = model._calculate_waveform_padding(target_len)
    local_ssl, global_ssl = model.forward_ssl_features(waveform.unsqueeze(0), padding=padding)

    with torch.no_grad():
        _, token_indices, _, _ = model.forward_content(local_ssl)
        if token_indices is None:
            raise RuntimeError("Content token extraction failed.")
        content_embeddings = model.decode_token_indices(token_indices)

    global_embeddings = model.forward_global(global_ssl)

    if not model.config.use_wave_decoder:
        raise RuntimeError("Current training script supports wave decoder models only.")
    stft_len = model._calculate_target_stft_length(target_len)
    pred_wave = model.forward_wave(content_embeddings, global_embeddings, stft_length=stft_len).squeeze(0)
    pred_wave, target_wave = align_audio(pred_wave, waveform)
    loss_wav = F.l1_loss(pred_wave, target_wave)
    loss_stft = mrstft_loss(pred_wave, target_wave)
    loss = loss_wav + stft_weight * loss_stft
    return loss, float(loss_wav.item()), float(loss_stft.item())


def filter_by_duration(
    samples: list[Sample],
    sample_rate: int,
    min_sec: float,
    max_sec: float,
) -> list[Sample]:
    min_samples = int(max(0.0, min_sec) * sample_rate)
    max_samples = int(max(0.0, max_sec) * sample_rate) if max_sec > 0 else 0
    kept: list[Sample] = []
    for s in samples:
        waveform = load_audio(str(s.audio_path), sample_rate=sample_rate).squeeze()
        if waveform.dim() != 1:
            waveform = waveform.flatten()
        n = int(waveform.numel())
        if n < min_samples:
            continue
        if max_samples > 0 and n > max_samples:
            continue
        kept.append(s)
    return kept


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    esd_path = Path(args.esd_list).expanduser().resolve()
    if not esd_path.exists():
        raise FileNotFoundError(f"esd.list not found: {esd_path}")
    audio_root = (
        Path(args.audio_root).expanduser().resolve()
        if args.audio_root
        else (esd_path.parent / "raw").resolve()
    )
    if not audio_root.exists():
        raise FileNotFoundError(f"audio root not found: {audio_root}")

    out_adapter = Path(args.output_adapter).expanduser().resolve()
    out_adapter.parent.mkdir(parents=True, exist_ok=True)
    out_meta = (
        Path(args.output_meta).expanduser().resolve()
        if args.output_meta
        else out_adapter.with_suffix(".json")
    )

    model = MioCodecModel.from_pretrained(args.codec_model_id).to(args.device)
    model.train()
    sample_rate = int(model.config.sample_rate)

    all_samples = load_samples(esd_path, audio_root, args.max_samples)
    if not all_samples:
        raise RuntimeError("No valid samples found.")

    all_samples = filter_by_duration(
        all_samples, sample_rate=sample_rate, min_sec=args.min_audio_sec, max_sec=args.max_audio_sec
    )
    if len(all_samples) < 2:
        raise RuntimeError("Not enough samples after duration filtering.")

    train_samples, eval_samples = split_samples(all_samples, args.eval_ratio, args.seed)
    print(
        f"[dataset] total={len(all_samples)} train={len(train_samples)} eval={len(eval_samples)} "
        f"sample_rate={sample_rate}"
    )

    freeze_content_path(model)
    train_prefixes = resolve_train_prefixes(
        model=model,
        train_global=args.train_global_encoder,
        train_decoder=args.train_decoder,
    )
    if not train_prefixes:
        raise RuntimeError("No trainable module selected.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    print(f"[trainable] params={total_trainable} prefixes={train_prefixes}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for sample in train_samples:
            waveform = load_audio(str(sample.audio_path), sample_rate=sample_rate).to(args.device)
            loss, loss_wav, loss_stft = compute_step_loss(model, waveform, stft_weight=args.stft_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            global_step += 1
            train_count += 1
            train_loss_sum += float(loss.item())
            if args.print_every > 0 and global_step % args.print_every == 0:
                print(
                    f"[step {global_step}] epoch={epoch} "
                    f"loss={loss.item():.5f} wav={loss_wav:.5f} stft={loss_stft:.5f}"
                )

        train_loss = train_loss_sum / max(1, train_count)

        eval_loss = None
        if eval_samples:
            model.eval()
            eval_sum = 0.0
            with torch.no_grad():
                for sample in eval_samples:
                    waveform = load_audio(str(sample.audio_path), sample_rate=sample_rate).to(args.device)
                    loss, _, _ = compute_step_loss(model, waveform, stft_weight=args.stft_loss_weight)
                    eval_sum += float(loss.item())
            eval_loss = eval_sum / len(eval_samples)

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": args.lr,
            "step": global_step,
        }
        if eval_loss is not None:
            row["eval_loss"] = round(eval_loss, 6)
            print(f"[epoch {epoch}] train_loss={train_loss:.6f} eval_loss={eval_loss:.6f}")
        else:
            print(f"[epoch {epoch}] train_loss={train_loss:.6f}")
        history.append(row)

    adapter_state: dict[str, torch.Tensor] = {}
    full_state = model.state_dict()
    for key, value in full_state.items():
        if any(key.startswith(prefix) for prefix in train_prefixes):
            adapter_state[key] = value.detach().cpu().contiguous()
    if not adapter_state:
        raise RuntimeError("No adapter tensors collected.")

    save_file(adapter_state, str(out_adapter))
    meta = {
        "base_codec_model_id": args.codec_model_id,
        "adapter_path": str(out_adapter),
        "sample_rate": sample_rate,
        "train_global_encoder": args.train_global_encoder,
        "train_decoder": args.train_decoder,
        "train_prefixes": train_prefixes,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stft_loss_weight": args.stft_loss_weight,
        "grad_clip": args.grad_clip,
        "dataset_total": len(all_samples),
        "dataset_train": len(train_samples),
        "dataset_eval": len(eval_samples),
        "history": history,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved codec adapter: {out_adapter}")
    print(f"Saved training metadata: {out_meta}")


if __name__ == "__main__":
    main()
