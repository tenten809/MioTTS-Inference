import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from miocodec import MioCodecModel
from miocodec.util import load_audio

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from miotts_server.text import normalize_text
from miotts_server.token_parser import tokens_to_str


@dataclass
class ESDEntry:
    audio_relpath: str
    speaker: str
    lang: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LoRA training JSONL from ESD-style list and paired wav files."
    )
    parser.add_argument("--esd-list", required=True, help="Path to esd.list")
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Root directory containing wav files. Default: <esd-list-dir>/raw",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path")
    parser.add_argument(
        "--codec-model-id",
        default="Aratako/MioCodec-25Hz-44.1kHz-v2",
        help="MioCodec model id",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--min-audio-sec", type=float, default=0.0, help="Skip shorter audio")
    parser.add_argument("--max-audio-sec", type=float, default=30.0, help="Skip longer audio")
    parser.add_argument(
        "--normalize-japanese",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply same Japanese normalization as inference path",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip missing wav entries instead of raising error",
    )
    parser.add_argument("--print-every", type=int, default=20, help="Progress interval")
    return parser.parse_args()


def parse_esd_line(line: str) -> ESDEntry | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split("|")
    if len(parts) < 4:
        return None
    audio_relpath, speaker, lang = parts[0], parts[1], parts[2]
    text = "|".join(parts[3:]).strip()
    return ESDEntry(audio_relpath=audio_relpath, speaker=speaker, lang=lang, text=text)


def should_normalize_ja(lang: str) -> bool:
    token = lang.strip().lower()
    return token in {"jp", "ja", "jpn", "ja-jp"}


def main() -> None:
    args = parse_args()
    esd_path = Path(args.esd_list).expanduser().resolve()
    if not esd_path.exists():
        raise FileNotFoundError(f"esd.list not found: {esd_path}")

    if args.audio_root:
        audio_root = Path(args.audio_root).expanduser().resolve()
    else:
        audio_root = esd_path.parent / "raw"
        audio_root = audio_root.resolve()
    if not audio_root.exists():
        raise FileNotFoundError(f"audio root not found: {audio_root}")

    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codec = MioCodecModel.from_pretrained(args.codec_model_id).eval().to(args.device)
    sample_rate = int(codec.config.sample_rate)
    min_samples = int(max(0.0, args.min_audio_sec) * sample_rate)
    max_samples = int(max(0.0, args.max_audio_sec) * sample_rate) if args.max_audio_sec > 0 else 0

    written = 0
    skipped_missing = 0
    skipped_invalid = 0
    skipped_length = 0

    with esd_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out, torch.inference_mode():
        for line_idx, raw_line in enumerate(f_in, start=1):
            entry = parse_esd_line(raw_line)
            if entry is None:
                skipped_invalid += 1
                continue

            wav_path = Path(entry.audio_relpath)
            if not wav_path.is_absolute():
                wav_path = audio_root / wav_path
            wav_path = wav_path.resolve()

            if not wav_path.exists():
                if args.skip_missing_audio:
                    skipped_missing += 1
                    continue
                raise FileNotFoundError(f"Missing audio: {wav_path}")

            waveform = load_audio(str(wav_path), sample_rate=sample_rate)
            waveform = waveform.squeeze()
            if waveform.dim() != 1:
                waveform = waveform.flatten()
            if waveform.numel() < min_samples:
                skipped_length += 1
                continue
            if max_samples > 0 and waveform.numel() > max_samples:
                skipped_length += 1
                continue

            waveform = waveform.to(args.device)
            features = codec.encode(waveform, return_content=True, return_global=False)
            if features.content_token_indices is None:
                skipped_invalid += 1
                continue
            token_ids = features.content_token_indices.detach().cpu().tolist()
            token_text = tokens_to_str(token_ids)

            text = entry.text
            if args.normalize_japanese and should_normalize_ja(entry.lang):
                text = normalize_text(text)

            record = {
                "id": f"{line_idx}",
                "audio_path": str(wav_path),
                "speaker": entry.speaker,
                "lang": entry.lang,
                "text": text,
                "target": token_text,
                "num_tokens": len(token_ids),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(f"[progress] written={written} latest={wav_path.name}")
            if args.max_samples > 0 and written >= args.max_samples:
                break

    print(f"Saved dataset JSONL: {output_path}")
    print(
        "Summary: "
        f"written={written} "
        f"skipped_missing={skipped_missing} "
        f"skipped_invalid={skipped_invalid} "
        f"skipped_length={skipped_length}"
    )


if __name__ == "__main__":
    main()
