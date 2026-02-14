import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model weights.")
    parser.add_argument("--base-model", required=True, help="Base model path or HF repo id")
    parser.add_argument("--adapter-dir", required=True, help="Trained LoRA adapter directory")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-shard-size", default="4GB", help="save_pretrained shard size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype_map[args.dtype]
    )
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "peft is required. Install with: .\\.venv\\Scripts\\python.exe -m pip install peft"
        ) from exc

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
