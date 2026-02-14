import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter for MioTTS token generation from JSONL dataset."
    )
    parser.add_argument("--base-model", required=True, help="Base model path or HF repo id")
    parser.add_argument("--train-jsonl", required=True, help="Train dataset JSONL path")
    parser.add_argument("--eval-jsonl", default=None, help="Eval dataset JSONL path")
    parser.add_argument("--output-dir", required=True, help="Adapter output dir")
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,out_proj",
        help="Comma-separated LoRA target module names",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="PyTorch attention backend for training",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "target" not in obj:
                raise ValueError(f"Missing text/target fields at {path}:{line_idx}")
            rows.append(obj)
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_ratio <= 0 or len(rows) < 10:
        return rows, []
    n_eval = max(1, int(len(rows) * eval_ratio))
    n_eval = min(n_eval, len(rows) - 1)
    idx = list(range(len(rows)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    eval_idx = set(idx[:n_eval])
    train_rows = [row for i, row in enumerate(rows) if i not in eval_idx]
    eval_rows = [row for i, row in enumerate(rows) if i in eval_idx]
    return train_rows, eval_rows


def build_chat_ids(
    tokenizer,
    text: str,
    target: str,
    max_length: int,
) -> dict[str, list[int]] | None:
    prompt_messages = [{"role": "user", "content": text}]
    full_messages = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": target},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )

    if len(full_ids) <= len(prompt_ids):
        return None
    if len(prompt_ids) >= max_length:
        return None
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "length": len(full_ids),
    }


class TokenizedJSONLDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        skipped = 0
        for row in rows:
            item = build_chat_ids(
                tokenizer=tokenizer,
                text=str(row["text"]),
                target=str(row["target"]),
                max_length=max_length,
            )
            if item is None:
                skipped += 1
                continue
            self.examples.append(item)
        if not self.examples:
            raise RuntimeError("No valid tokenized samples. Check dataset or max-length.")
        print(f"[dataset] kept={len(self.examples)} skipped={skipped}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


@dataclass
class CausalLMDataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def parse_target_modules(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def validate_target_modules(model, target_modules: list[str]) -> None:
    module_names = [name for name, _ in model.named_modules()]
    missing = []
    for target in target_modules:
        found = any(name.endswith(target) for name in module_names)
        if not found:
            missing.append(target)
    if missing:
        raise ValueError(
            "Target modules not found in model: "
            + ", ".join(missing)
            + ". Check --target-modules."
        )


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = Path(args.train_jsonl).expanduser().resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"train-jsonl not found: {train_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(train_path)
    if args.eval_jsonl:
        eval_rows = read_jsonl(Path(args.eval_jsonl).expanduser().resolve())
    else:
        train_rows, eval_rows = split_rows(train_rows, args.eval_ratio, args.seed)

    train_dataset = TokenizedJSONLDataset(train_rows, tokenizer=tokenizer, max_length=args.max_length)
    eval_dataset = (
        TokenizedJSONLDataset(eval_rows, tokenizer=tokenizer, max_length=args.max_length)
        if eval_rows
        else None
    )

    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    target_modules = parse_target_modules(args.target_modules)
    validate_target_modules(model, target_modules)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "peft is required. Install with: .\\.venv\\Scripts\\python.exe -m pip install peft"
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(f"Saved LoRA adapter to: {output_dir}")


if __name__ == "__main__":
    main()
