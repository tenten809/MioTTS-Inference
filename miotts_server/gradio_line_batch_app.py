from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

DEFAULT_TTS_API_BASE = os.getenv("MIOTTS_API_BASE", "http://localhost:8001")
DEFAULT_LLM_API_BASE = os.getenv("MIOTTS_LLM_BASE", "http://localhost:8000")
DEFAULT_PRESETS_DIR = Path(os.getenv("MIOTTS_PRESETS_DIR", "presets")).expanduser()
DEFAULT_LORAS_DIR = Path(os.getenv("MIOTTS_LORAS_DIR", "loras")).expanduser()
NONE_LORA_VALUE = "__none__"
PROCESS_LOCK = threading.Lock()
UNKNOWN_LORA_STATE = {"id": "__unknown__", "scale": -1.0}
UI_CSS = """
#tts-rows-table table {
  table-layout: fixed;
  width: 100%;
}
#tts-rows-table th,
#tts-rows-table td {
  white-space: nowrap !important;
  overflow: hidden;
  text-overflow: ellipsis;
}
#row-lora,
#row-preset {
  min-height: 84px;
}
#row-lora *,
#row-preset * {
  white-space: nowrap !important;
}
#long-text-input textarea {
  overflow-y: auto !important;
  resize: none !important;
}
"""


def _norm_base(url: str) -> str:
    return url.rstrip("/")


def _tts_url(api_base: str) -> str:
    return f"{_norm_base(api_base)}/v1/tts"


def _lora_url(llm_base: str) -> str:
    return f"{_norm_base(llm_base)}/lora-adapters"


def _decode_wav_bytes(data: bytes) -> tuple[int, np.ndarray]:
    with io.BytesIO(data) as buff:
        audio, sr = sf.read(buff, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return int(sr), audio


def _encode_wav_bytes(sr: int, audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    with io.BytesIO() as buff:
        sf.write(buff, audio, sr, format="WAV")
        return buff.getvalue()


def _audio_to_b64(sr: int, audio: np.ndarray) -> str:
    return base64.b64encode(_encode_wav_bytes(sr, audio)).decode("ascii")


def _audio_from_b64(audio_b64: str) -> tuple[int, np.ndarray]:
    return _decode_wav_bytes(base64.b64decode(audio_b64))


def _split_text_to_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    pattern = re.compile(r".+?(?:[。！？!?]+|$)")

    def _split_by_sentence(chunk: str) -> list[str]:
        out: list[str] = []
        chunk = chunk.strip()
        if not chunk:
            return out
        for part in pattern.findall(chunk):
            sentence = part.strip()
            if sentence:
                out.append(sentence)
        return out

    def _split_block_with_quotes(block: str) -> list[str]:
        out: list[str] = []
        i = 0
        n = len(block)
        while i < n:
            q_start = block.find("「", i)
            if q_start < 0:
                out.extend(_split_by_sentence(block[i:]))
                break
            out.extend(_split_by_sentence(block[i:q_start]))
            q_end = block.find("」", q_start + 1)
            if q_end < 0:
                out.extend(_split_by_sentence(block[q_start:]))
                break
            quoted = block[q_start : q_end + 1].strip()
            if quoted:
                if len(quoted) <= 50:
                    out.append(quoted)
                else:
                    q_parts = _split_by_sentence(quoted)
                    if len(q_parts) >= 2 and q_parts[-1] == "」":
                        q_parts[-2] = q_parts[-2] + "」"
                        q_parts.pop()
                    out.extend(q_parts)
            i = q_end + 1
        return out

    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for part in _split_block_with_quotes(block):
            sentence = part.strip()
            if sentence:
                lines.append(sentence)
    return lines


def _fetch_presets_from_dir(presets_dir: Path | str) -> list[str]:
    try:
        base = Path(presets_dir).expanduser().resolve()
    except Exception:
        return []
    if not base.exists():
        return []
    presets: list[str] = []
    for path in base.iterdir():
        if path.is_file() and path.suffix.lower() in {".pt", ".npz"}:
            presets.append(path.stem)
    return sorted(set(presets))


def _fetch_lora_adapters(llm_base: str) -> list[dict[str, Any]]:
    local_items = _scan_local_loras(DEFAULT_LORAS_DIR)
    loaded_by_norm_path: dict[str, dict[str, Any]] = {}
    try:
        res = httpx.get(_lora_url(llm_base), timeout=5.0)
        res.raise_for_status()
        data = res.json()
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "id" in item:
                    raw_path = str(item.get("path", ""))
                    if not raw_path:
                        continue
                    loaded_by_norm_path[_norm_path(raw_path)] = {
                        "id": int(item["id"]),
                        "path": raw_path,
                        "scale": float(item.get("scale", 0.0)),
                    }
    except Exception:
        loaded_by_norm_path = {}

    out: list[dict[str, Any]] = []
    for item in local_items:
        norm_local = _norm_path(item["path"])
        loaded = loaded_by_norm_path.get(norm_local)
        out.append(
            {
                "id": int(loaded["id"]) if loaded else None,
                "path": item["path"],
                "display": item["display"],
                "scale": float(loaded["scale"]) if loaded else 0.0,
            }
        )
    return out


def _scan_local_loras(loras_dir: Path | str) -> list[dict[str, str]]:
    try:
        root = Path(loras_dir).expanduser().resolve()
    except Exception:
        return []
    if not root.exists():
        return []

    items: list[dict[str, str]] = []
    for path in root.rglob("*.gguf"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix().replace("/", "\\")
        display = "\\" + rel
        items.append({"display": display, "path": str(path.resolve())})
    items.sort(key=lambda x: x["display"].lower())
    return items


def _norm_path(path: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(path).expanduser().resolve())))
    except Exception:
        return os.path.normcase(os.path.normpath(path))


def _lora_choices(adapters: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = [("none", NONE_LORA_VALUE)]
    for item in adapters:
        display = str(item.get("display") or "")
        if not display:
            continue
        lid = item.get("id")
        if lid is None:
            label = f"{display} (not loaded)"
        else:
            label = f"{display} (id:{int(lid)})"
        choices.append((label, f"path:{display}"))
    return choices


def _lora_key_from_value(value: str | None) -> str | None:
    if not value or value == NONE_LORA_VALUE:
        return None
    if value.startswith("path:"):
        return value.split(":", 1)[1]
    return None


def _parse_lora_key_from_table_cell(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "(none)", "__none__"}:
        return None
    if text.startswith("path:"):
        text = text.split(":", 1)[1].strip()
    text = re.sub(r"\s+\((id:\d+|not loaded|missing)\)\s*$", "", text, flags=re.IGNORECASE)
    return text or None


def _lora_value_from_key(lora_key: str | None) -> str:
    if not lora_key:
        return NONE_LORA_VALUE
    return f"path:{lora_key}"


def _lora_label(lora_key: str | None, adapters: list[dict[str, Any]]) -> str:
    if not lora_key:
        return "none"
    for item in adapters:
        if str(item.get("display")) == lora_key:
            lid = item.get("id")
            if lid is None:
                return f"{lora_key} (not loaded)"
            return f"{lora_key} (id:{int(lid)})"
    return f"{lora_key} (missing)"


def _resolve_lora_id(adapters: list[dict[str, Any]], lora_key: str | None) -> int | None:
    if not lora_key:
        return None
    for item in adapters:
        if str(item.get("display")) == lora_key:
            lid = item.get("id")
            if lid is None:
                raise ValueError(
                    f"Selected LoRA is not loaded in llama-server: {lora_key}. "
                    "Restart llama-server with --lora (multiple) and --lora-init-without-apply."
                )
            return int(lid)
    raise ValueError(f"Selected LoRA not found in .\\loras: {lora_key}")


def _rows_table(rows: list[dict[str, Any]], adapters: list[dict[str, Any]]) -> list[list[Any]]:
    table: list[list[Any]] = []
    for row in rows:
        table.append(
            [
                row["idx"],
                row["text"],
                _lora_label(row.get("lora_key"), adapters),
                row.get("preset_id") or "",
                row.get("status", "pending"),
                "yes" if row.get("audio_b64") else "no",
            ]
        )
    return table


def _global_cfg_signature(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
) -> str:
    payload = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "best_of_n_enabled": best_of_n_enabled,
        "best_of_n_n": best_of_n_n,
        "best_of_n_language": best_of_n_language,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _row_cache_key(
    row: dict[str, Any],
    global_sig: str,
    lora_scale: float,
) -> str:
    payload = {
        "text": row["text"],
        "preset_id": row.get("preset_id"),
        "lora_key": row.get("lora_key"),
        "lora_scale": round(float(lora_scale), 4),
        "global_sig": global_sig,
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _apply_lora(
    llm_base: str,
    adapters: list[dict[str, Any]],
    target_lora_id: int | None,
    lora_scale: float,
) -> None:
    if not adapters:
        return
    payload = []
    for item in adapters:
        lid = item.get("id")
        if lid is None:
            continue
        aid = int(lid)
        scale = float(lora_scale) if target_lora_id is not None and aid == target_lora_id else 0.0
        payload.append({"id": aid, "scale": scale})
    if not payload:
        return
    res = httpx.post(_lora_url(llm_base), json=payload, timeout=10.0)
    res.raise_for_status()


def _call_tts(
    api_base: str,
    row: dict[str, Any],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
) -> tuple[int, np.ndarray]:
    preset_id = row.get("preset_id")
    if not preset_id:
        raise ValueError("preset is required for each row.")

    llm_payload: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    if int(top_k) > 0:
        llm_payload["top_k"] = int(top_k)

    payload: dict[str, Any] = {
        "text": row["text"],
        "reference": {"type": "preset", "preset_id": preset_id},
        "llm": llm_payload,
        "output": {"format": "base64"},
    }
    if best_of_n_enabled:
        payload["best_of_n"] = {
            "enabled": True,
            "n": int(best_of_n_n),
            "language": best_of_n_language,
        }

    res = httpx.post(_tts_url(api_base), json=payload, timeout=240.0)
    res.raise_for_status()
    data = res.json()
    audio_b64 = data.get("audio")
    if not audio_b64:
        raise RuntimeError("No audio in /v1/tts response.")
    return _audio_from_b64(audio_b64)


def _format_tts_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else "?"
        detail = ""
        if exc.response is not None:
            try:
                payload = exc.response.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("detail", "")).strip()
            except Exception:
                detail = ""
            if not detail:
                try:
                    detail = exc.response.text.strip()
                except Exception:
                    detail = ""
        if len(detail) > 280:
            detail = detail[:280] + "..."
        return f"http {status}: {detail or str(exc)}"
    return str(exc)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    n_src = len(audio)
    n_dst = max(1, int(round(n_src * float(dst_sr) / float(src_sr))))
    x_src = np.linspace(0.0, 1.0, num=n_src, endpoint=True, dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=True, dtype=np.float32)
    return np.interp(x_dst, x_src, audio).astype(np.float32)


def _refresh_refs(llm_base: str):
    presets = _fetch_presets_from_dir(DEFAULT_PRESETS_DIR)
    adapters = _fetch_lora_adapters(llm_base)
    preset_default = presets[0] if presets else None
    loaded_count = sum(1 for x in adapters if x.get("id") is not None)
    return (
        presets,
        adapters,
        gr.update(choices=presets, value=preset_default),
        gr.update(choices=presets, value=preset_default),
        gr.update(choices=_lora_choices(adapters), value=NONE_LORA_VALUE),
        gr.update(choices=_lora_choices(adapters), value=NONE_LORA_VALUE),
        f"presets={len(presets)} from {DEFAULT_PRESETS_DIR}, loras={len(adapters)} from {DEFAULT_LORAS_DIR} (loaded={loaded_count})",
        UNKNOWN_LORA_STATE,
    )


def _split_into_rows(
    long_text: str,
    default_preset: str | None,
    default_lora_value: str,
    adapters: list[dict[str, Any]],
):
    lines = _split_text_to_lines(long_text)
    default_lora_key = _lora_key_from_value(default_lora_value)
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        rows.append(
            {
                "idx": i,
                "text": line,
                "lora_key": default_lora_key,
                "preset_id": default_preset,
                "status": "pending",
                "audio_b64": None,
                "cache_key": None,
                "sample_rate": None,
            }
        )
    selected = 1 if rows else 0
    selected_text = rows[0]["text"] if rows else ""
    return (
        rows,
        _rows_table(rows, adapters),
        selected,
        selected_text,
        f"split lines: {len(rows)}",
    )


def _on_table_select(evt: gr.SelectData):
    idx: Any = evt.index
    if isinstance(idx, (tuple, list)):
        if not idx:
            return 0
        idx = idx[0]
        if isinstance(idx, (tuple, list)):
            if not idx:
                return 0
            idx = idx[0]
    try:
        return int(idx) + 1
    except Exception:
        return 0


def _load_row_settings(
    selected_row: int,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    presets: list[str],
):
    if not rows:
        return "", gr.update(choices=_lora_choices(adapters), value=NONE_LORA_VALUE), gr.update(
            choices=presets, value=(presets[0] if presets else None)
        )
    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]
    return (
        row["text"],
        gr.update(choices=_lora_choices(adapters), value=_lora_value_from_key(row.get("lora_key"))),
        gr.update(choices=presets, value=row.get("preset_id")),
    )


def _apply_row_settings(
    selected_row: int,
    row_lora_value: str,
    row_preset: str,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
):
    if not rows:
        return rows, _rows_table(rows, adapters), "no rows"
    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]
    lora_key = _lora_key_from_value(row_lora_value)
    changed = (row.get("lora_key") != lora_key) or (row.get("preset_id") != row_preset)
    row["lora_key"] = lora_key
    row["preset_id"] = row_preset
    if changed:
        row["cache_key"] = None
        row["audio_b64"] = None
        row["sample_rate"] = None
        row["status"] = "pending"
    return rows, _rows_table(rows, adapters), f"row {row['idx']} updated"


def _apply_table_edits(
    table_value: Any,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
):
    if table_value is None:
        return rows, _rows_table(rows, adapters), "table edit: no data"
    if hasattr(table_value, "values"):
        table_rows = table_value.values.tolist()
    elif isinstance(table_value, list):
        table_rows = table_value
    else:
        return rows, _rows_table(rows, adapters), "table edit: unsupported payload"

    old_by_idx: dict[int, dict[str, Any]] = {int(r["idx"]): r for r in rows if "idx" in r}
    new_rows: list[dict[str, Any]] = []
    changed_count = 0
    removed_count = max(0, len(rows) - len(table_rows))

    for new_idx, cells in enumerate(table_rows, start=1):
        if not isinstance(cells, (list, tuple)):
            continue
        old_idx = None
        if len(cells) > 0:
            try:
                old_idx = int(float(cells[0]))
            except Exception:
                old_idx = None
        old = old_by_idx.get(old_idx) if old_idx is not None else (rows[new_idx - 1] if new_idx - 1 < len(rows) else None)

        text = str(cells[1] if len(cells) > 1 else "")
        lora_key = _parse_lora_key_from_table_cell(cells[2] if len(cells) > 2 else None)
        preset_id = str(cells[3] if len(cells) > 3 else "").strip()

        row = {
            "idx": new_idx,
            "text": text,
            "lora_key": lora_key,
            "preset_id": preset_id,
            "status": "pending",
            "audio_b64": None,
            "cache_key": None,
            "sample_rate": None,
        }
        if old is not None:
            same = (
                old.get("text") == text
                and old.get("lora_key") == lora_key
                and str(old.get("preset_id") or "") == preset_id
            )
            if same:
                row["status"] = old.get("status", "pending")
                row["audio_b64"] = old.get("audio_b64")
                row["cache_key"] = old.get("cache_key")
                row["sample_rate"] = old.get("sample_rate")
            else:
                changed_count += 1
        else:
            changed_count += 1
        new_rows.append(row)

    log_msg = f"table synced: rows={len(new_rows)}, changed={changed_count}, removed={removed_count}"
    return new_rows, _rows_table(new_rows, adapters), log_msg


def _insert_row_below(
    selected_row: int,
    row_lora_value: str,
    row_preset: str,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    presets: list[str],
):
    insert_idx = 0
    if rows:
        insert_idx = max(1, min(int(selected_row), len(rows)))
    lora_key = _lora_key_from_value(row_lora_value)
    preset = (row_preset or "").strip() or (presets[0] if presets else "")

    new_row = {
        "idx": 0,
        "text": "",
        "lora_key": lora_key,
        "preset_id": preset,
        "status": "pending",
        "audio_b64": None,
        "cache_key": None,
        "sample_rate": None,
    }
    rows.insert(insert_idx, new_row)
    for i, row in enumerate(rows, start=1):
        row["idx"] = i

    selected = insert_idx + 1
    return (
        rows,
        _rows_table(rows, adapters),
        selected,
        "",
        gr.update(choices=_lora_choices(adapters), value=_lora_value_from_key(new_row.get("lora_key"))),
        gr.update(choices=presets, value=new_row.get("preset_id")),
        f"row {selected} inserted",
    )


def _delete_selected_row(
    selected_row: int,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    presets: list[str],
):
    if not rows:
        return (
            rows,
            _rows_table(rows, adapters),
            0,
            "",
            gr.update(choices=_lora_choices(adapters), value=NONE_LORA_VALUE),
            gr.update(choices=presets, value=(presets[0] if presets else None)),
            "no rows",
        )
    idx = max(1, min(int(selected_row), len(rows))) - 1
    removed = rows.pop(idx)
    for i, row in enumerate(rows, start=1):
        row["idx"] = i
    if not rows:
        return (
            rows,
            _rows_table(rows, adapters),
            0,
            "",
            gr.update(choices=_lora_choices(adapters), value=NONE_LORA_VALUE),
            gr.update(choices=presets, value=(presets[0] if presets else None)),
            f"row {removed['idx']} deleted",
        )
    new_selected = max(1, min(idx + 1, len(rows)))
    row = rows[new_selected - 1]
    return (
        rows,
        _rows_table(rows, adapters),
        new_selected,
        row["text"],
        gr.update(choices=_lora_choices(adapters), value=_lora_value_from_key(row.get("lora_key"))),
        gr.update(choices=presets, value=row.get("preset_id")),
        f"row {removed['idx']} deleted",
    )


def _clear_cache(rows: list[dict[str, Any]], adapters: list[dict[str, Any]]):
    for row in rows:
        row["cache_key"] = None
        row["audio_b64"] = None
        row["sample_rate"] = None
        row["status"] = "pending"
    return rows, _rows_table(rows, adapters), "cache cleared"


def _process_selected_row(
    api_base: str,
    llm_base: str,
    selected_row: int,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    current_lora_state: dict[str, Any],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
    lora_scale: float,
):
    if not rows:
        return rows, _rows_table(rows, adapters), None, current_lora_state, "no rows"
    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]

    with PROCESS_LOCK:
        global_sig = _global_cfg_signature(
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            max_tokens=int(max_tokens),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of_n_enabled=best_of_n_enabled,
            best_of_n_n=int(best_of_n_n),
            best_of_n_language=best_of_n_language,
        )
        cache_key = _row_cache_key(row=row, global_sig=global_sig, lora_scale=lora_scale)
        if row.get("audio_b64") and row.get("cache_key") == cache_key:
            row["status"] = "cached"
            sr, audio = _audio_from_b64(row["audio_b64"])
            msg = f"row {row['idx']}: reused cached audio"
            return rows, _rows_table(rows, adapters), (sr, audio), current_lora_state, msg

        target_lora_key = row.get("lora_key")
        lora_warn = ""
        try:
            target_lora = _resolve_lora_id(adapters=adapters, lora_key=target_lora_key)
        except ValueError as exc:
            target_lora = None
            lora_warn = f" [warn: {exc} -> fallback none]"
        active_lora = current_lora_state.get("id")
        active_scale = float(current_lora_state.get("scale", 0.0))
        target_scale = float(lora_scale) if target_lora is not None else 0.0
        if active_lora != target_lora or abs(active_scale - target_scale) > 1e-9:
            _apply_lora(llm_base=llm_base, adapters=adapters, target_lora_id=target_lora, lora_scale=lora_scale)
            current_lora_state = {"id": target_lora, "scale": target_scale}

        try:
            sr, audio = _call_tts(
                api_base=api_base,
                row=row,
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                max_tokens=int(max_tokens),
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                best_of_n_enabled=best_of_n_enabled,
                best_of_n_n=int(best_of_n_n),
                best_of_n_language=best_of_n_language,
            )
        except Exception as exc:
            row["status"] = "error"
            row["audio_b64"] = None
            row["sample_rate"] = None
            row["cache_key"] = None
            msg = f"row {row['idx']}: tts failed - {_format_tts_error(exc)}{lora_warn}"
            return rows, _rows_table(rows, adapters), None, current_lora_state, msg
        row["audio_b64"] = _audio_to_b64(sr, audio)
        row["sample_rate"] = sr
        row["cache_key"] = cache_key
        row["status"] = "done"
        msg = f"row {row['idx']}: synthesized{lora_warn}"
        return rows, _rows_table(rows, adapters), (sr, audio), current_lora_state, msg


def _process_all_rows(
    api_base: str,
    llm_base: str,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    current_lora_state: dict[str, Any],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
    lora_scale: float,
    silence_sec: float,
):
    if not rows:
        return rows, _rows_table(rows, adapters), None, None, current_lora_state, "no rows"

    with PROCESS_LOCK:
        global_sig = _global_cfg_signature(
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            max_tokens=int(max_tokens),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of_n_enabled=best_of_n_enabled,
            best_of_n_n=int(best_of_n_n),
            best_of_n_language=best_of_n_language,
        )

        missing_by_lora: dict[str | None, list[int]] = {}
        for i, row in enumerate(rows):
            cache_key = _row_cache_key(row=row, global_sig=global_sig, lora_scale=lora_scale)
            if row.get("audio_b64") and row.get("cache_key") == cache_key:
                row["status"] = "cached"
                continue
            row["status"] = "pending"
            missing_by_lora.setdefault(row.get("lora_key"), []).append(i)

        total_new = sum(len(v) for v in missing_by_lora.values())
        logs = [f"rows={len(rows)}, synth_new={total_new}, groups={len(missing_by_lora)}"]

        for lora_key, indices in missing_by_lora.items():
            try:
                lora_id = _resolve_lora_id(adapters=adapters, lora_key=lora_key)
            except ValueError as exc:
                lora_id = None
                logs.append(f"{exc} -> fallback none")
            active_lora = current_lora_state.get("id")
            active_scale = float(current_lora_state.get("scale", 0.0))
            target_scale = float(lora_scale) if lora_id is not None else 0.0
            if active_lora != lora_id or abs(active_scale - target_scale) > 1e-9:
                _apply_lora(llm_base=llm_base, adapters=adapters, target_lora_id=lora_id, lora_scale=lora_scale)
                current_lora_state = {"id": lora_id, "scale": target_scale}
            logs.append(f"lora={lora_key or 'none'} rows={len(indices)}")

            for i in indices:
                row = rows[i]
                try:
                    sr, audio = _call_tts(
                        api_base=api_base,
                        row=row,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=int(top_k),
                        max_tokens=int(max_tokens),
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        best_of_n_enabled=best_of_n_enabled,
                        best_of_n_n=int(best_of_n_n),
                        best_of_n_language=best_of_n_language,
                    )
                except Exception as exc:
                    row["status"] = "error"
                    row["audio_b64"] = None
                    row["sample_rate"] = None
                    row["cache_key"] = None
                    logs.append(f"row {row['idx']}: tts failed - {_format_tts_error(exc)}")
                    continue
                row["audio_b64"] = _audio_to_b64(sr, audio)
                row["sample_rate"] = sr
                row["cache_key"] = _row_cache_key(row=row, global_sig=global_sig, lora_scale=lora_scale)
                row["status"] = "done"

        combined_chunks: list[np.ndarray] = []
        out_sr = None
        for i, row in enumerate(rows):
            if not row.get("audio_b64"):
                logs.append(f"row {row['idx']}: missing audio")
                return (
                    rows,
                    _rows_table(rows, adapters),
                    None,
                    None,
                    current_lora_state,
                    "\n".join(logs),
                )
            sr, audio = _audio_from_b64(row["audio_b64"])
            if out_sr is None:
                out_sr = sr
            elif sr != out_sr:
                audio = _resample_linear(audio, src_sr=sr, dst_sr=out_sr)
            combined_chunks.append(audio.astype(np.float32, copy=False))
            if i < len(rows) - 1 and silence_sec > 0:
                silence_len = int(round(float(silence_sec) * float(out_sr)))
                if silence_len > 0:
                    combined_chunks.append(np.zeros((silence_len,), dtype=np.float32))

        if out_sr is None:
            raise RuntimeError("No audio generated.")
        combined_audio = np.concatenate(combined_chunks, axis=0)
        out_dir = Path("outputs/line_tts").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"tts_combined_{ts}.wav"
        sf.write(str(out_path), combined_audio, out_sr, format="WAV")
        logs.append(f"saved={out_path}")

        return (
            rows,
            _rows_table(rows, adapters),
            (out_sr, combined_audio),
            str(out_path),
            current_lora_state,
            "\n".join(logs),
        )


def build_app() -> gr.Blocks:
    presets = _fetch_presets_from_dir(DEFAULT_PRESETS_DIR)
    adapters = _fetch_lora_adapters(DEFAULT_LLM_API_BASE)
    preset_default = presets[0] if presets else None
    default_lora_value = NONE_LORA_VALUE

    with gr.Blocks(title="MioTTS Line Batch UI") as demo:
        gr.Markdown("# MioTTS Line Batch UI")

        rows_state = gr.State([])
        adapters_state = gr.State(adapters)
        presets_state = gr.State(presets)
        current_lora_state = gr.State(dict(UNKNOWN_LORA_STATE))

        with gr.Row():
            tts_api_base = gr.Textbox(
                label="TTS API Base",
                value=DEFAULT_TTS_API_BASE,
                placeholder="http://localhost:8001",
            )
            llm_api_base = gr.Textbox(
                label="LLM API Base (for /lora-adapters)",
                value=DEFAULT_LLM_API_BASE,
                placeholder="http://localhost:8000",
            )
            refresh_btn = gr.Button("Refresh Presets/Loras")

        refresh_msg = gr.Markdown()

        with gr.Row():
            long_text = gr.Textbox(
                label="Long Text Input",
                lines=7,
                max_lines=7,
                placeholder="Paste long text. It will be split into lines by sentence.",
                elem_id="long-text-input",
            )
        with gr.Row():
            default_preset = gr.Dropdown(
                label="Default Preset For New Rows",
                choices=presets,
                value=preset_default,
                allow_custom_value=True,
            )
            default_lora = gr.Dropdown(
                label=r"Default LoRA For New Rows (from .\loras)",
                choices=_lora_choices(adapters),
                value=default_lora_value,
            )
            split_btn = gr.Button("Split Into Lines")
            clear_cache_btn = gr.Button("Clear Audio Cache")

        line_table = gr.Dataframe(
            headers=["idx", "text", "lora", "preset", "status", "cached"],
            datatype=["number", "str", "str", "str", "str", "str"],
            value=[],
            wrap=False,
            interactive=True,
            max_height=840,
            column_widths=["5%", "42%", "24%", "16%", "6%", "7%"],
            label="TTS Rows (scrollable)",
            elem_id="tts-rows-table",
        )

        with gr.Row():
            selected_row = gr.Number(label="Selected Row (1-based)", value=0, precision=0)
            row_text = gr.Textbox(label="Selected Text", interactive=False)
            insert_row_btn = gr.Button("Insert Row Below", variant="secondary")
            delete_row_btn = gr.Button("Delete Selected Row", variant="secondary")
        with gr.Row():
            row_lora = gr.Dropdown(
                label=r"Row LoRA (from .\loras)",
                choices=_lora_choices(adapters),
                value=NONE_LORA_VALUE,
                elem_id="row-lora",
            )
            row_preset = gr.Dropdown(
                label="Row Preset",
                choices=presets,
                value=preset_default,
                allow_custom_value=True,
                elem_id="row-preset",
            )

        with gr.Accordion("Global Settings", open=False):
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k (0=off)")
                max_tokens = gr.Slider(64, 2048, value=700, step=1, label="Max Tokens")
            with gr.Row():
                repetition_penalty = gr.Slider(1.0, 1.5, value=1.0, step=0.05, label="Repetition Penalty")
                presence_penalty = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Presence Penalty")
                frequency_penalty = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Frequency Penalty")
                lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA Scale")
            with gr.Row():
                best_of_n_enabled = gr.Checkbox(value=False, label="Best-of-N")
                best_of_n_n = gr.Slider(1, 8, value=2, step=1, label="N")
                best_of_n_language = gr.Dropdown(
                    choices=["auto", "ja", "en"], value="auto", label="Language"
                )
                silence_sec = gr.Slider(0.0, 2.0, value=0.2, step=0.05, label="Line Silence (sec)")

        with gr.Row():
            process_row_btn = gr.Button("Process Selected Row And Play")
            process_all_btn = gr.Button("Continuous TTS (All Rows)")

        row_audio = gr.Audio(label="Selected Row Audio", type="numpy")
        combined_audio = gr.Audio(label="Combined Audio", type="numpy")
        combined_file = gr.File(label="Combined WAV File")
        log_text = gr.Markdown()

        refresh_btn.click(
            fn=_refresh_refs,
            inputs=[llm_api_base],
            outputs=[
                presets_state,
                adapters_state,
                default_preset,
                row_preset,
                default_lora,
                row_lora,
                refresh_msg,
                current_lora_state,
            ],
        )

        split_btn.click(
            fn=_split_into_rows,
            inputs=[long_text, default_preset, default_lora, adapters_state],
            outputs=[rows_state, line_table, selected_row, row_text, log_text],
        )

        clear_cache_btn.click(
            fn=_clear_cache,
            inputs=[rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        )

        line_table.change(
            fn=_apply_table_edits,
            inputs=[line_table, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        ).then(
            fn=_load_row_settings,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[row_text, row_lora, row_preset],
        )

        line_table.select(fn=_on_table_select, outputs=[selected_row]).then(
            fn=_load_row_settings,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[row_text, row_lora, row_preset],
        )

        selected_row.change(
            fn=_load_row_settings,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[row_text, row_lora, row_preset],
        )

        row_lora.change(
            fn=_apply_row_settings,
            inputs=[selected_row, row_lora, row_preset, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        )

        row_preset.change(
            fn=_apply_row_settings,
            inputs=[selected_row, row_lora, row_preset, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        )

        delete_row_btn.click(
            fn=_delete_selected_row,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[rows_state, line_table, selected_row, row_text, row_lora, row_preset, log_text],
        )

        insert_row_btn.click(
            fn=_insert_row_below,
            inputs=[selected_row, row_lora, row_preset, rows_state, adapters_state, presets_state],
            outputs=[rows_state, line_table, selected_row, row_text, row_lora, row_preset, log_text],
        )

        process_row_btn.click(
            fn=_process_selected_row,
            inputs=[
                tts_api_base,
                llm_api_base,
                selected_row,
                rows_state,
                adapters_state,
                current_lora_state,
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
                lora_scale,
            ],
            outputs=[rows_state, line_table, row_audio, current_lora_state, log_text],
        )

        process_all_btn.click(
            fn=_process_all_rows,
            inputs=[
                tts_api_base,
                llm_api_base,
                rows_state,
                adapters_state,
                current_lora_state,
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
                lora_scale,
                silence_sec,
            ],
            outputs=[rows_state, line_table, combined_audio, combined_file, current_lora_state, log_text],
        )

    return demo


def main() -> None:
    app = build_app()
    try:
        app.launch(css=UI_CSS)
    except TypeError:
        app.launch()


if __name__ == "__main__":
    main()
