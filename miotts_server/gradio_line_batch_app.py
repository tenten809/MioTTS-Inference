from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import threading
import time
import unicodedata
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
DEFAULT_TEXT_REPLACE_CSV = Path(os.getenv("MIOTTS_TEXT_REPLACE_CSV", "text_replace_dict.csv")).expanduser()
NONE_LORA_VALUE = "__none__"
PROCESS_LOCK = threading.Lock()
UNKNOWN_LORA_STATE = {"id": "__unknown__", "scale": -1.0}
_ASCII_TO_FULLWIDTH = str.maketrans({chr(i): chr(i + 0xFEE0) for i in range(33, 127)})
_DAKUTEN_MARKS_RE = re.compile(r"[゛゜ﾞﾟ゙゚]")
_DAKUTEN_WITH_SPACE_RE = re.compile(r"[ \t\u3000]*[゛゜ﾞﾟ゙゚]")
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001FAFF"  # symbols and pictographs
    "\U00002700-\U000027BF"  # dingbats
    "]",
    flags=re.UNICODE,
)
_EMOJI_JOINERS_RE = re.compile(r"[\u200d\ufe0f]")
_CARD_SUIT_RE = re.compile(r"[♠♥♦♣♤♡♢♧]")
_CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]")
_RUBY_WITH_BASE_RE = re.compile(r"[｜|]([^｜|《》\n]+)《[^《》\n]+》")
_RUBY_BRACKET_RE = re.compile(r"《[^《》\n]+》")
_NOTE_MARK_RE = re.compile(r"※[^\n。！？!?]*")
_DATE_YMD_RE = re.compile(r"(?<!\d)(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?!\d)")
_SYMBOL_REPEAT_RE = re.compile(r"([!！?？。．，、…〜～~・:：;；,．.])\1{3,}")
_LONG_SOKUON_REPEAT_RE = re.compile(r"([ーっッ])\1{3,}")
_ELLIPSIS_REPEAT_RE = re.compile(r"…{2,}")
_ELLIPSIS_SENTINEL = "\uE000"
_OPEN_BRACKETS = {"「": "」", "『": "』", "(": ")", "（": "）", "[": "]", "［": "］", "{": "}", "｛": "｝", "〈": "〉", "《": "》", "【": "】", "〔": "〕", "〖": "〗"}
_SENTENCE_ENDINGS = {"。", "．", "！", "？", "!", "?"}
_BRACKET_SPLIT_DELIMS = _SENTENCE_ENDINGS | {"、", "，", ",", "：", ":", "；", ";"}
_BRACKET_MAX_CHARS = 50
MIN_SPEECH_RATE = 0.5
MAX_SPEECH_RATE = 2.0
DEFAULT_SPEECH_RATE = 1.0
_TEXT_REPLACE_CACHE_KEY: tuple[Path, float] | None = None
_TEXT_REPLACE_CACHE: list[tuple[str, str]] = []
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
#shortcut-controls {
  display: none !important;
}
"""
_ROW_SETTINGS_CLIPBOARD_TYPE = "miotts.row_settings.v1"
_ROW_SETTINGS_SHORTCUTS_JS = r"""
<script>
(() => {
  if (window.__miottsLineBatchShortcutsRegistered) {
    return;
  }
  window.__miottsLineBatchShortcutsRegistered = true;

  const directTargets = ["row-lora", "row-preset", "row-speech-rate", "selected-row"];
  const CONTEXT_TTL_MS = 4000;
  let shortcutGuardUntil = 0;
  let rememberedContextUntil = 0;

  const getEventElement = (evt) => {
    const t = evt && evt.target;
    if (t && typeof t.closest === "function") {
      return t;
    }
    return document.activeElement;
  };
  const stopDefault = (evt) => {
    evt.preventDefault();
    evt.stopPropagation();
    if (typeof evt.stopImmediatePropagation === "function") {
      evt.stopImmediatePropagation();
    }
  };
  const clickHostButton = (id) => {
    const host = document.getElementById(id);
    if (!host) return;
    const btn = host.querySelector("button") || host;
    if (typeof btn.click === "function") {
      btn.click();
    }
  };
  const runShortcut = (key) => {
    shortcutGuardUntil = Date.now() + 180;
    clickHostButton(key === "c" ? "copy-row-settings-shortcut" : "paste-row-settings-shortcut");
  };
  const parseColumnIndex = (el) => {
    if (!el || !el.closest) return null;
    const cell = el.closest("td,th,[role='gridcell']");
    if (!cell) return null;

    const dataCol = cell.getAttribute("data-col");
    if (dataCol && /^-?\d+$/.test(dataCol)) {
      return parseInt(dataCol, 10);
    }

    const ariaColIndex = cell.getAttribute("aria-colindex");
    if (ariaColIndex && /^-?\d+$/.test(ariaColIndex)) {
      return parseInt(ariaColIndex, 10) - 1;
    }
    return null;
  };
  const isTargetContext = (el) => {
    if (!el || !el.closest) return false;
    if (directTargets.some((id) => el.closest(`#${id}`))) {
      return true;
    }
    const inTable = !!el.closest("#tts-rows-table");
    if (!inTable) return false;

    const col = parseColumnIndex(el);
    if (col === 1) {
      return false; // text column keeps native copy/paste
    }
    return true;
  };

  const isTextContext = (el) => {
    if (!el || !el.closest) return false;
    if (el.closest("#long-text-input")) return true;
    if (!el.closest("#tts-rows-table")) return false;
    const col = parseColumnIndex(el);
    return col === 1;
  };

  const rememberContextFrom = (el) => {
    if (isTargetContext(el)) {
      rememberedContextUntil = Date.now() + CONTEXT_TTL_MS;
      return;
    }
    if (isTextContext(el)) {
      rememberedContextUntil = 0;
    }
  };

  const shouldHandle = (evt) => {
    const target = getEventElement(evt);
    rememberContextFrom(target);
    if (isTargetContext(target)) return true;
    const active = document.activeElement;
    rememberContextFrom(active);
    if (isTargetContext(active)) return true;
    if (Date.now() < rememberedContextUntil) {
      if (isTextContext(target) || isTextContext(active)) return false;
      return true;
    }
    return false;
  };

  document.addEventListener(
    "pointerdown",
    (evt) => {
      rememberContextFrom(getEventElement(evt));
    },
    true
  );

  document.addEventListener(
    "focusin",
    (evt) => {
      rememberContextFrom(getEventElement(evt));
    },
    true
  );

  document.addEventListener(
    "keydown",
    (evt) => {
      if (!(evt.ctrlKey || evt.metaKey) || evt.altKey || evt.shiftKey) return;
      const key = (evt.key || "").toLowerCase();
      if (key !== "c" && key !== "v") return;
      if (!shouldHandle(evt)) return;
      stopDefault(evt);
      runShortcut(key);
    },
    true
  );

  document.addEventListener(
    "copy",
    (evt) => {
      if (Date.now() < shortcutGuardUntil) return;
      if (!shouldHandle(evt)) return;
      stopDefault(evt);
      runShortcut("c");
    },
    true
  );

  document.addEventListener(
    "paste",
    (evt) => {
      if (Date.now() < shortcutGuardUntil) return;
      if (!shouldHandle(evt)) return;
      stopDefault(evt);
      runShortcut("v");
    },
    true
  );
})();
</script>
"""
_WRITE_ROW_SETTINGS_CLIPBOARD_JS = """
(payload) => {
  const text = (payload || "").toString();
  window.__miottsRowSettingsClipboard = text;
  if (!text) {
    return;
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
  }
}
"""
_READ_ROW_SETTINGS_CLIPBOARD_JS = """
async (selectedRow, cachedPayload, rows, adapters) => {
  let text = (cachedPayload || "").toString();
  if (navigator.clipboard && navigator.clipboard.readText) {
    try {
      const clipped = await navigator.clipboard.readText();
      if (clipped && clipped.trim()) {
        text = clipped;
      }
    } catch (err) {}
  }
  if ((!text || !text.trim()) && window.__miottsRowSettingsClipboard) {
    text = window.__miottsRowSettingsClipboard;
  }
  return [selectedRow, text, rows, adapters];
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


def _normalize_date_ymd(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        y = int(match.group(1))
        m = int(match.group(2))
        d = int(match.group(3))
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return match.group(0)
        return f"{y}年{m}月{d}日"

    return _DATE_YMD_RE.sub(_replace, text)


def _load_text_replace_dict(csv_path: Path | str) -> list[tuple[str, str]]:
    global _TEXT_REPLACE_CACHE_KEY, _TEXT_REPLACE_CACHE
    path = Path(csv_path).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        return []
    if not resolved.exists() or not resolved.is_file():
        return []
    try:
        mtime = resolved.stat().st_mtime
    except Exception:
        return []
    cache_key = (resolved, mtime)
    if _TEXT_REPLACE_CACHE_KEY == cache_key:
        return _TEXT_REPLACE_CACHE

    rows: list[tuple[str, str]] = []
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                src = row[0].strip()
                dst = row[1].strip()
                if not src or src.startswith("#"):
                    continue
                if src.lower() in {"word", "source"} and dst.lower() in {"reading", "kana", "yomi", "target"}:
                    continue
                rows.append((src, dst))
    except Exception:
        return []

    unique: dict[str, str] = {}
    for src, dst in rows:
        if src not in unique:
            unique[src] = dst
    items = sorted(unique.items(), key=lambda x: len(x[0]), reverse=True)
    _TEXT_REPLACE_CACHE_KEY = cache_key
    _TEXT_REPLACE_CACHE = items
    return items


def _apply_text_replace_dict(text: str) -> str:
    for src, dst in _load_text_replace_dict(DEFAULT_TEXT_REPLACE_CSV):
        text = text.replace(src, dst)
    return text


def _normalize_text_for_tts(text: str) -> str:
    # Keep U+2026 as-is across NFKC (NFKC may expand it to three periods).
    text = text.replace("…", _ELLIPSIS_SENTINEL)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_ELLIPSIS_SENTINEL, "…")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _RUBY_WITH_BASE_RE.sub(r"\1", text)
    text = _RUBY_BRACKET_RE.sub("", text)
    text = _NOTE_MARK_RE.sub("", text)
    text = _DAKUTEN_WITH_SPACE_RE.sub("", text)
    text = _DAKUTEN_MARKS_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _EMOJI_JOINERS_RE.sub("", text)
    text = _CARD_SUIT_RE.sub("", text)
    text = _normalize_date_ymd(text)
    text = _SYMBOL_REPEAT_RE.sub(lambda m: m.group(1) * 3, text)
    text = _LONG_SOKUON_REPEAT_RE.sub(lambda m: m.group(1) * 3, text)
    text = _ELLIPSIS_REPEAT_RE.sub("…", text)
    text = _apply_text_replace_dict(text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r" +([。．！？!?、，])", r"\1", text)
    text = re.sub(r" *\n *", "\n", text)
    text = text.translate(_ASCII_TO_FULLWIDTH)
    return text.strip()


def _split_text_to_lines(text: str) -> list[str]:
    text = _normalize_text_for_tts(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []

    def _split_plain_text(chunk: str) -> list[str]:
        out: list[str] = []
        buff: list[str] = []
        for ch in chunk:
            buff.append(ch)
            if ch in _SENTENCE_ENDINGS:
                sentence = "".join(buff).strip()
                if sentence:
                    out.append(sentence)
                buff = []
        tail = "".join(buff).strip()
        if tail:
            out.append(tail)
        return out

    def _split_long_bracketed_chunk(chunk: str) -> list[str]:
        chunk = chunk.strip()
        if len(chunk) <= _BRACKET_MAX_CHARS:
            return [chunk] if chunk else []

        out: list[str] = []
        start = 0
        n = len(chunk)
        while n - start > _BRACKET_MAX_CHARS:
            split_idx = -1
            i = start + _BRACKET_MAX_CHARS
            while i < n:
                if chunk[i] in _BRACKET_SPLIT_DELIMS:
                    split_idx = i
                    break
                i += 1
            if split_idx < 0:
                break
            part = chunk[start : split_idx + 1].strip()
            if part:
                out.append(part)
            start = split_idx + 1

        tail = chunk[start:].strip()
        if tail:
            out.append(tail)
        return out

    def _split_block_preserving_brackets(block: str) -> list[str]:
        out: list[str] = []
        i = 0
        n = len(block)
        plain_start = 0

        while i < n:
            ch = block[i]
            if ch not in _OPEN_BRACKETS:
                i += 1
                continue

            # Emit plain-text part before bracket.
            if plain_start < i:
                out.extend(_split_plain_text(block[plain_start:i]))

            # Find matching closing bracket with nesting.
            stack: list[str] = [_OPEN_BRACKETS[ch]]
            j = i + 1
            while j < n and stack:
                cj = block[j]
                if cj in _OPEN_BRACKETS:
                    stack.append(_OPEN_BRACKETS[cj])
                elif cj == stack[-1]:
                    stack.pop()
                j += 1

            if stack:
                # Unclosed bracket: treat the rest as plain text.
                out.extend(_split_plain_text(block[i:]))
                return out

            chunk = block[i:j]
            out.extend(_split_long_bracketed_chunk(chunk))
            i = j
            plain_start = i

        if plain_start < n:
            out.extend(_split_plain_text(block[plain_start:]))
        return out

    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for part in _split_block_preserving_brackets(block):
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
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pt", ".npz"}:
            continue
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


def _parse_lora_key_from_table_cell(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    # Keep only the first token if accidental CSV-like text was pasted.
    if "," in text:
        text = text.split(",", 1)[0].strip()
    lowered = text.lower()
    if lowered in {"none", "(none)", "__none__"}:
        return None
    while text.lower().startswith("path:"):
        text = text.split(":", 1)[1].strip()

    lead_anno = re.compile(r"^\s*\((?:id:\d+|not loaded|missing)\)\s*", flags=re.IGNORECASE)
    tail_anno = re.compile(r"\s*\((?:id:\d+|not loaded|missing)\)\s*$", flags=re.IGNORECASE)
    prev = None
    while prev != text:
        prev = text
        text = lead_anno.sub("", text)
        text = tail_anno.sub("", text)
        text = text.strip()

    if text and not text.startswith("\\"):
        first_bs = text.find("\\")
        if first_bs >= 0:
            text = text[first_bs:]

    match = re.search(r"(\\[^,\r\n]*?\.gguf)", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1)

    text = text.strip()
    if not text:
        return None
    if text.lower() in {"none", "(none)", "__none__"}:
        return None
    return text


def _is_known_lora_key(lora_key: str | None, adapters: list[dict[str, Any]]) -> bool:
    if not lora_key:
        return False
    target = str(lora_key)
    for item in adapters:
        if str(item.get("display") or "") == target:
            return True
    return False


def _lora_key_from_value(value: str | None, adapters: list[dict[str, Any]] | None = None) -> str | None:
    lora_key = _parse_lora_key_from_table_cell(value)
    if lora_key is None:
        return None
    if adapters is not None and not _is_known_lora_key(lora_key, adapters):
        return None
    return lora_key


def _normalize_speech_rate(value: Any, default: float = DEFAULT_SPEECH_RATE) -> float:
    try:
        rate = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(rate):
        return float(default)
    return float(min(MAX_SPEECH_RATE, max(MIN_SPEECH_RATE, rate)))


def _lora_value_from_key(lora_key: str | None, adapters: list[dict[str, Any]] | None = None) -> str:
    normalized = _parse_lora_key_from_table_cell(lora_key)
    if not normalized:
        return NONE_LORA_VALUE
    if adapters is not None and not _is_known_lora_key(normalized, adapters):
        return NONE_LORA_VALUE
    return f"path:{normalized}"


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
                _normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE),
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
        "speech_rate": _normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE),
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
        "speech_rate": _normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE),
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
    default_speech_rate: float,
    default_lora_value: str,
    adapters: list[dict[str, Any]],
):
    lines = _split_text_to_lines(long_text)
    default_lora_key = _lora_key_from_value(default_lora_value, adapters)
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        rows.append(
            {
                "idx": i,
                "text": line,
                "lora_key": default_lora_key,
                "preset_id": default_preset,
                "speech_rate": _normalize_speech_rate(default_speech_rate, DEFAULT_SPEECH_RATE),
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
        return (
            "",
            gr.update(choices=_lora_choices(adapters), value=NONE_LORA_VALUE),
            gr.update(choices=presets, value=(presets[0] if presets else None)),
            DEFAULT_SPEECH_RATE,
        )
    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]
    return (
        row["text"],
        gr.update(choices=_lora_choices(adapters), value=_lora_value_from_key(row.get("lora_key"), adapters)),
        gr.update(choices=presets, value=row.get("preset_id")),
        _normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE),
    )


def _apply_row_settings(
    selected_row: int,
    row_lora_value: str,
    row_preset: str,
    row_speech_rate: float,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
):
    if not rows:
        return rows, _rows_table(rows, adapters), "no rows"
    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]
    lora_key = _lora_key_from_value(row_lora_value, adapters)
    speech_rate = _normalize_speech_rate(row_speech_rate, DEFAULT_SPEECH_RATE)
    changed = (
        (row.get("lora_key") != lora_key)
        or (row.get("preset_id") != row_preset)
        or (_normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE) != speech_rate)
    )
    row["lora_key"] = lora_key
    row["preset_id"] = row_preset
    row["speech_rate"] = speech_rate
    if changed:
        row["cache_key"] = None
        row["audio_b64"] = None
        row["sample_rate"] = None
        row["status"] = "pending"
    return rows, _rows_table(rows, adapters), f"row {row['idx']} updated"


def _parse_row_settings_clipboard_payload(
    clipboard_text: str,
    adapters: list[dict[str, Any]],
) -> tuple[str | None, str, float] | None:
    text = str(clipboard_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload_type = str(payload.get("type") or "").strip()
    if payload_type != _ROW_SETTINGS_CLIPBOARD_TYPE:
        return None

    lora_raw = payload.get("lora_key")
    if lora_raw is None:
        lora_raw = payload.get("lora")
    lora_key = _lora_key_from_value(str(lora_raw or ""), adapters)
    preset_id = str(payload.get("preset_id") or payload.get("preset") or "").strip()
    speech_rate = _normalize_speech_rate(payload.get("speech_rate"), DEFAULT_SPEECH_RATE)
    return lora_key, preset_id, speech_rate


def _copy_row_settings(selected_row: int, rows: list[dict[str, Any]]):
    if not rows:
        return "", "copy settings: no rows"
    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]
    lora_key = _parse_lora_key_from_table_cell(row.get("lora_key"))
    payload = {
        "type": _ROW_SETTINGS_CLIPBOARD_TYPE,
        "lora_key": lora_key,
        "preset_id": str(row.get("preset_id") or ""),
        "speech_rate": _normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE),
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text, f"row {row['idx']} settings copied"


def _paste_row_settings(
    selected_row: int,
    clipboard_text: str,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
):
    if not rows:
        return rows, _rows_table(rows, adapters), "paste settings: no rows"
    parsed = _parse_row_settings_clipboard_payload(clipboard_text, adapters)
    if parsed is None:
        return rows, _rows_table(rows, adapters), "paste settings: clipboard is not row settings"

    idx = max(1, min(int(selected_row), len(rows))) - 1
    row = rows[idx]
    lora_key, preset_id, speech_rate = parsed
    changed = (
        (row.get("lora_key") != lora_key)
        or (str(row.get("preset_id") or "") != preset_id)
        or (_normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE) != speech_rate)
    )
    row["lora_key"] = lora_key
    row["preset_id"] = preset_id
    row["speech_rate"] = speech_rate
    if changed:
        row["cache_key"] = None
        row["audio_b64"] = None
        row["sample_rate"] = None
        row["status"] = "pending"
    return rows, _rows_table(rows, adapters), f"row {row['idx']} settings pasted"


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
        speech_rate = _normalize_speech_rate(cells[4] if len(cells) > 4 else DEFAULT_SPEECH_RATE, DEFAULT_SPEECH_RATE)

        row = {
            "idx": new_idx,
            "text": text,
            "lora_key": lora_key,
            "preset_id": preset_id,
            "speech_rate": speech_rate,
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
                and _normalize_speech_rate(old.get("speech_rate"), DEFAULT_SPEECH_RATE) == speech_rate
            )
            if same:
                row["status"] = old.get("status", "pending")
                row["audio_b64"] = old.get("audio_b64")
                row["cache_key"] = old.get("cache_key")
                row["sample_rate"] = old.get("sample_rate")
                row["speech_rate"] = _normalize_speech_rate(
                    old.get("speech_rate"),
                    DEFAULT_SPEECH_RATE,
                )
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
    row_speech_rate: float,
    rows: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    presets: list[str],
):
    insert_idx = 0
    if rows:
        insert_idx = max(1, min(int(selected_row), len(rows)))
    lora_key = _lora_key_from_value(row_lora_value, adapters)
    preset = (row_preset or "").strip() or (presets[0] if presets else "")
    speech_rate = _normalize_speech_rate(row_speech_rate, DEFAULT_SPEECH_RATE)

    new_row = {
        "idx": 0,
        "text": "",
        "lora_key": lora_key,
        "preset_id": preset,
        "speech_rate": speech_rate,
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
        gr.update(choices=_lora_choices(adapters), value=_lora_value_from_key(new_row.get("lora_key"), adapters)),
        gr.update(choices=presets, value=new_row.get("preset_id")),
        _normalize_speech_rate(new_row.get("speech_rate"), DEFAULT_SPEECH_RATE),
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
            DEFAULT_SPEECH_RATE,
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
            DEFAULT_SPEECH_RATE,
            f"row {removed['idx']} deleted",
        )
    new_selected = max(1, min(idx + 1, len(rows)))
    row = rows[new_selected - 1]
    return (
        rows,
        _rows_table(rows, adapters),
        new_selected,
        row["text"],
        gr.update(choices=_lora_choices(adapters), value=_lora_value_from_key(row.get("lora_key"), adapters)),
        gr.update(choices=presets, value=row.get("preset_id")),
        _normalize_speech_rate(row.get("speech_rate"), DEFAULT_SPEECH_RATE),
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
        gr.HTML(_ROW_SETTINGS_SHORTCUTS_JS)

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
                allow_custom_value=True,
            )
            default_speech_rate = gr.Slider(
                MIN_SPEECH_RATE,
                MAX_SPEECH_RATE,
                value=DEFAULT_SPEECH_RATE,
                step=0.05,
                label="Default Speech Rate For New Rows",
            )
            split_btn = gr.Button("Split Into Lines")
            clear_cache_btn = gr.Button("Clear Audio Cache")

        line_table = gr.Dataframe(
            headers=["idx", "text", "lora", "preset", "speech_rate", "status", "cached"],
            datatype=["number", "str", "str", "str", "number", "str", "str"],
            value=[],
            wrap=False,
            interactive=True,
            max_height=840,
            column_widths=["5%", "37%", "24%", "14%", "8%", "6%", "6%"],
            label="TTS Rows (scrollable)",
            elem_id="tts-rows-table",
        )
        with gr.Row(elem_id="shortcut-controls"):
            copy_row_settings_shortcut = gr.Button("Copy Row Settings", elem_id="copy-row-settings-shortcut")
            paste_row_settings_shortcut = gr.Button("Paste Row Settings", elem_id="paste-row-settings-shortcut")
            row_settings_clipboard = gr.Textbox(
                label="Row Settings Clipboard",
                value="",
                elem_id="row-settings-clipboard",
            )

        with gr.Row():
            selected_row = gr.Number(label="Selected Row (1-based)", value=0, precision=0, elem_id="selected-row")
            row_text = gr.Textbox(label="Selected Text", interactive=False)
            insert_row_btn = gr.Button("Insert Row Below", variant="secondary")
            delete_row_btn = gr.Button("Delete Selected Row", variant="secondary")
        with gr.Row():
            row_lora = gr.Dropdown(
                label=r"Row LoRA (from .\loras)",
                choices=_lora_choices(adapters),
                value=NONE_LORA_VALUE,
                allow_custom_value=True,
                elem_id="row-lora",
            )
            row_preset = gr.Dropdown(
                label="Row Preset",
                choices=presets,
                value=preset_default,
                allow_custom_value=True,
                elem_id="row-preset",
            )
            row_speech_rate = gr.Slider(
                MIN_SPEECH_RATE,
                MAX_SPEECH_RATE,
                value=DEFAULT_SPEECH_RATE,
                step=0.05,
                label="Row Speech Rate",
                elem_id="row-speech-rate",
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
            inputs=[long_text, default_preset, default_speech_rate, default_lora, adapters_state],
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
            outputs=[row_text, row_lora, row_preset, row_speech_rate],
        )

        line_table.select(fn=_on_table_select, outputs=[selected_row]).then(
            fn=_load_row_settings,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[row_text, row_lora, row_preset, row_speech_rate],
        )

        selected_row.change(
            fn=_load_row_settings,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[row_text, row_lora, row_preset, row_speech_rate],
        )

        copy_row_settings_shortcut.click(
            fn=_copy_row_settings,
            inputs=[selected_row, rows_state],
            outputs=[row_settings_clipboard, log_text],
        )

        row_settings_clipboard.change(
            fn=None,
            inputs=[row_settings_clipboard],
            outputs=[],
            js=_WRITE_ROW_SETTINGS_CLIPBOARD_JS,
        )

        paste_row_settings_shortcut.click(
            fn=_paste_row_settings,
            inputs=[selected_row, row_settings_clipboard, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
            js=_READ_ROW_SETTINGS_CLIPBOARD_JS,
        ).then(
            fn=_load_row_settings,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[row_text, row_lora, row_preset, row_speech_rate],
        )

        row_lora.change(
            fn=_apply_row_settings,
            inputs=[selected_row, row_lora, row_preset, row_speech_rate, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        )

        row_preset.change(
            fn=_apply_row_settings,
            inputs=[selected_row, row_lora, row_preset, row_speech_rate, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        )

        row_speech_rate.change(
            fn=_apply_row_settings,
            inputs=[selected_row, row_lora, row_preset, row_speech_rate, rows_state, adapters_state],
            outputs=[rows_state, line_table, log_text],
        )

        delete_row_btn.click(
            fn=_delete_selected_row,
            inputs=[selected_row, rows_state, adapters_state, presets_state],
            outputs=[rows_state, line_table, selected_row, row_text, row_lora, row_preset, row_speech_rate, log_text],
        )

        insert_row_btn.click(
            fn=_insert_row_below,
            inputs=[selected_row, row_lora, row_preset, row_speech_rate, rows_state, adapters_state, presets_state],
            outputs=[rows_state, line_table, selected_row, row_text, row_lora, row_preset, row_speech_rate, log_text],
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
