from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .asr import ASRConfig, ASRService
from .audio import load_reference_audio_bytes, write_wav_bytes
from .best_of_n import BestOfNCandidate, detect_language, score_candidates
from .codec import MioCodecService
from .config import get_audio_config, get_config, get_llm_defaults
from .llm_client import LLMClient
from .schemas import (
    BatchTTSItem,
    BatchTTSItemResponse,
    BatchTTSRequest,
    BatchTTSResponse,
    BestOfNConfig,
    LLMParams,
    OutputConfig,
    ReferenceConfig,
    TTSRequest,
    TTSResponse,
    TTSTimings,
)
from .text import normalize_text
from .token_parser import parse_speech_tokens

logger = logging.getLogger(__name__)
SPEECH_TOKEN_SYSTEM_PROMPT = (
    "You are a TTS token generator. "
    "Output only speech tokens in the exact format <|s_123|> with no extra text."
)


@dataclass
class _ResolvedLLMSettings:
    model: str
    temperature: float
    top_p: float
    top_k: int | None
    max_tokens: int
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float


@dataclass
class _PreparedBatchSynthesisItem:
    index: int
    normalized_text: str
    global_embedding: torch.Tensor
    speech_rate: float
    llm_sec: float = 0.0
    parse_sec: float = 0.0
    token_candidates: list[list[int]] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    _configure_torch(config)
    llm_client = LLMClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout,
    )
    codec_service = MioCodecService(
        model_id=config.codec_model_id,
        adapter_path=config.codec_adapter_path,
        device=config.device,
        presets_dir=config.presets_dir,
    )
    codec_service.load()
    asr_service = None
    if config.best_of_n_enabled:
        try:
            asr_config = ASRConfig(
                model_id=config.asr_model,
                device=config.asr_device,
                compute_type=config.asr_compute_type,
                batch_size=config.asr_batch_size,
                language=config.asr_language,
            )
            asr_service = ASRService(asr_config)
            asr_service.load()
        except Exception as exc:
            logger.warning("ASR unavailable: %s", exc)
            asr_service = None
    app.state.llm_client = llm_client
    app.state.codec_service = codec_service
    app.state.asr_service = asr_service

    yield

    await llm_client.close()


app = FastAPI(title="MioTTS API Server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/presets")
async def list_presets() -> dict[str, Any]:
    codec_service: MioCodecService = app.state.codec_service
    return {"presets": codec_service.list_presets()}


@app.post("/v1/tts")
async def tts_json(request: TTSRequest):
    output_format = _resolve_output_format(request.output, default_format="base64")
    result = await _run_tts(request, output_format)
    return result


@app.post("/v1/tts/batch")
async def tts_batch_json(request: BatchTTSRequest):
    output_format = _resolve_output_format(request.output, default_format="base64")
    if output_format != "base64":
        raise HTTPException(status_code=400, detail="batch endpoint supports only base64 output")
    return await _run_tts_batch(request)


@app.post("/v1/tts/file")
async def tts_file(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_preset_id: str | None = Form(None),
    model: str | None = Form(None),
    temperature: float | None = Form(None),
    top_p: float | None = Form(None),
    top_k: int | None = Form(None),
    max_tokens: int | None = Form(None),
    repetition_penalty: float | None = Form(None),
    presence_penalty: float | None = Form(None),
    frequency_penalty: float | None = Form(None),
    speech_rate: float | None = Form(None),
    output_format: str | None = Form(None),
    best_of_n_enabled: bool | None = Form(None),
    best_of_n_n: int | None = Form(None),
    best_of_n_language: str | None = Form(None),
):
    reference: ReferenceConfig | None = None
    reference_bytes: bytes | None = None
    if reference_audio is not None:
        reference_bytes = await _read_reference_file(reference_audio)
        reference = ReferenceConfig(type="base64", data="")
    elif reference_preset_id:
        reference = ReferenceConfig(type="preset", preset_id=reference_preset_id)

    try:
        best_of_n = None
        if any(
            value is not None
            for value in (
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
            )
        ):
            best_of_n = BestOfNConfig(
                enabled=best_of_n_enabled,
                n=best_of_n_n,
                language=best_of_n_language,
            )

        request = TTSRequest(
            text=text,
            reference=reference,
            llm=LLMParams(
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            ),
            output=OutputConfig(format=output_format),
            best_of_n=best_of_n,
            speech_rate=speech_rate,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    output_format = _resolve_output_format(request.output, default_format="wav")
    return await _run_tts(request, output_format, reference_bytes=reference_bytes)


async def _run_tts(
    request: TTSRequest,
    output_format: str,
    reference_bytes: bytes | None = None,
):
    config = get_config()
    llm_defaults = get_llm_defaults()
    codec_service: MioCodecService = app.state.codec_service
    llm_client: LLMClient = app.state.llm_client

    normalized = _normalize_text_for_tts(request.text, config)
    llm_settings = await _resolve_llm_settings(
        llm_params=request.llm,
        config=config,
        llm_defaults=llm_defaults,
        llm_client=llm_client,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": normalized}]

    best_of_n = _resolve_best_of_n(request, config)
    asr_service: ASRService | None = app.state.asr_service
    logger.debug(
        "Best-of-n resolved: enabled=%s n=%d lang=%s",
        best_of_n.enabled,
        best_of_n.n,
        best_of_n.language,
    )

    t0 = time.perf_counter()
    try:
        llm_texts = await _fetch_llm_candidates(
            llm_client=llm_client,
            messages=messages,
            model=llm_settings.model,
            temperature=llm_settings.temperature,
            top_p=llm_settings.top_p,
            top_k=llm_settings.top_k,
            max_tokens=llm_settings.max_tokens,
            repetition_penalty=llm_settings.repetition_penalty,
            presence_penalty=llm_settings.presence_penalty,
            frequency_penalty=llm_settings.frequency_penalty,
            n=best_of_n.n if best_of_n.enabled else 1,
        )
    except Exception as exc:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
    t1 = time.perf_counter()

    tokens_list = _parse_llm_candidates(llm_texts)
    if not tokens_list:
        logger.warning("No speech tokens found; retrying LLM with strict token prompt.")
        strict_messages = [
            {"role": "system", "content": SPEECH_TOKEN_SYSTEM_PROMPT},
            {"role": "user", "content": normalized},
        ]
        try:
            llm_texts_retry = await _fetch_llm_candidates(
                llm_client=llm_client,
                messages=strict_messages,
                model=llm_settings.model,
                temperature=min(float(llm_settings.temperature), 0.4),
                top_p=min(float(llm_settings.top_p), 0.95),
                top_k=llm_settings.top_k,
                max_tokens=llm_settings.max_tokens,
                repetition_penalty=llm_settings.repetition_penalty,
                presence_penalty=llm_settings.presence_penalty,
                frequency_penalty=llm_settings.frequency_penalty,
                n=best_of_n.n if best_of_n.enabled else 1,
            )
        except Exception as exc:
            logger.exception("LLM strict retry failed")
            raise HTTPException(status_code=502, detail=f"LLM strict retry failed: {exc}") from exc
        tokens_list = _parse_llm_candidates(llm_texts_retry)
    if not tokens_list:
        raise HTTPException(
            status_code=422,
            detail=(
                "No speech tokens found in LLM output. "
                "Try lower temperature/top_p, or verify llama-server model/settings."
            ),
        )
    logger.debug(
        "LLM candidates: count=%d token_lengths=%s", len(tokens_list), [len(t) for t in tokens_list]
    )
    t2 = time.perf_counter()

    reference_waveform = None
    global_embedding = None
    if request.reference is None:
        raise HTTPException(status_code=400, detail="reference is required")

    if request.reference.type == "base64":
        if reference_bytes is None:
            if not request.reference.data:
                raise HTTPException(status_code=400, detail="reference.data is required")
            try:
                payload = request.reference.data
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                max_bytes = config.max_reference_mb * 1024 * 1024
                payload = _strip_base64_whitespace(payload)
                estimated_size = _estimate_base64_decoded_size(payload)
                if estimated_size > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"reference audio too large (max {config.max_reference_mb} MB)",
                    )
                reference_bytes = base64.b64decode(payload, validate=True)
                if len(reference_bytes) > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"reference audio too large (max {config.max_reference_mb} MB)",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail="invalid base64 reference") from exc
        try:
            reference_waveform = load_reference_audio_bytes(reference_bytes, codec_service.sample_rate)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid reference audio") from exc
        reference_waveform = _trim_reference(
            reference_waveform, codec_service.sample_rate, config.max_reference_seconds
        )
    elif request.reference.type == "preset":
        preset_id = request.reference.preset_id
        if not preset_id:
            raise HTTPException(status_code=400, detail="reference.preset_id is required")
        try:
            global_embedding = codec_service.load_preset_embedding(preset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail="unsupported reference.type")

    best_of_n_sec = None
    asr_sec = None

    if best_of_n.enabled and best_of_n.n > 1 and len(tokens_list) > 1:
        if asr_service is None:
            raise HTTPException(status_code=400, detail="ASR is not available on the server")
        try:
            audio_batch, audio_lengths = codec_service.synthesize_batch(
                tokens_list,
                reference_waveform,
                global_embedding,
            )
        except Exception as exc:
            logger.exception("Codec synthesis failed")
            raise HTTPException(status_code=500, detail=f"Codec synthesis failed: {exc}") from exc

        t3 = time.perf_counter()

        lengths = (
            audio_lengths.tolist() if hasattr(audio_lengths, "tolist") else list(audio_lengths)
        )
        candidates: list[BestOfNCandidate] = []
        for idx, tokens in enumerate(tokens_list):
            audio_len = int(lengths[idx]) if lengths else audio_batch.shape[1]
            audio = audio_batch[idx, :audio_len]
            candidates.append(BestOfNCandidate(tokens=tokens, audio=audio))
        logger.debug("Decoded batch: audio_lengths=%s", lengths)

        rank_start = time.perf_counter()
        try:
            best_idx, asr_sec = await score_candidates(
                text=normalized,
                candidates=candidates,
                sample_rate=codec_service.sample_rate,
                language=best_of_n.language,
                asr_service=asr_service,
            )
        except Exception as exc:
            logger.exception("Best-of-n scoring failed")
            raise HTTPException(status_code=500, detail=f"Best-of-n scoring failed: {exc}") from exc
        rank_end = time.perf_counter()
        total_rank_sec = rank_end - rank_start
        best_of_n_sec = total_rank_sec
        if asr_sec is not None:
            best_of_n_sec = max(0.0, total_rank_sec - asr_sec)

        selected = candidates[best_idx]
        tokens = selected.tokens
        audio = selected.audio
        logger.debug("Best-of-n selected index=%d tokens=%d", best_idx, len(tokens))
    else:
        tokens = tokens_list[0]
        try:
            audio = codec_service.synthesize(tokens, reference_waveform, global_embedding)
        except Exception as exc:
            logger.exception("Codec synthesis failed")
            raise HTTPException(status_code=500, detail=f"Codec synthesis failed: {exc}") from exc
        t3 = time.perf_counter()

    codec_sample_rate = codec_service.sample_rate

    speech_rate = request.speech_rate if request.speech_rate is not None else 1.0
    if abs(float(speech_rate) - 1.0) > 1e-6:
        audio = _apply_speech_rate(audio, float(speech_rate), codec_sample_rate)

    t4 = time.perf_counter()

    audio_sec = 0.0
    if codec_sample_rate > 0:
        audio_sec = float(audio.numel()) / float(codec_sample_rate)
    rtf = (t4 - t0) / audio_sec if audio_sec > 0 else 0.0

    timings = TTSTimings(
        llm_sec=round(t1 - t0, 4),
        parse_sec=round(t2 - t1, 4),
        codec_sec=round(t3 - t2, 4),
        total_sec=round(t4 - t0, 4),
        best_of_n_sec=round(best_of_n_sec, 4) if best_of_n_sec is not None else None,
        asr_sec=round(asr_sec, 4) if asr_sec is not None else None,
    )
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.info(
        "TTS timings: total=%.3fs llm=%.3fs parse=%.3fs codec=%.3fs best_of_n=%.3fs asr=%.3fs rtf=%.3f tokens=%d",
        timings.total_sec,
        timings.llm_sec,
        timings.parse_sec,
        timings.codec_sec,
        timings.best_of_n_sec or 0.0,
        timings.asr_sec or 0.0,
        rtf,
        len(tokens),
    )

    wav_bytes = write_wav_bytes(audio, codec_sample_rate)
    if output_format == "wav":
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"},
        )

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    response = TTSResponse(
        audio=audio_b64,
        format="base64",
        sample_rate=codec_sample_rate,
        token_count=len(tokens),
        timings=timings,
        normalized_text=normalized,
    )
    return JSONResponse(content=response.model_dump())


async def _run_tts_batch(request: BatchTTSRequest) -> JSONResponse:
    config = get_config()
    llm_defaults = get_llm_defaults()
    codec_service: MioCodecService = app.state.codec_service
    llm_client: LLMClient = app.state.llm_client
    asr_service: ASRService | None = app.state.asr_service

    best_of_n = _resolve_best_of_n(TTSRequest(text="batch-probe", best_of_n=request.best_of_n), config)

    llm_settings = await _resolve_llm_settings(
        llm_params=request.llm,
        config=config,
        llm_defaults=llm_defaults,
        llm_client=llm_client,
    )

    responses: list[BatchTTSItemResponse] = [BatchTTSItemResponse(error="not processed")] * len(
        request.items
    )
    prepared_items: list[_PreparedBatchSynthesisItem] = []

    for idx, item in enumerate(request.items):
        try:
            normalized = _normalize_text_for_tts(item.text, config)
            global_embedding = _resolve_batch_reference_embedding(item, codec_service)
            speech_rate = item.speech_rate if item.speech_rate is not None else 1.0
            responses[idx] = BatchTTSItemResponse(error=None)
            prepared_items.append(
                _PreparedBatchSynthesisItem(
                    index=idx,
                    normalized_text=normalized,
                    global_embedding=global_embedding,
                    speech_rate=float(speech_rate),
                )
            )
        except HTTPException as exc:
            responses[idx] = BatchTTSItemResponse(
                error=f"request invalid: {_http_exception_detail_text(exc)}"
            )
        except Exception as exc:
            responses[idx] = BatchTTSItemResponse(error=f"request invalid: {exc}")

    async def _prepare_tokens(
        item: _PreparedBatchSynthesisItem,
    ) -> tuple[int, list[list[int]], float, float]:
        token_candidates, llm_sec, parse_sec = await _generate_token_candidates_for_text(
            normalized=item.normalized_text,
            llm_client=llm_client,
            llm_settings=llm_settings,
            n=best_of_n.n if best_of_n.enabled else 1,
        )
        return item.index, token_candidates, llm_sec, parse_sec

    if prepared_items:
        token_results = await asyncio.gather(
            *[_prepare_tokens(item) for item in prepared_items],
            return_exceptions=True,
        )

        decodable_items: list[_PreparedBatchSynthesisItem] = []
        by_index = {item.index: item for item in prepared_items}
        for item, result in zip(prepared_items, token_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Batch token generation failed for item %d: %s", item.index, result)
                responses[item.index] = BatchTTSItemResponse(error=f"token generation failed: {result}")
                continue
            row_index, token_candidates, llm_sec, parse_sec = result
            prepared = by_index[row_index]
            prepared.token_candidates = token_candidates
            prepared.llm_sec = llm_sec
            prepared.parse_sec = parse_sec
            decodable_items.append(prepared)

        if decodable_items:
            codec_start = time.perf_counter()
            try:
                flat_tokens: list[list[int]] = []
                flat_embeddings: list[torch.Tensor] = []
                flat_item_indices: list[int] = []
                for item in decodable_items:
                    candidates = item.token_candidates or []
                    for tokens in candidates:
                        flat_tokens.append(tokens)
                        flat_embeddings.append(item.global_embedding)
                        flat_item_indices.append(item.index)
                global_embeddings = torch.stack(flat_embeddings, dim=0)
                audio_batch, audio_lengths = codec_service.synthesize_batch(
                    flat_tokens,
                    global_embedding=global_embeddings,
                )
            except Exception as exc:
                logger.exception("Codec batch synthesis failed")
                for item in decodable_items:
                    responses[item.index] = BatchTTSItemResponse(
                        error=f"codec batch synthesis failed: {exc}"
                    )
            else:
                codec_end = time.perf_counter()
                codec_total_sec = max(0.0, codec_end - codec_start)
                codec_per_candidate_sec = codec_total_sec / max(1, len(flat_tokens))
                lengths = (
                    audio_lengths.tolist() if hasattr(audio_lengths, "tolist") else list(audio_lengths)
                )
                codec_sample_rate = codec_service.sample_rate
                grouped_candidates: dict[int, list[BestOfNCandidate]] = {
                    item.index: [] for item in decodable_items
                }
                flat_by_item: dict[int, int] = {item.index: 0 for item in decodable_items}

                for batch_idx, item_index in enumerate(flat_item_indices):
                    try:
                        audio_len = int(lengths[batch_idx]) if lengths else audio_batch.shape[1]
                        audio = audio_batch[batch_idx, :audio_len]
                        item = by_index[item_index]
                        candidate_idx = flat_by_item[item_index]
                        flat_by_item[item_index] = candidate_idx + 1
                        token_candidates = item.token_candidates or []
                        grouped_candidates[item_index].append(
                            BestOfNCandidate(
                                tokens=token_candidates[candidate_idx],
                                audio=audio,
                            )
                        )
                    except Exception as exc:
                        responses[item_index] = BatchTTSItemResponse(
                            error=f"candidate decode mapping failed: {exc}"
                        )

                async def _finalize_item(
                    item: _PreparedBatchSynthesisItem,
                ) -> tuple[int, BatchTTSItemResponse]:
                    candidates = grouped_candidates.get(item.index, [])
                    if not candidates:
                        return item.index, BatchTTSItemResponse(error="no decoded candidates")

                    best_of_n_sec = None
                    asr_sec = None
                    selected = candidates[0]
                    if best_of_n.enabled and len(candidates) > 1:
                        if asr_service is None:
                            return item.index, BatchTTSItemResponse(
                                error="ASR is not available on the server"
                            )
                        rank_start = time.perf_counter()
                        best_idx, asr_sec = await score_candidates(
                            text=item.normalized_text,
                            candidates=candidates,
                            sample_rate=codec_sample_rate,
                            language=best_of_n.language,
                            asr_service=asr_service,
                        )
                        rank_end = time.perf_counter()
                        best_of_n_sec = rank_end - rank_start
                        if asr_sec is not None:
                            best_of_n_sec = max(0.0, best_of_n_sec - asr_sec)
                        selected = candidates[best_idx]

                    post_start = time.perf_counter()
                    audio = selected.audio
                    if abs(float(item.speech_rate) - 1.0) > 1e-6:
                        audio = _apply_speech_rate(audio, float(item.speech_rate), codec_sample_rate)
                    post_end = time.perf_counter()
                    item_codec_sec = codec_per_candidate_sec * max(1, len(candidates))
                    total_sec = (
                        item.llm_sec
                        + item.parse_sec
                        + item_codec_sec
                        + (best_of_n_sec or 0.0)
                        + (asr_sec or 0.0)
                        + (post_end - post_start)
                    )
                    wav_bytes = write_wav_bytes(audio, codec_sample_rate)
                    return item.index, BatchTTSItemResponse(
                        audio=base64.b64encode(wav_bytes).decode("ascii"),
                        format="base64",
                        sample_rate=codec_sample_rate,
                        token_count=len(selected.tokens),
                        timings=TTSTimings(
                            llm_sec=round(item.llm_sec, 4),
                            parse_sec=round(item.parse_sec, 4),
                            codec_sec=round(item_codec_sec, 4),
                            total_sec=round(total_sec, 4),
                            best_of_n_sec=round(best_of_n_sec, 4) if best_of_n_sec is not None else None,
                            asr_sec=round(asr_sec, 4) if asr_sec is not None else None,
                        ),
                        normalized_text=item.normalized_text,
                        error=None,
                    )

                finalized = await asyncio.gather(
                    *[_finalize_item(item) for item in decodable_items],
                    return_exceptions=True,
                )
                for item, result in zip(decodable_items, finalized, strict=False):
                    if isinstance(result, Exception):
                        responses[item.index] = BatchTTSItemResponse(
                            error=f"finalize failed: {result}"
                        )
                        continue
                    item_index, response = result
                    responses[item_index] = response

    return JSONResponse(content=BatchTTSResponse(items=responses).model_dump())


@dataclass
class _ResolvedBestOfN:
    enabled: bool
    n: int
    language: str


def _resolve_best_of_n(request: TTSRequest, config) -> _ResolvedBestOfN:
    if not config.best_of_n_enabled:
        if request.best_of_n and request.best_of_n.enabled:
            raise HTTPException(status_code=400, detail="best_of_n is disabled on the server")
        return _ResolvedBestOfN(
            enabled=False,
            n=1,
            language=config.best_of_n_language,
        )

    req = request.best_of_n or BestOfNConfig()
    n = req.n if req.n is not None else config.best_of_n_default
    n = max(1, min(n, config.best_of_n_max))
    enabled = req.enabled if req.enabled is not None else (n > 1)
    language = (req.language or config.best_of_n_language).lower()
    if language not in {"ja", "en", "auto"}:
        language = "auto"
    if not enabled:
        n = 1
    return _ResolvedBestOfN(
        enabled=enabled,
        n=n,
        language=language,
    )


def _parse_llm_candidates(llm_texts: list[str]) -> list[list[int]]:
    tokens_list: list[list[int]] = []
    for llm_text in llm_texts:
        try:
            tokens_list.append(parse_speech_tokens(llm_text))
        except ValueError as exc:
            logger.warning("Skipping candidate with invalid tokens: %s", exc)
    return tokens_list


async def _fetch_llm_candidates(
    llm_client: LLMClient,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    n: int,
) -> list[str]:
    if n <= 1:
        text = await llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return [text]

    tasks = [
        llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    texts: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("LLM candidate failed: %s", result)
            continue
        texts.append(result)
    if not texts:
        raise RuntimeError("All LLM candidate requests failed.")
    return texts


async def _generate_token_candidates_for_text(
    normalized: str,
    llm_client: LLMClient,
    llm_settings: _ResolvedLLMSettings,
    n: int,
) -> tuple[list[list[int]], float, float]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": normalized}]
    t0 = time.perf_counter()
    llm_texts = await _fetch_llm_candidates(
        llm_client=llm_client,
        messages=messages,
        model=llm_settings.model,
        temperature=llm_settings.temperature,
        top_p=llm_settings.top_p,
        top_k=llm_settings.top_k,
        max_tokens=llm_settings.max_tokens,
        repetition_penalty=llm_settings.repetition_penalty,
        presence_penalty=llm_settings.presence_penalty,
        frequency_penalty=llm_settings.frequency_penalty,
        n=n,
    )
    t1 = time.perf_counter()

    tokens_list = _parse_llm_candidates(llm_texts)
    if not tokens_list:
        strict_messages = [
            {"role": "system", "content": SPEECH_TOKEN_SYSTEM_PROMPT},
            {"role": "user", "content": normalized},
        ]
        llm_texts_retry = await _fetch_llm_candidates(
            llm_client=llm_client,
            messages=strict_messages,
            model=llm_settings.model,
            temperature=min(float(llm_settings.temperature), 0.4),
            top_p=min(float(llm_settings.top_p), 0.95),
            top_k=llm_settings.top_k,
            max_tokens=llm_settings.max_tokens,
            repetition_penalty=llm_settings.repetition_penalty,
            presence_penalty=llm_settings.presence_penalty,
            frequency_penalty=llm_settings.frequency_penalty,
            n=n,
        )
        t1 = time.perf_counter()
        tokens_list = _parse_llm_candidates(llm_texts_retry)
    if not tokens_list:
        raise RuntimeError("No speech tokens found in LLM output.")
    t2 = time.perf_counter()
    return tokens_list, (t1 - t0), (t2 - t1)


async def _generate_tokens_for_text(
    normalized: str,
    llm_client: LLMClient,
    llm_settings: _ResolvedLLMSettings,
    n: int,
) -> tuple[list[int], float, float]:
    token_candidates, llm_sec, parse_sec = await _generate_token_candidates_for_text(
        normalized=normalized,
        llm_client=llm_client,
        llm_settings=llm_settings,
        n=n,
    )
    return token_candidates[0], llm_sec, parse_sec


async def _read_reference_file(file: UploadFile) -> bytes:
    config = get_config()
    audio_config = get_audio_config()
    max_bytes = config.max_reference_mb * 1024 * 1024
    if file.filename:
        ext = ("." + file.filename.rsplit(".", 1)[-1]).lower() if "." in file.filename else ""
        if ext not in audio_config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported audio extension: {ext}",
            )
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"reference audio too large (max {config.max_reference_mb} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _strip_base64_whitespace(data: str) -> str:
    return "".join(data.split())


def _estimate_base64_decoded_size(data: str) -> int:
    if not data:
        return 0
    padding_len = len(data) - len(data.rstrip("="))
    return max(0, (len(data) * 3) // 4 - padding_len)


def _resolve_output_format(output: OutputConfig | None, default_format: str) -> str:
    if output and output.format:
        return output.format
    return default_format


def _normalize_text_for_tts(text: str, config) -> str:
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > config.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"text is too long (max {config.max_text_length} characters)",
        )
    detected_language = detect_language(text)
    if detected_language == "ja":
        return normalize_text(text)
    return text.strip()


async def _resolve_llm_settings(
    llm_params: LLMParams | None,
    config,
    llm_defaults,
    llm_client: LLMClient,
) -> _ResolvedLLMSettings:
    llm_params = llm_params or LLMParams()
    model = llm_params.model or config.llm_model
    temperature = (
        llm_params.temperature if llm_params.temperature is not None else llm_defaults.temperature
    )
    top_p = llm_params.top_p if llm_params.top_p is not None else llm_defaults.top_p
    top_k = llm_params.top_k if llm_params.top_k is not None else llm_defaults.top_k
    max_tokens = (
        llm_params.max_tokens if llm_params.max_tokens is not None else llm_defaults.max_tokens
    )
    repetition_penalty = (
        llm_params.repetition_penalty
        if llm_params.repetition_penalty is not None
        else llm_defaults.repetition_penalty
    )
    presence_penalty = (
        llm_params.presence_penalty
        if llm_params.presence_penalty is not None
        else llm_defaults.presence_penalty
    )
    frequency_penalty = (
        llm_params.frequency_penalty
        if llm_params.frequency_penalty is not None
        else llm_defaults.frequency_penalty
    )
    if not model:
        try:
            model = await llm_client.resolve_model(model)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to resolve LLM model: {exc}"
            ) from exc
    return _ResolvedLLMSettings(
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )


def _resolve_batch_reference_embedding(
    item: BatchTTSItem,
    codec_service: MioCodecService,
) -> torch.Tensor:
    if item.reference is None:
        raise HTTPException(status_code=400, detail="reference is required")
    if item.reference.type != "preset":
        raise HTTPException(
            status_code=400,
            detail="batch endpoint supports only preset references",
        )
    preset_id = item.reference.preset_id
    if not preset_id:
        raise HTTPException(status_code=400, detail="reference.preset_id is required")
    try:
        return codec_service.load_preset_embedding(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _http_exception_detail_text(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, str):
        return detail
    return str(detail)


def _largest_power_of_two_leq(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (value.bit_length() - 1)


def _select_time_stretch_n_fft(src_len: int, sample_rate: int) -> int:
    target = 2048
    if sample_rate > 0:
        target = int(round(sample_rate * 0.046))
    target = min(4096, max(256, target))
    return _largest_power_of_two_leq(min(src_len, target))


def _phase_vocoder_time_stretch(
    complex_spec: torch.Tensor, speech_rate: float, hop_length: int
) -> torch.Tensor:
    frame_count = int(complex_spec.shape[-1])
    if frame_count <= 1:
        return complex_spec

    complex_spec = torch.cat([complex_spec, complex_spec[..., -1:]], dim=-1)
    real_dtype = complex_spec.real.dtype
    time_steps = torch.arange(
        0,
        frame_count,
        speech_rate,
        device=complex_spec.device,
        dtype=real_dtype,
    )
    if time_steps.numel() == 0:
        time_steps = torch.zeros(1, device=complex_spec.device, dtype=real_dtype)

    time_indices = torch.floor(time_steps).to(torch.long)
    alphas = (time_steps - time_indices.to(real_dtype)).unsqueeze(0)

    spec0 = complex_spec[..., time_indices]
    spec1 = complex_spec[..., time_indices + 1]
    magnitudes = (1.0 - alphas) * spec0.abs() + alphas * spec1.abs()

    phase_advance = torch.linspace(
        0.0,
        math.pi * hop_length,
        complex_spec.shape[-2],
        device=complex_spec.device,
        dtype=real_dtype,
    ).unsqueeze(-1)
    phase0 = torch.angle(spec0)
    phase1 = torch.angle(spec1)
    delta_phase = phase1 - phase0 - phase_advance
    two_pi = 2.0 * math.pi
    delta_phase = delta_phase - two_pi * torch.round(delta_phase / two_pi)
    delta_phase = delta_phase + phase_advance

    phase = torch.cat([phase0[..., :1], delta_phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, dim=-1)
    return torch.polar(magnitudes, phase_acc)


def _apply_speech_rate(audio: torch.Tensor, speech_rate: float, sample_rate: int) -> torch.Tensor:
    if speech_rate <= 0:
        return audio
    audio = audio.float().flatten()
    src_len = int(audio.numel())
    if src_len <= 1:
        return audio
    dst_len = max(1, int(round(src_len / speech_rate)))
    if dst_len == src_len:
        return audio

    n_fft = _select_time_stretch_n_fft(src_len, sample_rate)
    if n_fft < 64:
        return F.interpolate(audio.view(1, 1, -1), size=dst_len, mode="linear", align_corners=False).view(-1)

    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    complex_spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    stretched_spec = _phase_vocoder_time_stretch(complex_spec, speech_rate, hop_length)
    stretched = torch.istft(
        stretched_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=dst_len,
    )
    if stretched.numel() != dst_len:
        stretched = F.interpolate(
            stretched.view(1, 1, -1),
            size=dst_len,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return stretched


def _trim_reference(waveform: torch.Tensor, sample_rate: int, max_seconds: float) -> torch.Tensor:
    if max_seconds <= 0:
        return waveform
    max_samples = int(sample_rate * max_seconds)
    if waveform.numel() > max_samples:
        original_sec = waveform.numel() / sample_rate
        logger.info(
            "Reference audio trimmed: %.2fs -> %.2fs (max %.2fs)",
            original_sec,
            max_seconds,
            max_seconds,
        )
        return waveform[:max_samples]
    return waveform


def _configure_torch(config):
    try:
        import torch
    except Exception:
        return
    uvicorn_logger = logging.getLogger("uvicorn.error")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    uvicorn_logger.info("Enabled TF32 matmul/cudnn")
