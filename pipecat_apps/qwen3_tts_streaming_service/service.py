"""HTTP and WebSocket endpoints for the standalone streaming TTS service."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import uuid
from typing import Any, Dict, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .config import ServiceConfig
from .runtime import PCMChunk, TTSRuntime
from .schemas import (
    ClonePromptResponse,
    HealthResponse,
    StreamChunkEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TTSStreamRequest,
)

LOGGER = logging.getLogger(__name__)


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def create_router(config: ServiceConfig, runtime: TTSRuntime) -> APIRouter:
    """Create the API router bound to a concrete runtime implementation."""

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        snapshot = runtime.health()
        return HealthResponse(
            ok=snapshot.model_loaded,
            model_loaded=snapshot.model_loaded,
            active_streams=snapshot.active_streams,
            prompt_cache_size=snapshot.prompt_cache_size,
            model_path=snapshot.model_path,
            tts_model_type=snapshot.tts_model_type,
            last_error=snapshot.last_error,
            default_prompt_keys=snapshot.default_prompt_keys,
            config=snapshot.config,
        )

    @router.post("/v1/clone-prompts", response_model=ClonePromptResponse)
    async def create_clone_prompt(
        reference_audio: UploadFile = File(...),
        reference_text: str = Form(...),
        x_vector_only_mode: bool = Form(False),
    ) -> ClonePromptResponse:
        audio_bytes = await reference_audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="reference_audio must not be empty.")

        try:
            record = await asyncio.to_thread(
                runtime.create_clone_prompt,
                audio_bytes,
                reference_text if not x_vector_only_mode else None,
                x_vector_only_mode,
            )
        except Exception as exc:
            runtime.record_error(exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ClonePromptResponse(
            prompt_id=record.prompt_id,
            cached=True,
            x_vector_only_mode=record.x_vector_only_mode,
            reference_text_provided=bool(record.reference_text),
        )

    @router.websocket("/v1/tts/stream")
    async def tts_stream(websocket: WebSocket) -> None:
        # WebSocket is preferred here because this service emits incremental binary PCM chunks
        # and may later support duplex control messages without changing the transport model.
        await websocket.accept()
        request_id = str(uuid.uuid4())

        try:
            payload = await websocket.receive_json()
            request = TTSStreamRequest(**payload)
            await asyncio.to_thread(runtime.ensure_loaded)

            await websocket.send_json(
                _model_to_dict(
                    StreamStartEvent(
                        request_id=request_id,
                        audio_format={
                            "encoding": config.output_encoding,
                            "sample_rate": config.output_sample_rate,
                            "channels": config.output_channels,
                            "transport": request.response_audio_transport,
                        },
                    )
                )
            )

            await _stream_over_websocket(
                websocket=websocket,
                runtime=runtime,
                request=request,
                request_id=request_id,
            )
        except WebSocketDisconnect:
            LOGGER.info("WebSocket disconnected before stream completion: %s", request_id)
        except Exception as exc:
            runtime.record_error(exc)
            await websocket.send_json(
                _model_to_dict(
                    StreamErrorEvent(
                        request_id=request_id,
                        message=str(exc),
                    )
                )
            )
            await websocket.close(code=1011)

    return router


async def _stream_over_websocket(
    websocket: WebSocket,
    runtime: TTSRuntime,
    request: TTSStreamRequest,
    request_id: str,
) -> None:
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Tuple[str, Any]]" = asyncio.Queue()
    stop_event = threading.Event()

    def worker() -> None:
        chunk_count = 0
        try:
            for chunk in runtime.stream_tts(request, stop_event=stop_event):
                chunk_count += 1
                asyncio.run_coroutine_threadsafe(
                    queue.put(("chunk", chunk)),
                    loop,
                ).result()

            asyncio.run_coroutine_threadsafe(
                queue.put(
                    (
                        "end",
                        StreamEndEvent(
                            request_id=request_id,
                            total_chunks=chunk_count,
                        ),
                    )
                ),
                loop,
            ).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", exc)),
                loop,
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

    worker_thread = threading.Thread(target=worker, daemon=True, name="qwen3-tts-stream")
    worker_thread.start()

    try:
        while True:
            event_type, payload = await queue.get()
            if event_type == "done":
                break
            if event_type == "error":
                raise payload
            if event_type == "end":
                await websocket.send_json(_model_to_dict(payload))
                break
            if event_type == "chunk":
                await _send_chunk(
                    websocket=websocket,
                    request_id=request_id,
                    request=request,
                    chunk=payload,
                )
    except WebSocketDisconnect:
        stop_event.set()
        raise
    finally:
        stop_event.set()


async def _send_chunk(
    websocket: WebSocket,
    request_id: str,
    request: TTSStreamRequest,
    chunk: PCMChunk,
) -> None:
    payload = chunk.to_pcm_s16le_bytes()
    if request.response_audio_transport == "base64":
        await websocket.send_json(
            _model_to_dict(
                StreamChunkEvent(
                    request_id=request_id,
                    sequence=chunk.sequence,
                    num_samples=len(chunk.samples),
                    num_bytes=len(payload),
                    audio_base64=base64.b64encode(payload).decode("ascii"),
                )
            )
        )
        return

    await websocket.send_json(
        _model_to_dict(
            StreamChunkEvent(
                request_id=request_id,
                sequence=chunk.sequence,
                num_samples=len(chunk.samples),
                num_bytes=len(payload),
            )
        )
    )
    await websocket.send_bytes(payload)
