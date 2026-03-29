"""FastAPI application entrypoint for the standalone streaming TTS service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from .config import ServiceConfig
from .runtime import QwenStreamingRuntime, TTSRuntime
from .service import create_router

LOGGER = logging.getLogger(__name__)


def create_app(
    config: Optional[ServiceConfig] = None,
    runtime: Optional[TTSRuntime] = None,
) -> FastAPI:
    """Create a FastAPI app with lazy model loading and optional warmup."""

    resolved_config = config or ServiceConfig.from_env()
    resolved_runtime = runtime or QwenStreamingRuntime(resolved_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if resolved_config.load_on_startup:
            try:
                resolved_runtime.ensure_loaded()
                resolved_runtime.preload_default_prompts()
                resolved_runtime.warmup()
            except Exception as exc:
                # The service is allowed to start even when model loading fails so `/health`
                # can explain the failure to operators without requiring local audio devices.
                resolved_runtime.record_error(exc)
                LOGGER.exception("Startup initialization failed: %s", exc)
        yield

    app = FastAPI(
        title="Qwen3 Streaming TTS Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = resolved_runtime
    app.state.config = resolved_config
    app.include_router(create_router(resolved_config, resolved_runtime))
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entrypoint.
    import uvicorn

    settings = ServiceConfig.from_env()
    uvicorn.run(
        "pipecat_apps.qwen3_tts_streaming_service.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
