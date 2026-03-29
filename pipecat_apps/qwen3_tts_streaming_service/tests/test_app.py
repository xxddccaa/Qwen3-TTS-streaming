"""Smoke tests for the standalone FastAPI streaming TTS service."""

from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from pipecat_apps.qwen3_tts_streaming_service.app import create_app
from pipecat_apps.qwen3_tts_streaming_service.config import ServiceConfig
from pipecat_apps.qwen3_tts_streaming_service.runtime import PCMChunk, PromptRecord, RuntimeHealth


class FakeRuntime:
    """Small fake runtime used to validate the API contract without loading a model."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._prompt = PromptRecord(
            prompt_id="prompt-test-1",
            prompt_items=["fake-prompt"],
            reference_text="reference text",
            x_vector_only_mode=False,
        )
        self.default_prompt_keys = ["chinese", "english"]
        self.errors: list[str] = []

    def ensure_loaded(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def preload_default_prompts(self) -> None:
        return None

    def record_error(self, exc: Exception) -> None:
        self.errors.append(str(exc))

    def create_clone_prompt(
        self,
        audio_bytes: bytes,
        reference_text: str | None,
        x_vector_only_mode: bool,
    ) -> PromptRecord:
        assert audio_bytes
        self._prompt = PromptRecord(
            prompt_id="prompt-uploaded-1",
            prompt_items=["fake-prompt"],
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        return self._prompt

    def stream_tts(self, request, stop_event=None):
        del request
        chunk_a = np.array([0.0, 0.25, -0.25], dtype=np.float32)
        chunk_b = np.array([0.5, -0.5], dtype=np.float32)
        yield PCMChunk(sequence=0, sample_rate=24_000, samples=chunk_a)
        yield PCMChunk(sequence=1, sample_rate=24_000, samples=chunk_b)

    def health(self) -> RuntimeHealth:
        return RuntimeHealth(
            model_loaded=True,
            active_streams=0,
            prompt_cache_size=1,
            model_path=self.config.model_path,
            last_error=self.errors[-1] if self.errors else None,
            default_prompt_keys=self.default_prompt_keys,
            config=self.config.summary(),
        )


def _build_test_wav_bytes() -> bytes:
    samples = np.array([0, 2000, -2000, 1000, -1000], dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()


def _build_client() -> TestClient:
    config = ServiceConfig(load_on_startup=False)
    runtime = FakeRuntime(config)
    app = create_app(config=config, runtime=runtime)
    return TestClient(app)


def test_health_endpoint() -> None:
    with _build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["model_loaded"] is True
    assert payload["default_prompt_keys"] == ["chinese", "english"]
    assert payload["config"]["model_path"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def test_clone_prompt_endpoint() -> None:
    with _build_client() as client:
        response = client.post(
            "/v1/clone-prompts",
            data={"reference_text": "hello reference", "x_vector_only_mode": "false"},
            files={"reference_audio": ("reference.wav", _build_test_wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_id"] == "prompt-uploaded-1"
    assert payload["cached"] is True
    assert payload["reference_text_provided"] is True


def test_websocket_stream_binary_protocol() -> None:
    with _build_client() as client:
        with client.websocket_connect("/v1/tts/stream") as websocket:
            websocket.send_json(
                {
                    "text": "hello world",
                    "language": "English",
                    "prompt_id": "prompt-test-1",
                    "response_audio_transport": "binary",
                }
            )

            start_event = websocket.receive_json()
            first_chunk_meta = websocket.receive_json()
            first_chunk_bytes = websocket.receive_bytes()
            second_chunk_meta = websocket.receive_json()
            second_chunk_bytes = websocket.receive_bytes()
            end_event = websocket.receive_json()

    assert start_event["type"] == "stream_start"
    assert start_event["audio_format"]["encoding"] == "pcm_s16le"
    assert first_chunk_meta["type"] == "chunk"
    assert first_chunk_meta["sequence"] == 0
    assert first_chunk_meta["num_bytes"] == len(first_chunk_bytes)
    assert len(first_chunk_bytes) == 6
    assert second_chunk_meta["sequence"] == 1
    assert second_chunk_meta["num_bytes"] == len(second_chunk_bytes)
    assert len(second_chunk_bytes) == 4
    assert end_event["type"] == "stream_end"
    assert end_event["total_chunks"] == 2


def test_websocket_stream_uses_default_prompt_when_missing_prompt_id() -> None:
    with _build_client() as client:
        with client.websocket_connect("/v1/tts/stream") as websocket:
            websocket.send_json(
                {
                    "text": "hello default prompt",
                    "language": "English",
                    "response_audio_transport": "binary",
                }
            )

            start_event = websocket.receive_json()
            first_chunk_meta = websocket.receive_json()
            first_chunk_bytes = websocket.receive_bytes()
            end_or_next = websocket.receive_json()

            if end_or_next["type"] == "chunk":
                websocket.receive_bytes()
                end_event = websocket.receive_json()
            else:
                end_event = end_or_next

    assert start_event["type"] == "stream_start"
    assert first_chunk_meta["type"] == "chunk"
    assert len(first_chunk_bytes) > 0
    assert end_event["type"] == "stream_end"
