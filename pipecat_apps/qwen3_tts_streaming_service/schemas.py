"""Request and response schemas for the standalone streaming TTS service."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class StreamingOptions(BaseModel):
    """Client-overridable parameters for incremental audio emission."""

    emit_every_frames: Optional[int] = Field(default=None, ge=1)
    decode_window_frames: Optional[int] = Field(default=None, ge=1)
    overlap_samples: Optional[int] = Field(default=None, ge=0)
    max_frames: Optional[int] = Field(default=None, ge=1)
    first_chunk_emit_every: Optional[int] = Field(default=None, ge=0)
    first_chunk_decode_window: Optional[int] = Field(default=None, ge=1)
    first_chunk_frames: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0)
    repetition_penalty_window: Optional[int] = Field(default=None, ge=0)
    use_optimized_decode: Optional[bool] = None


class TTSStreamRequest(BaseModel):
    """Incoming WebSocket request for streaming synthesis.

    Two synthesis paths are supported depending on the loaded model:

    Voice-clone path (Base model):
        Provide one of: prompt_id, reference_audio_base64, or rely on a
        preloaded service default prompt.

    CustomVoice path (CustomVoice model):
        Provide speaker (required) and optionally instruct.
        prompt_id / reference_audio_base64 are ignored.
    """

    text: str = Field(..., min_length=1)
    language: str = Field(default="Auto")
    # Voice-clone fields
    prompt_id: Optional[str] = None
    reference_audio_base64: Optional[str] = None
    reference_text: Optional[str] = None
    x_vector_only_mode: bool = False
    # CustomVoice fields
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    response_audio_transport: Literal["binary", "base64"] = "binary"
    streaming: StreamingOptions = Field(default_factory=StreamingOptions)

    @model_validator(mode="after")
    def validate_prompt_source(self) -> "TTSStreamRequest":
        """Validate prompt-related fields.

        The runtime may fall back to a preloaded default prompt when neither `prompt_id`
        nor `reference_audio_base64` is provided, so this validator only enforces the
        fields required for inline prompt creation.
        """

        if self.reference_audio_base64 and not self.x_vector_only_mode and not self.reference_text:
            raise ValueError(
                "reference_text is required when using inline reference audio in ICL mode."
            )
        return self


class ClonePromptResponse(BaseModel):
    """Response returned after building and caching a clone prompt."""

    prompt_id: str
    cached: bool = True
    x_vector_only_mode: bool
    reference_text_provided: bool


class HealthResponse(BaseModel):
    """High-level service health response."""

    ok: bool
    model_loaded: bool
    active_streams: int
    prompt_cache_size: int
    model_path: str
    tts_model_type: Optional[str] = None
    last_error: Optional[str] = None
    default_prompt_keys: list[str] = Field(default_factory=list)
    config: Dict[str, Any]


class StreamStartEvent(BaseModel):
    """Control event that starts a WebSocket stream."""

    type: Literal["stream_start"] = "stream_start"
    request_id: str
    audio_format: Dict[str, Any]


class StreamChunkEvent(BaseModel):
    """Chunk metadata event for JSON or binary WebSocket transport."""

    type: Literal["chunk"] = "chunk"
    request_id: str
    sequence: int
    num_samples: int
    num_bytes: int
    audio_base64: Optional[str] = None


class StreamEndEvent(BaseModel):
    """Control event that marks successful stream completion."""

    type: Literal["stream_end"] = "stream_end"
    request_id: str
    total_chunks: int


class StreamErrorEvent(BaseModel):
    """Control event sent when synthesis fails."""

    type: Literal["error"] = "error"
    request_id: Optional[str] = None
    message: str
