"""Environment-driven configuration for the standalone streaming TTS service."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _get_optional_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


@dataclass(slots=True)
class StreamingDefaults:
    """Default streaming parameters used when the client does not override them."""

    emit_every_frames: int = 8
    decode_window_frames: int = 80
    overlap_samples: int = 512
    max_frames: int = 10_000
    first_chunk_emit_every: int = 5
    first_chunk_decode_window: int = 48
    first_chunk_frames: int = 48
    repetition_penalty: float = 1.0
    repetition_penalty_window: int = 100
    use_optimized_decode: bool = True


@dataclass(slots=True)
class ServiceConfig:
    """Runtime configuration for the standalone FastAPI service."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device_map: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    load_on_startup: bool = True
    warmup_enabled: bool = False
    warmup_reference_audio_path: Optional[str] = None
    warmup_reference_text: Optional[str] = None
    warmup_text: str = "Hello, this is a streaming warmup request."
    enable_streaming_optimizations: bool = True
    compile_mode: str = "reduce-overhead"
    use_compile: bool = True
    use_cuda_graphs: bool = False
    use_fast_codebook: bool = False
    compile_codebook_predictor: bool = True
    prompt_cache_max_entries: int = 256
    default_auto_reference_audio_path: Optional[str] = None
    default_auto_reference_text: Optional[str] = None
    default_chinese_reference_audio_path: Optional[str] = None
    default_chinese_reference_text: Optional[str] = None
    default_english_reference_audio_path: Optional[str] = None
    default_english_reference_text: Optional[str] = None
    output_encoding: str = "pcm_s16le"
    output_sample_rate: int = 24_000
    output_channels: int = 1
    default_audio_transport: str = "binary"
    streaming: StreamingDefaults = field(default_factory=StreamingDefaults)

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables."""

        return cls(
            host=os.getenv("QWEN_TTS_SERVICE_HOST", "0.0.0.0"),
            port=_get_int("QWEN_TTS_SERVICE_PORT", 8000),
            log_level=os.getenv("QWEN_TTS_SERVICE_LOG_LEVEL", "info"),
            model_path=os.getenv("QWEN_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            device_map=os.getenv("QWEN_TTS_DEVICE_MAP", "cuda:0"),
            torch_dtype=os.getenv("QWEN_TTS_TORCH_DTYPE", "bfloat16"),
            attn_implementation=_get_optional_str("QWEN_TTS_ATTN_IMPLEMENTATION") or "flash_attention_2",
            load_on_startup=_get_bool("QWEN_TTS_LOAD_ON_STARTUP", True),
            warmup_enabled=_get_bool("QWEN_TTS_WARMUP_ENABLED", False),
            warmup_reference_audio_path=_get_optional_str("QWEN_TTS_WARMUP_REFERENCE_AUDIO_PATH"),
            warmup_reference_text=_get_optional_str("QWEN_TTS_WARMUP_REFERENCE_TEXT"),
            warmup_text=os.getenv(
                "QWEN_TTS_WARMUP_TEXT",
                "Hello, this is a streaming warmup request.",
            ),
            enable_streaming_optimizations=_get_bool("QWEN_TTS_ENABLE_STREAMING_OPTIMIZATIONS", True),
            compile_mode=os.getenv("QWEN_TTS_COMPILE_MODE", "reduce-overhead"),
            use_compile=_get_bool("QWEN_TTS_USE_COMPILE", True),
            use_cuda_graphs=_get_bool("QWEN_TTS_USE_CUDA_GRAPHS", False),
            use_fast_codebook=_get_bool("QWEN_TTS_USE_FAST_CODEBOOK", False),
            compile_codebook_predictor=_get_bool("QWEN_TTS_COMPILE_CODEBOOK_PREDICTOR", True),
            prompt_cache_max_entries=_get_int("QWEN_TTS_PROMPT_CACHE_MAX_ENTRIES", 256),
            default_auto_reference_audio_path=_get_optional_str("QWEN_TTS_DEFAULT_AUTO_REFERENCE_AUDIO_PATH"),
            default_auto_reference_text=_get_optional_str("QWEN_TTS_DEFAULT_AUTO_REFERENCE_TEXT"),
            default_chinese_reference_audio_path=_get_optional_str("QWEN_TTS_DEFAULT_CHINESE_REFERENCE_AUDIO_PATH"),
            default_chinese_reference_text=_get_optional_str("QWEN_TTS_DEFAULT_CHINESE_REFERENCE_TEXT"),
            default_english_reference_audio_path=_get_optional_str("QWEN_TTS_DEFAULT_ENGLISH_REFERENCE_AUDIO_PATH"),
            default_english_reference_text=_get_optional_str("QWEN_TTS_DEFAULT_ENGLISH_REFERENCE_TEXT"),
            output_encoding=os.getenv("QWEN_TTS_OUTPUT_ENCODING", "pcm_s16le"),
            output_sample_rate=_get_int("QWEN_TTS_OUTPUT_SAMPLE_RATE", 24_000),
            output_channels=_get_int("QWEN_TTS_OUTPUT_CHANNELS", 1),
            default_audio_transport=os.getenv("QWEN_TTS_DEFAULT_AUDIO_TRANSPORT", "binary"),
            streaming=StreamingDefaults(
                emit_every_frames=_get_int("QWEN_TTS_EMIT_EVERY_FRAMES", 8),
                decode_window_frames=_get_int("QWEN_TTS_DECODE_WINDOW_FRAMES", 80),
                overlap_samples=_get_int("QWEN_TTS_OVERLAP_SAMPLES", 512),
                max_frames=_get_int("QWEN_TTS_MAX_FRAMES", 10_000),
                first_chunk_emit_every=_get_int("QWEN_TTS_FIRST_CHUNK_EMIT_EVERY", 5),
                first_chunk_decode_window=_get_int("QWEN_TTS_FIRST_CHUNK_DECODE_WINDOW", 48),
                first_chunk_frames=_get_int("QWEN_TTS_FIRST_CHUNK_FRAMES", 48),
                repetition_penalty=_get_float("QWEN_TTS_REPETITION_PENALTY", 1.0),
                repetition_penalty_window=_get_int("QWEN_TTS_REPETITION_PENALTY_WINDOW", 100),
                use_optimized_decode=_get_bool("QWEN_TTS_USE_OPTIMIZED_DECODE", True),
            ),
        )

    def summary(self) -> Dict[str, Any]:
        """Return a small config summary that is safe to expose in health responses."""

        data = asdict(self)
        data["warmup_reference_audio_path"] = bool(self.warmup_reference_audio_path)
        data["warmup_reference_text"] = bool(self.warmup_reference_text)
        data["default_auto_reference_audio_path"] = bool(self.default_auto_reference_audio_path)
        data["default_auto_reference_text"] = bool(self.default_auto_reference_text)
        data["default_chinese_reference_audio_path"] = bool(self.default_chinese_reference_audio_path)
        data["default_chinese_reference_text"] = bool(self.default_chinese_reference_text)
        data["default_english_reference_audio_path"] = bool(self.default_english_reference_audio_path)
        data["default_english_reference_text"] = bool(self.default_english_reference_text)
        return data
