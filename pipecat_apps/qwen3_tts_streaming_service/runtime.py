"""Runtime layer that owns model loading, prompt caching, and streaming generation."""

from __future__ import annotations

import base64
import io
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Protocol

import numpy as np
import soundfile as sf

from .config import ServiceConfig, StreamingDefaults
from .schemas import TTSStreamRequest

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptRecord:
    """In-memory prompt cache entry.

    The prompt is intentionally cached in-process first because prompt extraction is expensive
    and many chat turns reuse the same voice. If you later need multi-instance deployments,
    this is the boundary where Redis or a database-backed prompt store should be introduced.
    """

    prompt_id: str
    prompt_items: List[Any]
    reference_text: Optional[str]
    x_vector_only_mode: bool


@dataclass(slots=True)
class PCMChunk:
    """One PCM chunk produced by streaming generation."""

    sequence: int
    sample_rate: int
    samples: np.ndarray

    def to_pcm_s16le_bytes(self) -> bytes:
        """Convert float32 mono samples to little-endian signed 16-bit PCM."""

        clipped = np.clip(self.samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2", copy=False)
        return pcm.tobytes()


@dataclass(slots=True)
class RuntimeHealth:
    """Internal health snapshot returned to the HTTP layer."""

    model_loaded: bool
    active_streams: int
    prompt_cache_size: int
    model_path: str
    tts_model_type: Optional[str]
    last_error: Optional[str]
    default_prompt_keys: List[str]
    config: Dict[str, Any]


class TTSRuntime(Protocol):
    """Minimal protocol used by the FastAPI layer and tests."""

    def ensure_loaded(self) -> None:
        """Load the model if it is not already available."""

    def create_clone_prompt(
        self,
        audio_bytes: bytes,
        reference_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> PromptRecord:
        """Build and cache a reusable voice clone prompt."""

    def stream_tts(
        self,
        request: TTSStreamRequest,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[PCMChunk, None, None]:
        """Generate streaming PCM chunks for one request."""

    def health(self) -> RuntimeHealth:
        """Return a snapshot suitable for `/health`."""

    def warmup(self) -> None:
        """Optionally run a warmup request after model load."""

    def preload_default_prompts(self) -> None:
        """Prebuild and cache default prompts for no-parameter TTS requests."""

    def record_error(self, exc: Exception) -> None:
        """Persist the latest startup/runtime error for observability."""


class QwenStreamingRuntime:
    """Default runtime backed by the repository's streaming voice clone path."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._model_wrapper: Optional[Any] = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._prompt_cache: "OrderedDict[str, PromptRecord]" = OrderedDict()
        self._default_prompt_ids: Dict[str, str] = {}
        self._active_streams = 0
        self._last_error: Optional[str] = None

    def ensure_loaded(self) -> None:
        """Lazy-load the Qwen model once per process."""

        if self._model_wrapper is not None:
            return

        with self._load_lock:
            if self._model_wrapper is not None:
                return

            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                load_kwargs: Dict[str, Any] = {
                    "device_map": self.config.device_map,
                    "dtype": self._resolve_torch_dtype(torch, self.config.torch_dtype),
                }
                if self.config.attn_implementation:
                    load_kwargs["attn_implementation"] = self.config.attn_implementation

                LOGGER.info("Loading model from %s", self.config.model_path)
                model_wrapper = Qwen3TTSModel.from_pretrained(
                    self.config.model_path,
                    **load_kwargs,
                )

                if self.config.enable_streaming_optimizations:
                    # Streaming optimizations are optional, but should be enabled before serving
                    # requests so the decoder window matches the default stream configuration.
                    model_wrapper.enable_streaming_optimizations(
                        decode_window_frames=self.config.streaming.decode_window_frames,
                        use_compile=self.config.use_compile,
                        use_cuda_graphs=self.config.use_cuda_graphs,
                        compile_mode=self.config.compile_mode,
                        use_fast_codebook=self.config.use_fast_codebook,
                        compile_codebook_predictor=self.config.compile_codebook_predictor,
                    )

                self._model_wrapper = model_wrapper
                self._last_error = None
            except Exception as exc:  # pragma: no cover - depends on local runtime/model files.
                self.record_error(exc)
                raise

    def warmup(self) -> None:
        """Run an optional warmup request to pay initialization cost before live traffic."""

        if not self.config.warmup_enabled:
            return

        if not self.config.warmup_reference_audio_path or not self.config.warmup_reference_text:
            raise ValueError(
                "Warmup requires both QWEN_TTS_WARMUP_REFERENCE_AUDIO_PATH and "
                "QWEN_TTS_WARMUP_REFERENCE_TEXT."
            )

        self.ensure_loaded()
        with open(self.config.warmup_reference_audio_path, "rb") as audio_file:
            prompt = self.create_clone_prompt(
                audio_bytes=audio_file.read(),
                reference_text=self.config.warmup_reference_text,
                x_vector_only_mode=False,
            )

        request = TTSStreamRequest(
            text=self.config.warmup_text,
            language="Auto",
            prompt_id=prompt.prompt_id,
            response_audio_transport=self.config.default_audio_transport,
        )
        for _ in self.stream_tts(request):
            pass

    def preload_default_prompts(self) -> None:
        """Prebuild default prompts so first live requests avoid prompt extraction overhead."""

        self.ensure_loaded()
        default_specs = self._default_prompt_specs()
        if not default_specs:
            return

        for key, spec in default_specs.items():
            if key in self._default_prompt_ids:
                continue
            with open(spec["audio_path"], "rb") as audio_file:
                record = self.create_clone_prompt(
                    audio_bytes=audio_file.read(),
                    reference_text=spec["reference_text"],
                    x_vector_only_mode=False,
                )
            with self._state_lock:
                self._default_prompt_ids[key] = record.prompt_id

    def create_clone_prompt(
        self,
        audio_bytes: bytes,
        reference_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> PromptRecord:
        """Extract a reusable prompt from uploaded audio bytes."""

        self.ensure_loaded()
        waveform, sample_rate = self._decode_audio_bytes(audio_bytes)
        prompt_items = self._build_prompt_items(
            waveform=waveform,
            sample_rate=sample_rate,
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        prompt_id = str(uuid.uuid4())
        record = PromptRecord(
            prompt_id=prompt_id,
            prompt_items=prompt_items,
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        self._store_prompt(record)
        return record

    def stream_tts(
        self,
        request: TTSStreamRequest,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[PCMChunk, None, None]:
        """Stream incremental PCM using the repository's true streaming generator.

        Routes to one of two backends based on the loaded model type:
          - CustomVoice model: calls stream_generate_custom_voice() with a speaker name.
          - Base model:        calls stream_generate_voice_clone() with a voice prompt.

        Neither path fakes streaming by generating a full WAV first.
        """

        self.ensure_loaded()
        stream_kwargs = self._merge_streaming_options(request.streaming)
        model_wrapper = self._require_model_wrapper()

        is_custom_voice = getattr(model_wrapper.model, "tts_model_type", None) == "custom_voice"

        if is_custom_voice:
            if not request.speaker:
                raise ValueError(
                    "The loaded model is a CustomVoice model. "
                    "Provide 'speaker' in your request (e.g. 'Vivian')."
                )
            generator = model_wrapper.stream_generate_custom_voice(
                text=request.text,
                speaker=request.speaker,
                language=request.language,
                instruct=request.instruct,
                emit_every_frames=stream_kwargs.emit_every_frames,
                decode_window_frames=stream_kwargs.decode_window_frames,
                overlap_samples=stream_kwargs.overlap_samples,
                max_frames=stream_kwargs.max_frames,
                use_optimized_decode=stream_kwargs.use_optimized_decode,
                first_chunk_emit_every=stream_kwargs.first_chunk_emit_every,
                first_chunk_decode_window=stream_kwargs.first_chunk_decode_window,
                first_chunk_frames=stream_kwargs.first_chunk_frames,
                repetition_penalty=stream_kwargs.repetition_penalty,
                repetition_penalty_window=stream_kwargs.repetition_penalty_window,
            )
        else:
            prompt_items = self._resolve_prompt_items(request)
            generator = model_wrapper.stream_generate_voice_clone(
                text=request.text,
                language=request.language,
                voice_clone_prompt=prompt_items,
                emit_every_frames=stream_kwargs.emit_every_frames,
                decode_window_frames=stream_kwargs.decode_window_frames,
                overlap_samples=stream_kwargs.overlap_samples,
                max_frames=stream_kwargs.max_frames,
                use_optimized_decode=stream_kwargs.use_optimized_decode,
                first_chunk_emit_every=stream_kwargs.first_chunk_emit_every,
                first_chunk_decode_window=stream_kwargs.first_chunk_decode_window,
                first_chunk_frames=stream_kwargs.first_chunk_frames,
                repetition_penalty=stream_kwargs.repetition_penalty,
                repetition_penalty_window=stream_kwargs.repetition_penalty_window,
            )

        with self._generation_lock:
            with self._track_active_stream():
                try:
                    for sequence, (chunk, sample_rate) in enumerate(generator):
                        if stop_event and stop_event.is_set():
                            break
                        yield PCMChunk(
                            sequence=sequence,
                            sample_rate=sample_rate,
                            samples=np.asarray(chunk, dtype=np.float32),
                        )
                finally:
                    close_fn = getattr(generator, "close", None)
                    if callable(close_fn):
                        close_fn()

    def health(self) -> RuntimeHealth:
        """Expose runtime state without forcing a load attempt."""

        tts_model_type: Optional[str] = None
        if self._model_wrapper is not None:
            tts_model_type = getattr(self._model_wrapper.model, "tts_model_type", None)

        with self._state_lock:
            return RuntimeHealth(
                model_loaded=self._model_wrapper is not None,
                active_streams=self._active_streams,
                prompt_cache_size=len(self._prompt_cache),
                model_path=self.config.model_path,
                tts_model_type=tts_model_type,
                last_error=self._last_error,
                default_prompt_keys=sorted(self._default_prompt_ids.keys()),
                config=self.config.summary(),
            )

    def record_error(self, exc: Exception) -> None:
        """Persist the latest error so `/health` can explain startup failures."""

        message = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Runtime error: %s", message)
        with self._state_lock:
            self._last_error = message

    def _resolve_prompt_items(self, request: TTSStreamRequest) -> List[Any]:
        if request.prompt_id:
            with self._state_lock:
                record = self._prompt_cache.get(request.prompt_id)
            if record is None:
                raise KeyError(f"Unknown prompt_id: {request.prompt_id}")
            return record.prompt_items

        default_prompt = self._resolve_default_prompt(request.language)
        if default_prompt is not None:
            return default_prompt.prompt_items

        if not request.reference_audio_base64:
            raise ValueError(
                "No prompt source was provided and no default prompt matches the requested language."
            )

        audio_bytes = self._decode_base64_audio(request.reference_audio_base64)
        waveform, sample_rate = self._decode_audio_bytes(audio_bytes)
        return self._build_prompt_items(
            waveform=waveform,
            sample_rate=sample_rate,
            reference_text=request.reference_text,
            x_vector_only_mode=request.x_vector_only_mode,
        )

    def _store_prompt(self, record: PromptRecord) -> None:
        with self._state_lock:
            self._prompt_cache[record.prompt_id] = record
            self._prompt_cache.move_to_end(record.prompt_id)
            while len(self._prompt_cache) > self.config.prompt_cache_max_entries:
                self._prompt_cache.popitem(last=False)

    def _resolve_default_prompt(self, language: str) -> Optional[PromptRecord]:
        key = self._match_default_prompt_key(language)
        if key is None:
            return None

        with self._state_lock:
            prompt_id = self._default_prompt_ids.get(key)
            if prompt_id is None:
                return None
            return self._prompt_cache.get(prompt_id)

    def _merge_streaming_options(self, options: Any) -> StreamingDefaults:
        defaults = self.config.streaming
        if hasattr(options, "model_dump"):
            payload = options.model_dump(exclude_none=True)
        else:
            payload = options.dict(exclude_none=True)
        return StreamingDefaults(
            emit_every_frames=payload.get("emit_every_frames", defaults.emit_every_frames),
            decode_window_frames=payload.get("decode_window_frames", defaults.decode_window_frames),
            overlap_samples=payload.get("overlap_samples", defaults.overlap_samples),
            max_frames=payload.get("max_frames", defaults.max_frames),
            first_chunk_emit_every=payload.get(
                "first_chunk_emit_every",
                defaults.first_chunk_emit_every,
            ),
            first_chunk_decode_window=payload.get(
                "first_chunk_decode_window",
                defaults.first_chunk_decode_window,
            ),
            first_chunk_frames=payload.get("first_chunk_frames", defaults.first_chunk_frames),
            repetition_penalty=payload.get(
                "repetition_penalty",
                defaults.repetition_penalty,
            ),
            repetition_penalty_window=payload.get(
                "repetition_penalty_window",
                defaults.repetition_penalty_window,
            ),
            use_optimized_decode=payload.get(
                "use_optimized_decode",
                defaults.use_optimized_decode,
            ),
        )

    def _decode_audio_bytes(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        with io.BytesIO(audio_bytes) as buffer:
            waveform, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        return waveform, int(sample_rate)

    def _decode_base64_audio(self, payload: str) -> bytes:
        normalized = payload
        if payload.startswith("data:audio") and "," in payload:
            normalized = payload.split(",", 1)[1]
        return base64.b64decode(normalized)

    def _default_prompt_specs(self) -> Dict[str, Dict[str, str]]:
        specs: Dict[str, Dict[str, str]] = {}
        if (
            self.config.default_auto_reference_audio_path
            and self.config.default_auto_reference_text
        ):
            specs["auto"] = {
                "audio_path": self.config.default_auto_reference_audio_path,
                "reference_text": self.config.default_auto_reference_text,
            }
        if (
            self.config.default_chinese_reference_audio_path
            and self.config.default_chinese_reference_text
        ):
            specs["chinese"] = {
                "audio_path": self.config.default_chinese_reference_audio_path,
                "reference_text": self.config.default_chinese_reference_text,
            }
        if (
            self.config.default_english_reference_audio_path
            and self.config.default_english_reference_text
        ):
            specs["english"] = {
                "audio_path": self.config.default_english_reference_audio_path,
                "reference_text": self.config.default_english_reference_text,
            }
        return specs

    def _match_default_prompt_key(self, language: str) -> Optional[str]:
        normalized = (language or "").strip().lower()
        available = self._default_prompt_specs()
        if not available:
            return None

        chinese_aliases = {"chinese", "zh", "zh-cn", "zh_cn", "mandarin"}
        english_aliases = {"english", "en", "en-us", "en-gb", "en_us", "en_gb"}
        auto_aliases = {"", "auto", "automatic"}

        if normalized in chinese_aliases and "chinese" in available:
            return "chinese"
        if normalized in english_aliases and "english" in available:
            return "english"
        if normalized in auto_aliases and "auto" in available:
            return "auto"
        if normalized in auto_aliases and "english" in available and "chinese" not in available:
            return "english"
        if normalized in auto_aliases and "chinese" in available and "english" not in available:
            return "chinese"
        if "auto" in available:
            return "auto"
        return None

    def _build_prompt_items(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        reference_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> List[Any]:
        # Prompt creation is still serialized with generation because the current MVP keeps a
        # single loaded model instance in-process. To add multi-instance routing or a queue-based
        # scheduler later, change this lock boundary first.
        with self._generation_lock:
            model_wrapper = self._require_model_wrapper()
            return model_wrapper.create_voice_clone_prompt(
                ref_audio=(waveform, sample_rate),
                ref_text=reference_text,
                x_vector_only_mode=x_vector_only_mode,
            )

    def _require_model_wrapper(self) -> Any:
        if self._model_wrapper is None:
            raise RuntimeError("Model is not loaded.")
        return self._model_wrapper

    def _resolve_torch_dtype(self, torch_module: Any, dtype_name: str) -> Any:
        normalized = dtype_name.strip().lower()
        mapping = {
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return mapping[normalized]

    def _track_active_stream(self) -> "_ActiveStreamContext":
        return _ActiveStreamContext(self)


class _ActiveStreamContext:
    """Small helper that keeps `active_streams` correct across generator lifecycles."""

    def __init__(self, runtime: QwenStreamingRuntime):
        self.runtime = runtime

    def __enter__(self) -> None:
        with self.runtime._state_lock:
            self.runtime._active_streams += 1
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        with self.runtime._state_lock:
            self.runtime._active_streams -= 1
        return None
