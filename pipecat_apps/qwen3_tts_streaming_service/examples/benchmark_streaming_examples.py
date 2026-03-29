"""Benchmark Chinese and English streaming TTS requests against the local service."""

from __future__ import annotations

import asyncio
import json
import os
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets


@dataclass(slots=True)
class ExampleCase:
    """One synthesis example sent to the streaming API."""

    case_id: str
    language: str
    text: str
    output_filename: str


@dataclass(slots=True)
class StreamBenchmarkResult:
    """Measured latency and output summary for one streaming request."""

    case_id: str
    language: str
    output_wav_path: str
    request_text: str
    sample_rate: int
    channels: int
    encoding: str
    transport: str
    stream_start_latency_ms: Optional[float]
    first_chunk_meta_latency_ms: Optional[float]
    first_audio_byte_latency_ms: Optional[float]
    total_latency_ms: float
    chunk_count: int
    audio_bytes: int
    audio_duration_ms: float


EXAMPLE_CASES: List[ExampleCase] = [
    ExampleCase(
        case_id="zh_cn_chat",
        language="Chinese",
        text="你好，这是一个中文流式语音测试。我们正在验证首包延迟、连续音频分块，以及最终音频是否适合实时对话场景。Hello, this is an English streaming TTS test. ",
        output_filename="zh_cn_streaming_example.wav",
    ),
    ExampleCase(
        case_id="en_chat",
        language="English",
        text="Hello, this is an English streaming TTS test. We are measuring first audio latency and checking whether the chunks arrive continuously for real time voice chat.",
        output_filename="en_streaming_example.wav",
    ),
]

WARMUP_CASE = ExampleCase(
    case_id="warmup",
    language="Auto",
    text="你好，这是一个流式语音预热请求，用于触发默认 prompt 和首次编译缓存。",
    output_filename="",
)


def save_pcm_s16le_wav(
    pcm_bytes: bytes,
    output_path: Path,
    sample_rate: int,
    channels: int,
) -> None:
    """Save raw PCM bytes to a standard WAV file for listening."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


async def stream_case(
    ws_url: str,
    case: ExampleCase,
    output_dir: Path,
    save_output: bool = True,
    stop_after_first_audio_byte: bool = False,
    speaker: Optional[str] = None,
    instruct: Optional[str] = None,
) -> StreamBenchmarkResult:
    """Run one streaming synthesis request and persist the received audio."""

    request_payload: Dict[str, Any] = {
        "text": case.text,
        "language": case.language,
        "response_audio_transport": "binary",
        "streaming": {
            "emit_every_frames": 8,
            "decode_window_frames": 80,
            "first_chunk_emit_every": 5,
            "first_chunk_decode_window": 48,
            "first_chunk_frames": 48,
            "overlap_samples": 512,
            # 1.05 prevents token-loop runaway generation when compile=false.
            # Keep at 1.0 only when using compile=true with CUDA graph constraints.
            "repetition_penalty": 1.05,
            "max_frames": 400,
        },
    }
    # CustomVoice path: add speaker (and optional instruct) to the request.
    if speaker:
        request_payload["speaker"] = speaker
    if instruct:
        request_payload["instruct"] = instruct

    audio_buffer = bytearray()
    chunk_count = 0
    sample_rate = 24_000
    channels = 1
    encoding = "pcm_s16le"
    transport = "binary"
    stream_start_latency_ms: Optional[float] = None
    first_chunk_meta_latency_ms: Optional[float] = None
    first_audio_byte_latency_ms: Optional[float] = None
    started_at = time.perf_counter()

    async with websockets.connect(ws_url, max_size=None) as websocket:
        await websocket.send(json.dumps(request_payload))

        while True:
            message = await websocket.recv()
            now = time.perf_counter()
            if isinstance(message, bytes):
                if first_audio_byte_latency_ms is None:
                    first_audio_byte_latency_ms = (now - started_at) * 1000.0
                    if stop_after_first_audio_byte:
                        await websocket.close()
                        return StreamBenchmarkResult(
                            case_id=case.case_id,
                            language=case.language,
                            output_wav_path="",
                            request_text=case.text,
                            sample_rate=sample_rate,
                            channels=channels,
                            encoding=encoding,
                            transport=transport,
                            stream_start_latency_ms=stream_start_latency_ms,
                            first_chunk_meta_latency_ms=first_chunk_meta_latency_ms,
                            first_audio_byte_latency_ms=first_audio_byte_latency_ms,
                            total_latency_ms=first_audio_byte_latency_ms,
                            chunk_count=chunk_count,
                            audio_bytes=len(message),
                            audio_duration_ms=(
                                (len(message) / (2 * max(channels, 1) * sample_rate)) * 1000.0
                            ),
                        )
                audio_buffer.extend(message)
                continue

            payload = json.loads(message)
            message_type = payload.get("type")

            if message_type == "stream_start":
                stream_start_latency_ms = (now - started_at) * 1000.0
                audio_format = payload.get("audio_format", {})
                sample_rate = int(audio_format.get("sample_rate", sample_rate))
                channels = int(audio_format.get("channels", channels))
                encoding = str(audio_format.get("encoding", encoding))
                transport = str(audio_format.get("transport", transport))
                continue

            if message_type == "chunk":
                chunk_count += 1
                if first_chunk_meta_latency_ms is None:
                    first_chunk_meta_latency_ms = (now - started_at) * 1000.0
                continue

            if message_type == "stream_end":
                total_latency_ms = (now - started_at) * 1000.0
                output_path = output_dir / case.output_filename if save_output else None
                if output_path is not None:
                    save_pcm_s16le_wav(
                        pcm_bytes=bytes(audio_buffer),
                        output_path=output_path,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                audio_duration_ms = (
                    (len(audio_buffer) / (2 * max(channels, 1) * sample_rate)) * 1000.0
                )
                return StreamBenchmarkResult(
                    case_id=case.case_id,
                    language=case.language,
                    output_wav_path=str(output_path) if output_path is not None else "",
                    request_text=case.text,
                    sample_rate=sample_rate,
                    channels=channels,
                    encoding=encoding,
                    transport=transport,
                    stream_start_latency_ms=stream_start_latency_ms,
                    first_chunk_meta_latency_ms=first_chunk_meta_latency_ms,
                    first_audio_byte_latency_ms=first_audio_byte_latency_ms,
                    total_latency_ms=total_latency_ms,
                    chunk_count=chunk_count,
                    audio_bytes=len(audio_buffer),
                    audio_duration_ms=audio_duration_ms,
                )

            if message_type == "error":
                raise RuntimeError(payload.get("message", "Unknown streaming error."))

    raise RuntimeError("WebSocket stream ended unexpectedly.")


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "pipecat_apps" / "qwen3_tts_streaming_service" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ws_url = "ws://127.0.0.1:8000/v1/tts/stream"
    warmup_rounds = max(int(os.getenv("QWEN_TTS_BENCHMARK_WARMUP_ROUNDS", "1")), 0)
    ttft_only = os.getenv("QWEN_TTS_BENCHMARK_TTFT_ONLY", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # CustomVoice: set QWEN_TTS_BENCHMARK_SPEAKER to use the CustomVoice path.
    # Leave unset (or empty) to use the default voice-clone path.
    speaker = os.getenv("QWEN_TTS_BENCHMARK_SPEAKER", "").strip() or None
    instruct = os.getenv("QWEN_TTS_BENCHMARK_INSTRUCT", "").strip() or None

    if speaker:
        print(f"CustomVoice mode: speaker={speaker}, instruct={instruct!r}")
    else:
        print("Using preloaded service default prompt (voice-clone path).")
    print(f"TTFT-only mode: {ttft_only}")
    if not ttft_only:
        print("Full-stream mode enabled. The script will wait for stream_end and save WAV files.")
    if warmup_rounds:
        print(f"Running {warmup_rounds} warmup request(s) before measuring latency.")
        for warmup_index in range(warmup_rounds):
            warmup_result = await stream_case(
                ws_url=ws_url,
                case=WARMUP_CASE,
                output_dir=output_dir,
                save_output=False,
                stop_after_first_audio_byte=ttft_only,
                speaker=speaker,
                instruct=instruct,
            )
            print(
                f"warmup_{warmup_index + 1}: "
                f"first_audio_byte={warmup_result.first_audio_byte_latency_ms:.2f} ms, "
                f"total={warmup_result.total_latency_ms:.2f} ms, "
                f"chunks={warmup_result.chunk_count}"
            )

    results: List[StreamBenchmarkResult] = []
    for case in EXAMPLE_CASES:
        result = await stream_case(
            ws_url=ws_url,
            case=case,
            output_dir=output_dir,
            save_output=not ttft_only,
            stop_after_first_audio_byte=ttft_only,
            speaker=speaker,
            instruct=instruct,
        )
        results.append(result)
        print(
            f"{case.case_id}: first_audio_byte={result.first_audio_byte_latency_ms:.2f} ms, "
            f"total={result.total_latency_ms:.2f} ms, chunks={result.chunk_count}, "
            f"wav={result.output_wav_path}"
        )

    report_path = output_dir / "streaming_benchmark_results.json"
    report = {
        "websocket_url": ws_url,
        "used_service_default_prompt": True,
        "warmup_rounds": warmup_rounds,
        "ttft_only": ttft_only,
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


# QWEN_TTS_BENCHMARK_SPEAKER=Serena QWEN_TTS_BENCHMARK_WARMUP_ROUNDS=0 python pipecat_apps/qwen3_tts_streaming_service/examples/benchmark_streaming_examples.py
if __name__ == "__main__":
    asyncio.run(main())