
docker run -it -v /data/xiedong/Qwen/Qwen3-TTS-12Hz-1.7B-Base:/Qwen3-TTS-12Hz-1.7B-Base -v /data/xiedong/Qwen3-TTS-streaming:/data/xiedong/Qwen3-TTS-streaming --net host --gpus all modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.2 bash

2026-03-29
- Added `pipecat_apps/qwen3_tts_streaming_service` as a standalone FastAPI service skeleton for true streaming TTS over WebSocket.
- The service wraps the repository's Base voice-clone streaming path instead of offline WAV generation.
- Prompt caching is in-memory for the MVP and is intentionally isolated in `runtime.py` so it can later be replaced by Redis or a database-backed store.
- Added `examples/benchmark_streaming_examples.py` to benchmark Chinese and English WebSocket streaming requests, persist returned audio as WAV files, and record first-audio latency in JSON.
- The service can now preload default prompts for `Auto`, `Chinese`, and `English`, so simple TTS requests do not need to send a new reference audio or `prompt_id` every time.

