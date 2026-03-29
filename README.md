# Qwen3-TTS Streaming

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的实时流式语音合成服务。

## 功能特性

来自 [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) 的基础能力：

- `stream_generate_voice_clone()` — 带音色克隆的流式生成
- `stream_generate_pcm()` — 底层实时 PCM 流式生成
- `torch.compile` + CUDA Graph 推理优化

本 fork 新增：

- **两阶段流式（Two-phase streaming）** — 首包更快，后续音质稳定
- **多 EOS token 检测** — 更广泛的终止条件覆盖，修复流式生成中的加速音频和 runaway 问题
- **Hann 窗交叉淡化** — 消除分块边界处的爆音和咔哒声
- **流式 repetition penalty** — 防止 token 循环导致的音频重复和 runaway 生成
- **`stream_generate_custom_voice()`** — CustomVoice 模型的真流式路径（底层复用 `stream_generate_pcm`，非先生成再切块的假流式）
- **FastAPI WebSocket 服务** — 独立部署，按模型类型自动路由生成路径

---

## 支持的模型

| 模型 | 类型 | 流式支持 | 说话人控制 |
|------|------|---------|----------|
| `Qwen3-TTS-12Hz-1.7B-Base` | Base | ✅ 真流式 | 参考音频克隆 |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | ✅ 真流式 | 固定说话人（Vivian / Serena 等） |

---

## Docker 部署

### 1. 构建镜像

```bash
cd /data/xiedong/Qwen3-TTS-streaming
docker build -t qwen3-tts-streaming-service:local .
```

### 2. 启动服务

**方案 A：使用 CustomVoice 模型（固定说话人，无需参考音频）**

```bash
docker run -d --gpus all --network host \
  --name qwen3-tts-custom \
  -v /data/xiedong/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice:/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  -e QWEN_TTS_MODEL_PATH=/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  -e QWEN_TTS_USE_COMPILE=false \
  -e QWEN_TTS_WARMUP_ENABLED=false \
  qwen3-tts-streaming-service:local
```

**方案 B：使用 Base 模型（音色克隆，需提供参考音频）**

```bash
docker run -d --gpus all --network host \
  --name qwen3-tts-base \
  -v /data/xiedong/Qwen/Qwen3-TTS-12Hz-1.7B-Base:/Qwen3-TTS-12Hz-1.7B-Base \
  -e QWEN_TTS_MODEL_PATH=/Qwen3-TTS-12Hz-1.7B-Base \
  -e QWEN_TTS_USE_COMPILE=false \
  -e QWEN_TTS_DEFAULT_AUTO_REFERENCE_AUDIO_PATH=/ref/ref.wav \
  -e QWEN_TTS_DEFAULT_AUTO_REFERENCE_TEXT="参考音频对应的文字" \
  -v /path/to/ref/audio:/ref \
  qwen3-tts-streaming-service:local
```

> `QWEN_TTS_USE_COMPILE=false`：禁用 `torch.compile`，避免多线程 CUDA graph 冲突。如需更低延迟可开启，但需确保推理在同一线程。

### 3. 验证服务

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "ok": true,
  "model_loaded": true,
  "tts_model_type": "custom_voice",
  "model_path": "/Qwen3-TTS-12Hz-0.6B-CustomVoice"
}
```

---

## Benchmark 测试脚本

脚本路径：`pipecat_apps/qwen3_tts_streaming_service/examples/benchmark_streaming_examples.py`

该脚本通过 WebSocket 向服务发起流式请求，测量首包延迟（TTFT）并保存完整 WAV 文件。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_TTS_BENCHMARK_SPEAKER` | （空）| CustomVoice 说话人名称，为空时走 VoiceClone 路径 |
| `QWEN_TTS_BENCHMARK_INSTRUCT` | （空）| 说话风格指令（仅 1.5B+ 模型支持，0.6B 自动忽略）|
| `QWEN_TTS_BENCHMARK_WARMUP_ROUNDS` | `1` | 正式测试前的预热请求次数 |
| `QWEN_TTS_BENCHMARK_TTFT_ONLY` | `false` | 设为 `true` 时只测首包延迟，不保存 WAV |

### 运行示例

```bash
# CustomVoice — Vivian 说话人，保存 WAV
QWEN_TTS_BENCHMARK_SPEAKER=Vivian \
python pipecat_apps/qwen3_tts_streaming_service/examples/benchmark_streaming_examples.py

# CustomVoice — Serena 说话人，跳过预热
QWEN_TTS_BENCHMARK_SPEAKER=Serena \
QWEN_TTS_BENCHMARK_WARMUP_ROUNDS=0 \
python pipecat_apps/qwen3_tts_streaming_service/examples/benchmark_streaming_examples.py

# Base 模型（需服务端配置好默认参考音频），仅测 TTFT
QWEN_TTS_BENCHMARK_TTFT_ONLY=true \
python pipecat_apps/qwen3_tts_streaming_service/examples/benchmark_streaming_examples.py
```

### 输出文件

```
pipecat_apps/qwen3_tts_streaming_service/output/
  zh_cn_streaming_example.wav
  en_streaming_example.wav
  streaming_benchmark_results.json
```

### 实测延迟参考（CustomVoice 0.6B，RTX A100，无预热）

| 说话人 | 语言 | 首包延迟 (TTFT) | 总耗时 |
|--------|------|---------------|--------|
| Vivian | 中文 | 774 ms | 16.6 s |
| Vivian | 英文 | 579 ms | 17.4 s |
| Serena | 中文 | 448 ms | 13.7 s |
| Serena | 英文 | 392 ms | 15.2 s |

---

## 流式参数说明

WebSocket 请求中 `streaming` 字段的所有参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `emit_every_frames` | `8` | 每生成 N 个 codec 帧向客户端发送一块 PCM，越小后续块越密集 |
| `decode_window_frames` | `80` | 解码器上下文窗口大小，越大音质越好但延迟越高（稳态值）|
| `first_chunk_emit_every` | `5` | 首块阶段的 emit 间隔，比稳态更激进以降低 TTFT |
| `first_chunk_decode_window` | `48` | 首块阶段的解码窗口，比稳态更小以加快首块发送 |
| `first_chunk_frames` | `48` | 生成前 N 帧使用首块参数，之后切换为稳态参数 |
| `overlap_samples` | `512` | 相邻两块 PCM 的交叉淡化长度（样本数），≈21ms@24kHz，防止边界爆音 |
| `repetition_penalty` | `1.05` | 重复 token 惩罚系数，`1.0` 为不惩罚。不开 compile 时必须 `>1.0`，否则易陷入 token 循环死循环 |
| `max_frames` | `400` | 最大 codec 帧数上限（400帧≈33秒），防止 runaway 时无限生成 |

**两阶段流式示意：**

```
前 48 帧（首块阶段）          48 帧之后（稳态阶段）
─────────────────────────    ────────────────────────
emit_every = 5（更快发送）    emit_every = 8（正常节奏）
decode_window = 48           decode_window = 80
→ TTFT 更低                  → 音质更稳定
```

---

## WebSocket 请求格式

### CustomVoice 路径（固定说话人）

```json
{
  "text": "外面的天气有些阴沉，记得带伞，保持干爽会让心情更愉快。",
  "language": "Auto",
  "speaker": "Vivian",
  "response_audio_transport": "binary",
  "streaming": {
    "emit_every_frames": 8,
    "decode_window_frames": 80,
    "first_chunk_emit_every": 5,
    "first_chunk_decode_window": 48,
    "first_chunk_frames": 48,
    "overlap_samples": 512,
    "repetition_penalty": 1.05,
    "max_frames": 400
  }
}
```

### VoiceClone 路径（参考音频克隆）

```json
{
  "text": "你好，这是一个流式语音合成测试。",
  "language": "Auto",
  "response_audio_transport": "binary",
  "streaming": {
    "emit_every_frames": 8,
    "decode_window_frames": 80,
    "first_chunk_emit_every": 5,
    "first_chunk_decode_window": 48,
    "first_chunk_frames": 48,
    "overlap_samples": 512,
    "repetition_penalty": 1.05,
    "max_frames": 400
  }
}
```

> `language` 建议设为 `"Auto"`，对中英混合文本更稳健（模型自动判断语言）。

### 音频输出格式

- 编码：`pcm_s16le`（有符号 16 位小端）
- 采样率：`24000 Hz`
- 声道：`1`（单声道）

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务健康状态，含模型加载状态、模型类型、配置摘要 |
| `POST` | `/v1/clone-prompts` | 上传参考音频，构建并缓存 prompt，返回 `prompt_id`（Base 模型专用）|
| `WebSocket` | `/v1/tts/stream` | 流式 TTS，返回 `stream_start` / `chunk` / `stream_end` / `error` 事件 |

---

## 服务环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_TTS_MODEL_PATH` | `/Qwen3-TTS-12Hz-1.7B-Base` | 模型路径 |
| `QWEN_TTS_DEVICE_MAP` | `cuda:0` | 推理设备 |
| `QWEN_TTS_TORCH_DTYPE` | `bfloat16` | 推理精度 |
| `QWEN_TTS_USE_COMPILE` | `true` | 是否开启 torch.compile（多线程场景建议 false）|
| `QWEN_TTS_USE_CUDA_GRAPHS` | `false` | 是否开启 CUDA Graph |
| `QWEN_TTS_LOAD_ON_STARTUP` | `true` | 服务启动时立即加载模型 |
| `QWEN_TTS_WARMUP_ENABLED` | `false` | 是否在启动后执行预热请求 |
| `QWEN_TTS_DEFAULT_AUTO_REFERENCE_AUDIO_PATH` | — | 默认参考音频路径（Base 模型） |
| `QWEN_TTS_DEFAULT_AUTO_REFERENCE_TEXT` | — | 默认参考音频对应文字（Base 模型） |

---

## 已知限制

- CustomVoice 0.6B 模型不支持 `instruct` 参数（1.5B+ 才支持），服务会自动忽略
- `torch.compile` 在多线程推理时可能触发 CUDA graph TLS 冲突，建议生产环境设 `QWEN_TTS_USE_COMPILE=false`
- 当前运行时为单进程单模型实例，不支持并发多路流（多请求串行处理）
- 扩展为多实例或队列调度的改动入口在 `pipecat_apps/qwen3_tts_streaming_service/runtime.py`

---

基于：
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
