FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.2

ENV PYTHONUNBUFFERED=1 \
    QWEN_TTS_SERVICE_HOST=0.0.0.0 \
    QWEN_TTS_SERVICE_PORT=8000 \
    QWEN_TTS_MODEL_PATH=/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN_TTS_DEVICE_MAP=cuda:0 \
    QWEN_TTS_TORCH_DTYPE=bfloat16 \
    QWEN_TTS_LOAD_ON_STARTUP=true \
    QWEN_TTS_ENABLE_STREAMING_OPTIMIZATIONS=true \
    QWEN_TTS_USE_COMPILE=true \
    QWEN_TTS_USE_CUDA_GRAPHS=false

RUN apt update && apt install -y sox libsox-fmt-all && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install -e . && \
    pip install transformers==4.57.3 && \
    pip install -r pipecat_apps/qwen3_tts_streaming_service/requirements.txt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "pipecat_apps.qwen3_tts_streaming_service.app:app", "--host", "0.0.0.0", "--port", "8000"]