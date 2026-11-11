# The product in ONE image, with model 2 (ASTT improved) inside: anyone with Docker runs it with
# one command, no code, data or DVC needed. Web page at /, API at /docs, same port.
#   docker run --gpus all -p 7860:7860 ghcr.io/<owner>/dsp391m-app        (NVIDIA GPU: fast)
#   docker run -p 7860:7860 ghcr.io/<owner>/dsp391m-app                   (no GPU: works, much slower)
# Built and published by docker/publish.sh on the computer that has the trained model (outputs/).
# BASE=python:3.11-slim TORCH=cpu builds a smaller CPU-only variant (used by CI).
ARG BASE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${BASE}
ARG TORCH=gpu

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git ca-certificates ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt requirements-ui.txt ./
RUN if [ "$TORCH" = cpu ]; then \
        pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu; fi \
    && pip install --no-cache-dir -r requirements-api.txt -r requirements-ui.txt

# People tracking (Keypoint R-CNN, COCO weights) and the thesis code model 2 builds on.
COPY src/models/nets.py src/models/nets.py
RUN touch src/__init__.py src/models/__init__.py && python -m src.models.nets download keypointrcnn
COPY third_party/ third_party/
RUN python third_party/fetch.py

COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY app/ app/
# Model 2: trained weights, its parameter count, its test metrics (threshold = EER point).
COPY outputs/training/astt_improved/model.pth outputs/training/astt_improved/run.json outputs/training/astt_improved/
COPY outputs/evaluation/astt_improved/metrics.json outputs/evaluation/astt_improved/
RUN chmod -R a+rwX /app                       # also runs as a non-root user (e.g. Hugging Face Spaces)

ENV VAD_MODEL=astt_improved VAD_MODELS=astt_improved VAD_MAX_SECONDS=60 PORT=7860 \
    HOME=/tmp MPLCONFIGDIR=/tmp GRADIO_ANALYTICS_ENABLED=False PYTHONUNBUFFERED=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD curl -fs http://localhost:7860/health || exit 1
CMD ["python", "-m", "app.server"]
