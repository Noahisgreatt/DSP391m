# API image: FastAPI + the models + YOLO. Trained weights are NOT baked in; they come from
# outputs/ (dvc pull), mounted by docker-compose.yml.
#   docker compose up --build            -> API :8000, Prometheus :9090, Grafana :3000
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Pretrained weights downloaded once at build time: YOLO (people tracking), and for model 4
# YOLO11x-pose + RAFT. Model 3 (AI-VAD, ~333M, needs detectron2) is the heavy research baseline
# and is not served; model 4 is its deployable version.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); YOLO('yolo11x-pose.pt')" && \
    python -c "from torchvision.models.optical_flow import raft_large, Raft_Large_Weights; raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)"

COPY configs/ configs/
COPY third_party/ third_party/
COPY src/ src/
COPY api/ api/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
