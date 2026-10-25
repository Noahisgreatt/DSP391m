# API image: FastAPI + our models. Trained weights are NOT baked in; they come from
# outputs/ (dvc pull), mounted by docker-compose.yml.
#   docker compose up --build            -> API :8000, Prometheus :9090, Grafana :3000
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Pretrained weights for people tracking and model 4 (the deployable one), downloaded once.
# Model 3 (AI-VAD, 331M) is the heavy research baseline and is not served.
COPY src/models/nets.py src/models/nets.py
RUN touch src/__init__.py src/models/__init__.py && python -m src.models.nets download keypointrcnn flownet2

# The papers' code is fetched at its pinned commits and checked (sha256), so a fresh clone builds.
COPY third_party/ third_party/
RUN python third_party/fetch.py

COPY configs/ configs/
COPY src/ src/
COPY api/ api/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
