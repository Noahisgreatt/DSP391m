# MLOps: DVC, FastAPI, Docker, Prometheus, Grafana

## DVC
- **Remote**: Google Drive (`gdrive_remote` in `.dvc/config`). The OAuth client id/secret live
  only in `.dvc/config.local` (never committed). With 400 GB of Drive, everything can be pushed.
  To use another Drive folder: `dvc remote modify gdrive_remote url gdrive://<folder id>`.
- **Versioned with DVC** (pushed): `data/raw/shanghaitech` (train.zip + test.zip, ~20 GB),
  `data/raw/manual`, every `outputs/training/<model>/model.pth`.
  ShanghaiTech is kept as 2 zips because 180,000 small JPEGs are very slow on Google Drive.
- **Re-created, not pushed** (`cache: false`): unpacked frames, features, checkpoints, scores.
- **Committed to git** (small): data cards, split, metrics / plots JSON and CSV, release cards.
- **Pipeline**: `dvc.yaml`, 4 groups of stages: data (`src/data.py`), behavior (`src/behavior.py`),
  models (`src/train.py`), evaluation + comparison (`src/evaluate.py`);
  `dvc dag` draws it, `dvc repro <stage>` runs a stage and everything it needs.
- **Parameters**: every `configs/*.yaml` value a stage uses is declared under `params:`, so
  `dvc status` knows when a setting change makes a result out of date.
- **Metrics / plots**: `dvc metrics show`, `dvc plots show` (training loss, ROC of every model).
- **Model registry**: `artifacts:` in `dvc.yaml` describes the 4 models (path, type, labels).

## FastAPI (`api/main.py`, uses `src/demo.py`)
`uvicorn api.main:app --port 8000`, then http://localhost:8000/docs
| Endpoint | |
|---|---|
| `GET /health` | status + which model, device, threshold |
| `POST /analyze` | upload a video -> per-frame scores, abnormal frames, unusual objects (models 3/4), LOITERING / COUNTER_FLOW / TOO_FAST events, link to the result video |
| `GET /results/{job}` / `.../video` | the JSON / the annotated video again |
| `GET /metrics` | Prometheus metrics |
Served model: `configs/api.yaml` or `VAD_MODEL=astt_baseline|astt_improved|aivad_lite|none`.
If that model isn't trained + evaluated yet, the API still starts with the behavior rules only and
`/health` says why (`warning`); after training: `docker compose restart api`.

## Docker
`docker compose up --build -d` starts 3 containers: `api` (FastAPI, model weights mounted from
`outputs/`), `prometheus`, `grafana`. GPU: add `-f docker-compose.gpu.yml`.

## Monitoring (Prometheus + Grafana)
Grafana http://localhost:3000 (admin/admin), dashboard **DSP391m - Video anomaly detection**:
- **Serving**: requests/s, latency p50/p95, error rate, time per pipeline step (frames, tracking,
  model, render), frames/s, behavior events per type, unusual objects per attribute (models 3/4),
  **frame-score distribution vs. the threshold and share of abnormal frames: if these drift
  upwards, incoming videos no longer look like the training data (data drift)**.
- **Training (live)**: `src/train.py` serves its metrics on port 8001 while it runs: loss, learning
  rate, it/s, progress, time left, epoch, validation AUC, GPU memory.
Prometheus config: `monitoring/prometheus.yml`; dashboard JSON: `monitoring/grafana/dashboards/`.
