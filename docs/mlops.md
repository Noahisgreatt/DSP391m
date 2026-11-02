# MLOps: DVC, FastAPI, web page, Docker, Prometheus, Grafana

```
 browser ──> web page (Gradio, :7860) ──HTTP──> FastAPI (:8000) ──> src/demo.py: tracking + rules + model
                                                     │ /metrics
                                         Prometheus (:9090) ──> Grafana (:3000)
```

## DVC
- **Remote**: Google Drive (`gdrive_remote` in `.dvc/config`). The OAuth client id/secret live
  only in `.dvc/config.local` (never committed). With 400 GB of Drive, everything can be pushed.
  To use another Drive folder: `dvc remote modify gdrive_remote url gdrive://<folder id>`.
- **Versioned with DVC** (pushed): `data/raw/shanghaitech` (train.zip + test.zip, ~20 GB),
  `data/raw/manual`, every `outputs/training/<model>/model.pth`.
  ShanghaiTech is kept as 2 zips because 180,000 small JPEGs are very slow on Google Drive.
- **Re-created, not pushed** (`cache: false`): unpacked frames, features, checkpoints, scores.
- **Committed to git** (small): data cards, split, metrics / plots JSON and CSV, version cards.
- **Versions** v1.0, v1.1, v2.0 ... with metadata: `python -m src.release` ([versioning.md](versioning.md)).
- **Pipeline**: `dvc.yaml`: data (`prepare_*`, `split`), `behavior@<dataset>`, `train@<model>`,
  `evaluate@<model>`, `compare`, `benchmark`;
  `dvc dag` draws it, `dvc repro <stage>` runs a stage and everything it needs.
- **Parameters**: every `configs/*.yaml` value a stage uses is declared under `params:`, so
  `dvc status` knows when a setting change makes a result out of date.
- **Metrics / plots**: `dvc metrics show`, `dvc plots show` (ROC of every model).
- **Model registry**: `artifacts:` in `dvc.yaml` describes the 4 models (path, type, labels).

## FastAPI (`api/main.py`, uses `src/demo.py`)
`uvicorn api.main:app --port 8000`, then http://localhost:8000/docs
| Endpoint | |
|---|---|
| `GET /health` | status + default model, device, threshold |
| `GET /models` | the four models: ready or not, parameters, test AUC / EER (the web page's model list) |
| `POST /analyze?model=...` | upload a video -> every person (time seen, main walking direction), LOITERING / COUNTER_FLOW / TOO_FAST events (who, when, why), per-frame scores, abnormal frames, link to the result video |
| `GET /results/{job}/video` | the annotated video (H.264, plays in a browser) |
| `GET /metrics` | Prometheus metrics |
Default model: `configs/api.yaml` or `VAD_MODEL=astt_baseline|astt_improved|mulde|mulde_improved|none`;
`?model=` picks another one per request (loaded on first use, at most `max_models_loaded` kept in
GPU memory). A model that isn't trained + evaluated yet serves the behavior rules only, with a
`warning`. Only the newest `keep_jobs` results are kept on disk.

## Web page (`ui/app.py`, Gradio)
http://localhost:7860: upload a video, pick a model (with its test AUC), press **Analyze**. Shows the
result video, "N of M people behave abnormally" with who / what / when, a table of every person and
their main direction, the anomaly score over time, and the raw JSON to download. It is only a
client of the API (no PyTorch in its container), so the same API serves the page, scripts and Grafana.
Without Docker: `pip install -r requirements-ui.txt && python -m ui.app` (API_URL, default :8000).

## Docker
`docker compose up --build -d` starts 4 containers: `ui` (web page), `api` (FastAPI, model weights
mounted from `outputs/`), `prometheus`, `grafana`. GPU: add `-f docker-compose.gpu.yml`.
The models must be trained (or `dvc pull`ed) first, so that `outputs/training/<model>/model.pth`
and `outputs/evaluation/<model>/metrics.json` exist.

## Monitoring (Prometheus + Grafana)
Grafana http://localhost:3000 (admin/admin), dashboard **DSP391m - Video anomaly detection**:
- **Serving**: requests/s, latency p50/p95, error rate, time per pipeline step (frames, tracking,
  model, render), frames/s, behavior events per type,
  **frame-score distribution vs. the threshold and share of abnormal frames: if these drift
  upwards, incoming videos no longer look like the training data (data drift)**.
- **Training (live)**: `src/train.py` serves its metrics on port 8001 while it runs: loss, learning
  rate, it/s, progress, time left, epoch.
Prometheus config: `monitoring/prometheus.yml`; dashboard JSON: `monitoring/grafana/dashboards/`.
