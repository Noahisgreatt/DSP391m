# MLOps: DVC, FastAPI, web page, Docker, CI/CD, Prometheus, Grafana

```
 browser ──> web page (Gradio, :7860) ──HTTP──> FastAPI (:8000) ──> src/inference.py: tracking + rules + model
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
- **Model registry**: `artifacts:` in `dvc.yaml` describes the trained models (path, type, labels).

## FastAPI (`api/main.py`, uses `src/inference.py`)
`uvicorn api.main:app --port 8000`, then http://localhost:8000/docs
| Endpoint | |
|---|---|
| `GET /health` | status + default model, device, threshold |
| `GET /models` | every model: ready or not, parameters, test AUC / EER (the web page's model list) |
| `POST /analyze?model=...` | upload a video -> every person (time seen, main walking direction), LOITERING / COUNTER_FLOW / TOO_FAST events (who, when, why), per-frame scores, abnormal frames, link to the result video |
| `GET /results/{job}/video` | the annotated video (H.264, plays in a browser) |
| `GET /metrics` | Prometheus metrics |
Every response has a schema (`api/schemas.py`: `AnalyzeResult`, `Event`, `Person`, `ModelList`, `Health`),
shown with examples on `/docs`. Every error is `{"error": "<code>", "detail": "<text>"}`: `unknown_model`,
`empty_file` (400), `not_found` (404), `file_too_large` (413), `unsupported_file_type` (415),
`invalid_request`, `cannot_analyze` (422), `internal_error` (500).
Default model: `configs/api.yaml` or `VAD_MODEL=astt_baseline|astt_improved|mulde|mulde_improved|none`;
`?model=` picks another one per request (loaded on first use, at most `max_models_loaded` kept in
GPU memory). A model that isn't trained + evaluated yet serves the behavior rules only, with a
`warning`. Only the newest `keep_jobs` results are kept on disk.

## Web page (`app/gradio_app.py`, Gradio)
http://localhost:7860: upload a video, pick a model (with its test AUC), press **Analyze**. Shows the
result video, "N of M people behave abnormally" with who / what / when, a table of every person and
their main direction, the anomaly score over time, and the raw JSON to download. It is only a
client of the API (no PyTorch in its container), so the same API serves the page, scripts and Grafana.
Without Docker: `pip install -r requirements-ui.txt && python -m app.gradio_app` (API_URL, default :8000).

## Docker
`docker compose up --build -d` builds `docker/Dockerfile.api` and `docker/Dockerfile.ui` and starts 4 containers: `ui` (web page), `api` (FastAPI, model weights
mounted from `outputs/`), `prometheus`, `grafana`. GPU: add `-f docker-compose.gpu.yml`.
The models must be trained (or `dvc pull`ed) first, so that `outputs/training/<model>/model.pth`
and `outputs/evaluation/<model>/metrics.json` exist.

## The app image (`docker/Dockerfile.app`): everything + model 2 in one container
`app/server.py` serves the web page (`/`) and the API (`/docs`, `/analyze`, ...) on ONE port, and
the image contains model 2's trained weights, so no code, data or DVC is needed to run it:
`docker run --gpus all -p 7860:7860 ghcr.io/<owner>/dsp391m-app`. Built and published from the
computer that has the model: `bash docker/publish.sh v5.0` (once: `docker login ghcr.io`).
Settings by environment: `VAD_MODELS` (offered models, default model 2 only), `VAD_MAX_SECONDS`
(longest part of a video analyzed, default 60; 0 = all), and for a server without a GPU
`VAD_DETECTOR_SIZE` / `VAD_DETECT_EVERY` (find people at a smaller size, on every n-th frame).
The API scores models 2, 5, 6 with `src/models/astt_fast.py`: each frame through the ViT once
instead of 4 times, the same scores (tested), ~2.8x faster on a CPU.

**Live demo, fast:** `bash serve.sh --share` on the laptop (GPU) gives a public link while it runs.

## Online for everyone: Hugging Face Space (free, CPU), deployed by CI/CD
`docker/deploy_space.py` uploads the app's code, the Space page (`docker/huggingface/README.md`)
and a Dockerfile made from `docker/Dockerfile.app` (CPU variant + light mode: first 15 s, people
found at 480 px on every 3rd frame); the Space builds and runs it. Page at
`https://huggingface.co/spaces/<user>/<space>`, API at `https://<user>-<space>.hf.space/docs`.

One-time setup (5 minutes):
1. huggingface.co: create an account, then Settings -> Access Tokens -> a token with **write** access.
2. On the computer with the trained model: `huggingface-cli login` (type the token in the terminal),
   then `python docker/deploy_space.py <user>/dsp391m --with-model`: creates the Space and uploads
   the code + model 2 (~400 MB, once).
3. GitHub repository -> Settings -> Secrets and variables -> Actions: secret `HF_TOKEN` = the token,
   variable `HF_SPACE` = `<user>/dsp391m`.

From then on every version (`python -m src.release`, `git push --follow-tags`) or the **Run
workflow** button (Actions -> CI/CD) runs lint + tests + the app image check and then **deploys
the code to the Space automatically**; the Space keeps model 2's weights and rebuilds itself.

## CI/CD (`.github/workflows/ci.yml`, GitHub Actions)
| When | Job | What it does |
|---|---|---|
| every push / pull request | **Lint** | `ruff check .` (settings: `ruff.toml`) |
| | **Tests** | CPU PyTorch, fetch the thesis code, all tests (`pytest`) |
| | **Docker setup** | compose files valid, web page image builds |
| | **App image** | builds exactly what the Space builds (`deploy_space.py --dry-run --out`), runs it as a non-root user, checks `/health`, `/models`, the page, `/docs`, analyzes a test video through the API and downloads the result video (stand-in weights: CI has no trained model) |
| every version tag + Run workflow button | **Deploy** | uploads the code to the Hugging Face Space (needs `HF_SPACE`, `HF_TOKEN`; skipped without them) |
| every version tag | **Publish** | compose images to GitHub Packages: `ghcr.io/<owner>/dsp391m-api:<version>`, `dsp391m-ui:<version>` |
A job runs only if the earlier ones passed. Results: the repository's **Actions** tab.

## Monitoring (Prometheus + Grafana)
Grafana http://localhost:3000 (admin/admin), dashboard **DSP391m - Video anomaly detection**:
- **Serving**: requests/s, latency p50/p95, error rate, time per pipeline step (frames, tracking,
  model, render), frames/s, behavior events per type,
  **frame-score distribution vs. the threshold and share of abnormal frames: if these drift
  upwards, incoming videos no longer look like the training data (data drift)**.
- **Training (live)**: `src/train.py` serves its metrics on port 8001 while it runs: loss, learning
  rate, it/s, progress, time left, epoch.
Prometheus config: `monitoring/prometheus.yml`; dashboard JSON: `monitoring/grafana/dashboards/`.
