# DSP391m: Video anomaly detection (loitering and counter-flow)

Find abnormal behaviour in surveillance video: people **loitering**, people moving **against
the crowd** (counter-flow) or **much faster** than it, and anything else the models learn is
unusual. Dataset: ShanghaiTech Campus (official split: 330 normal training videos, 107 test
videos with frame labels) plus our own phone videos.

| # | Model | What it is | Parameters | Trained? |
|---|---|---|---|---|
| 1 | `astt_baseline` | The thesis model (Tran Minh Tung, ASTT), **unchanged** | 150.6M | yes, from scratch |
| 2 | `astt_improved` | Our upgrade of model 1 (pretrained ViT, spatial tokens, residual prediction) | 100.4M | yes, fine-tuned |
| 3 | `aivad` | AI-VAD (Reiss & Hoshen, TMLR 2025), original networks | 333.6M | no: pretrained networks + density fit |
| 4 | `aivad_lite` | Our lighter AI-VAD (YOLO11x-pose + RAFT, per-camera, track-aware) | 64.1M | no: same |
| - | behavior rules | Tracking + rules: LOITERING / COUNTER_FLOW / TOO_FAST per person | 0 | no |

Models 1-2 score a frame by how badly they **predict** it; models 3-4 by how **unusual** each
object's motion and pose are. All are measured the same way (frame AUC / EER, `src/evaluation`).
Details and every difference: [docs/models.md](docs/models.md).

## Folder map: one folder = one job

```
configs/            ALL settings (DVC tracks them as parameters)
  data.yaml           paths, dataset, split            behavior.yaml  rule thresholds
  astt_baseline.yaml  model 1     astt_improved.yaml  model 2
  aivad.yaml          model 3     aivad_lite.yaml     model 4     api.yaml  serving + monitoring
src/
  data/         1  download, unpack, split, PyTorch datasets
  models/       2  astt_baseline.py, astt_improved.py, aivad/ (models 3 + 4)
  training/     3  train.py (any model; the config decides), live training metrics
  evaluation/   4  evaluate.py, metrics.py, compare.py, benchmark.py
  behavior/     5  tracking, optical flow, loitering.py, counter_flow.py, detect.py
  inference/    6  analyze ONE video end to end + draw the result video
  release.py       metadata card of a version (v1, v2, ...)
api/            7  FastAPI service + Prometheus metrics
monitoring/        Prometheus + Grafana (dashboard included)
third_party/       original code of ASTT / AI-VAD / AlphaPose (downloaded, checksummed, never edited)
tests/             pytest: models, metrics, rules, data, API
docs/              models, versions, MLOps notes for the report
data/  (DVC)       raw/ (zips, manual videos) -> processed/ -> features/ ; metadata/ (data cards)
outputs/ (DVC)     training/<model>/  evaluation/<model>/  behavior/  comparison/
releases/          one metadata card per version (git-tagged)
```

## Setup (WSL Ubuntu, conda env `dsp391m`, Python 3.10)
```bash
conda activate dsp391m
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"  # model 3 only
python third_party/fetch.py              # original ASTT / AI-VAD / AlphaPose code, verified
python -m pytest -q tests                # everything works? (~1 min)
```

## Data
```bash
dvc pull                                  # if the data is already on our DVC remote, or:
python -m src.data.download               # Kaggle zips (or --pack-train/--pack-test DIR)
dvc add data/raw/shanghaitech && dvc push
dvc repro split                           # unpack + data card + official split (437 / 330 / 107)
```

## Run (same commands for every model)
```bash
python -m src.training.train      --config configs/astt_baseline.yaml --smoke   # 2-minute test
python -m src.training.train      --config configs/astt_baseline.yaml           # stopped? run it again
python -m src.evaluation.evaluate --config configs/astt_baseline.yaml
python -m src.evaluation.compare                     # all evaluated models + rules -> outputs/comparison/
python -m src.evaluation.benchmark                   # size + speed of every model
python -m src.inference.analyze --video clip.mp4 --model aivad_lite             # one video -> result.mp4
```
Or with DVC: `dvc repro train@astt_baseline`, `dvc repro evaluate@aivad`, `dvc repro compare`.
Model 3 also needs `python -m src.models.aivad.weights --model aivad` first.

## MLOps
- **DVC**: data, models and the whole pipeline (`dvc dag`); versions v1-v4 with metadata cards
  ([docs/versioning.md](docs/versioning.md)); `dvc metrics diff v1.0 v2.0`.
- **FastAPI**: `uvicorn api.main:app` -> http://localhost:8000/docs
- **Docker + Prometheus + Grafana**: `docker compose up --build -d` -> API :8000, Grafana :3000
  (dashboard "DSP391m - Video anomaly detection": serving and live training).
Details: [docs/mlops.md](docs/mlops.md).

## Where is ...? (for questions in the defence)
| Question | Answer |
|---|---|
| Where is the thesis model? Is it really unchanged? | `src/models/astt_baseline.py` builds it from `third_party/ASTT` (checksums: `python third_party/fetch.py --check`); `pytest tests/test_models.py` checks the 150,579,876 parameters |
| What did you improve? | `configs/astt_improved.yaml` and `configs/aivad_lite.yaml`, every change marked `[CHANGE]`; reasons in docs/models.md |
| How is loitering / counter-flow decided? | `src/behavior/loitering.py`, `src/behavior/counter_flow.py`; thresholds in `configs/behavior.yaml` |
| How is accuracy measured? | `src/evaluation/metrics.py` (one file for all models) |
| Which videos are train / test? | `data/metadata/split.json`, made by `src/data/split.py` |
| How do I get exactly version 1 back? | `git checkout v1.0 && dvc pull` |
| Where are training logs / results? | `outputs/training/<model>/`, `outputs/evaluation/<model>/`, `outputs/comparison/` |
