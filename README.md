# DSP391m: Video anomaly detection (loitering and counter-flow)

Find abnormal behaviour in surveillance video: people **loitering**, people moving **against the
crowd** (counter-flow) or **much faster** than it, and anything else the models learn is unusual.
Data: ShanghaiTech Campus (official split: 330 normal training videos, 107 test videos with frame
labels) + our own phone videos.

| # | Model | What it is | Parameters |
|---|---|---|---|
| 1 | `astt_baseline` | The thesis model (Tran Minh Tung, ASTT), **unchanged**, trained from scratch | 150.6M |
| 2 | `astt_improved` | Our upgrade of model 1 (pretrained ViT = transfer learning, spatial tokens, residual prediction) | 100.4M |
| 3 | `aivad` | AI-VAD (Reiss & Hoshen, TMLR 2025) with its original networks | 331.0M |
| 4 | `aivad_lite` | Our lighter AI-VAD (Keypoint R-CNN + FlowNetC, per-camera, track-aware) | 98.3M |
| - | behavior rules | Tracking + rules: LOITERING / COUNTER_FLOW / TOO_FAST per person | 0 |

**No model comes from a library call.** Every network (ViT, CLIP image encoder, Faster / Keypoint
R-CNN, FastPose, FlowNet2) is PyTorch code in this project, loaded with pretrained weights where
the method uses them. `tests/test_nets.py` shows each gives the same output as the reference
implementation; `tests/test_no_wrappers.py` builds all four models with timm, ultralytics, CLIP
and detectron2 blocked. Details: [docs/models.md](docs/models.md).

## Folder map

```
configs/             ALL settings
  data.yaml  behavior.yaml  api.yaml
  astt_baseline.yaml  astt_improved.yaml  aivad.yaml  aivad_lite.yaml     (one per model)
src/                 ALL code, one file per step
  data.py        1  get, unpack and split the data; feed it to the models
  models/        2  astt.py (models 1, 2)  aivad.py (models 3, 4)  nets.py (the networks)
  train.py       3  train any model (the config decides which)
  evaluate.py    4  AUC / EER of one model, comparison table, size + speed benchmark
  behavior.py    5  people tracking, loitering, counter-flow
  demo.py        6  one video in -> annotated video + JSON out
  utils.py          settings, paths, reading frames
  release.py        metadata card of a version (v1, v2, ...)
api/main.py       FastAPI service + Prometheus metrics          monitoring/  Prometheus + Grafana
tests/            pytest                                          docs/        notes for the report
third_party/      thesis / AI-VAD / AlphaPose code (downloaded at fixed versions, never edited)
data/   (DVC)     raw/ -> processed/ ; metadata/ (data cards, split)
outputs/(DVC)     training/<model>/  evaluation/<model>/  behavior/  comparison/
```

## Setup (WSL Ubuntu, conda env `dsp391m`, Python 3.10)
```bash
pip install -r requirements.txt
python third_party/fetch.py                  # original thesis / AI-VAD / AlphaPose code, checked
python -m src.models.nets download behavior  # pretrained weights the steps need (see each config)
python -m pytest -q tests                    # everything works?
```

## Run (the same commands for every model)
```bash
dvc repro split                                              # data: unpack + split (437 / 330 / 107)
python -m src.train    --config configs/astt_baseline.yaml --smoke   # 2-minute test
python -m src.train    --config configs/astt_baseline.yaml   # stopped? run it again: it continues
python -m src.evaluate --config configs/astt_baseline.yaml
python -m src.evaluate --compare                             # all evaluated models + the rules
python -m src.evaluate --benchmark                           # size and speed of every model
python -m src.demo --video clip.mp4 --model aivad_lite       # one video -> outputs/demo/clip/
```
Or with DVC: `dvc repro train@astt_baseline`, `dvc repro evaluate@aivad`, `dvc repro compare`.

## MLOps
DVC (data, models, pipeline, versions v1-v4 with metadata cards: [docs/versioning.md](docs/versioning.md)),
FastAPI (`uvicorn api.main:app` -> http://localhost:8000/docs), Docker + Prometheus + Grafana
(`docker compose up --build -d` -> Grafana http://localhost:3000). Details: [docs/mlops.md](docs/mlops.md).

## Where is ...? (for questions in the defence)
| Question | Answer |
|---|---|
| Where is the thesis model? Is it really unchanged? | `src/models/astt.py` (ASTTThesis) on top of `third_party/ASTT`; `python third_party/fetch.py --check` proves the repo files are unedited; `tests/test_models.py` checks 150,579,876 parameters |
| Did you use a model API / wrapper? | No: `src/models/nets.py`; proof in `tests/test_nets.py` and `tests/test_no_wrappers.py` |
| What did you improve? | `configs/astt_improved.yaml`, `configs/aivad_lite.yaml`: every change marked `[CHANGE]`; reasons in docs/models.md |
| How are loitering / counter-flow decided? | `src/behavior.py` (functions `loitering`, `counter_flow`); thresholds in `configs/behavior.yaml` |
| How is accuracy measured? | `src/evaluate.py` (one metric code for all models) |
| Which videos are train / test? | `data/metadata/split.json`, made by `src/data.py` |
| How do I get exactly version 1 back? | `git checkout v1.0 && dvc pull` |
