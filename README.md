# DSP391m: Video anomaly detection (loitering and counter-flow)

Find abnormal behaviour in surveillance video: people **loitering**, people moving **against the
crowd** (counter-flow) or **much faster** than it, and anything else the models learn is unusual.
Data: ShanghaiTech Campus (official split: 330 normal training videos, 107 test videos with frame
labels) + our own phone videos. Four models are trained and compared on the same test set.

## Run everything with one command

```bash
git clone https://github.com/Noahisgreatt/DSP391m.git
cd DSP391m
bash run_all.sh --smoke     # first: quick check that every step works (tiny training runs)
bash run_all.sh             # then: the real thing, all 4 models from start to finish
```

**Stopped halfway** (power cut, closed laptop, an error you fixed)? Run the same command again:
finished steps are skipped and training continues from its last checkpoint (saved every 20 min).
To keep it running after you close the terminal:
`nohup bash run_all.sh > run_all.log 2>&1 &` and watch it with `tail -f run_all.log`.

### What you need
| | |
|---|---|
| Computer | Linux, or Windows with WSL2 Ubuntu |
| GPU | NVIDIA, 6 GB or more (developed on an RTX 4050 Laptop) |
| Disk | about 100 GB free (data 21 GB zipped + 21 GB unpacked, weights 2 GB, features, models) |
| Python | conda (the script makes an env `dsp391m`), or Python >= 3.10 (it makes `.venv`) |
| Data | access to our DVC remote (ask the project owner), **or** a free Kaggle API token in `~/.kaggle/kaggle.json` (kaggle.com -> Settings -> API -> Create New Token) |

### What `run_all.sh` does
| Step | What | Time (rough, RTX 4050) |
|---|---|---|
| 1 | Python environment (conda env `dsp391m` or `.venv`) | minutes |
| 2 | Packages from `requirements.txt` (+ `gcc`, which PyTorch's GPU kernels need) | minutes |
| 3 | Original code of the papers into `third_party/`, checked byte-for-byte | 1 min |
| 4 | Pretrained weights into `weights/` (2 GB) | minutes |
| 5 | Data: download, unpack, official split (437 videos -> 330 train / 107 test) | ~1 h the first time |
| 6 | Behavior rules, then train + evaluate model 1, 2, 3, 4, saving version v1.0 ... v4.0 after each (`--smoke`: tiny runs) | full: 1-2 days in total |
| 7 | Print the comparison table, the speed benchmark and the list of versions | - |

### Results
| File | Content |
|---|---|
| `outputs/comparison/comparison.md` | **the main table**: AUC (micro / per-video / macro), EER, parameters, speed of every model |
| `outputs/comparison/roc.png` | ROC curves of all models |
| `outputs/comparison/benchmark.md` | parameters, ms per frame (per step), frames/s, GPU memory |
| `outputs/evaluation/<model>/metrics.json` | the numbers of one model |
| `outputs/training/<model>/` | trained model (`model.pth`), training log, settings used (`run.json`) |

## The four models

| # | Model | What it is | Parameters |
|---|---|---|---|
| 1 | `astt_baseline` | The thesis model (Tran Minh Tung, ASTT), **unchanged**, trained from scratch | 150.6M |
| 2 | `astt_improved` | Our upgrade of model 1 (pretrained ViT = transfer learning, spatial tokens, residual prediction) | 100.4M |
| 3 | `aivad` | AI-VAD (Reiss & Hoshen, TMLR 2025) with its original networks | 331.0M |
| 4 | `aivad_lite` | Our lighter AI-VAD (Keypoint R-CNN + FlowNetC, per-camera, track-aware) | 98.3M |
| - | behavior rules | Tracking + rules: LOITERING / COUNTER_FLOW / TOO_FAST per person | 0 |

**No model comes from a library call.** Every network (ViT, CLIP image encoder, Faster / Keypoint
R-CNN, FastPose, FlowNet2) is PyTorch code in this project; only the pretrained *weights* are
downloaded. `tests/test_nets.py` shows each gives the same output as the reference implementation;
`tests/test_no_wrappers.py` builds all four models with timm, ultralytics, CLIP and detectron2
blocked. Details and the reason for every change: [docs/models.md](docs/models.md).

## Folder map

```
run_all.sh           ONE command for everything (above)
kaggle.sh            train one model on a free Kaggle GPU at the same time (docs/kaggle.md)
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
  release.py        versions v1.0, v1.1, v2.0 ... with metadata (what changed)
api/main.py       FastAPI service + Prometheus metrics          monitoring/  Prometheus + Grafana
tests/            pytest                                          docs/        notes for the report
third_party/      thesis / AI-VAD / AlphaPose code (downloaded at fixed versions, never edited)
dvc.yaml          the pipeline run_all.sh uses (dvc dag draws it)
data/   (DVC)     raw/ -> processed/ ; metadata/ (data cards, split)
outputs/(DVC)     training/<model>/  evaluation/<model>/  behavior/  comparison/
```

## One step at a time (what run_all.sh calls)
```bash
python -m src.train    --config configs/astt_baseline.yaml --smoke   # 2-minute test of one model
python -m src.train    --config configs/astt_baseline.yaml   # train one model (resumes if stopped)
python -m src.evaluate --config configs/astt_baseline.yaml   # AUC / EER on the 107 test videos
python -m src.evaluate --compare                             # table of all evaluated models
python -m src.evaluate --benchmark                           # size and speed of every model
python -m src.behavior --dataset shanghaitech                # loitering / counter-flow rules
python -m src.demo --video clip.mp4 --model aivad_lite       # one video -> outputs/demo/clip/
python -m src.release --note "what changed"                  # save a new version (v1.0, v1.1, v2.0 ...)
python -m pytest -q tests                                    # everything still works?
```
Same with DVC (records what was run): `dvc repro train@astt_baseline`, `dvc repro evaluate@aivad`,
`dvc repro compare`.

## Versions (v1.0, v1.1, v2.0 ...)
Every change gets a version with a metadata card (data md5, code, every setting, each model's
weights md5 and test metrics). The number is chosen automatically: a **new model** -> v2.0, v3.0;
**anything else changed** (data, code, a setting, retrained weights) -> v2.1, v2.2.
```bash
python -m src.release --note "learning rate 2e-4"   # make the next version (git commit + tag)
python -m src.release --list                       # all versions (also in releases/README.md)
python -m src.release --diff v1.0 v2.1             # what changed, in words
git checkout v1.0 && dvc pull                      # exactly that version back
```
`run_all.sh` saves one after each model (v1.0 = model 1 ... v4.0 = all four). Details:
[docs/versioning.md](docs/versioning.md).

## MLOps
DVC (data, models, pipeline, versions v1-v4 with metadata cards: [docs/versioning.md](docs/versioning.md)),
FastAPI (`uvicorn api.main:app` -> http://localhost:8000/docs), Docker + Prometheus + Grafana
(`docker compose up --build -d` -> Grafana http://localhost:3000). Details: [docs/mlops.md](docs/mlops.md).

## Where is ...? (for questions in the defence)
| Question | Answer |
|---|---|
| How do I run everything? | `bash run_all.sh` |
| Where is the thesis model? Is it really unchanged? | `src/models/astt.py` (ASTTThesis) on top of `third_party/ASTT`; `python third_party/fetch.py --check` proves the repo files are unedited; `tests/test_models.py` checks 150,579,876 parameters |
| Did you use a model API / wrapper? | No: `src/models/nets.py`; proof in `tests/test_nets.py` and `tests/test_no_wrappers.py` |
| What did you improve? | `configs/astt_improved.yaml`, `configs/aivad_lite.yaml`: every change marked `[CHANGE]`; reasons in docs/models.md |
| How are loitering / counter-flow decided? | `src/behavior.py` (functions `loitering`, `counter_flow`); thresholds in `configs/behavior.yaml` |
| How is accuracy measured? | `src/evaluate.py` (one metric code for all models) |
| Which videos are train / test? | `data/metadata/split.json`, made by `src/data.py` |
| What changed between two versions? | `python -m src.release --diff v1.0 v2.0` (or `dvc metrics diff v1.0 v2.0`) |
| How do I get exactly version 1 back? | `git checkout v1.0 && dvc pull` |
