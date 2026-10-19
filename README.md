# DSP391m: Video anomaly detection (loitering and abnormal movement)

Two models are compared on ShanghaiTech:
the **thesis model (ASTT)**, unchanged from Tran Minh Tung's thesis and GitHub repo, and
**CrowdFormer**, our own model, which marks which *person* is abnormal.
A rule-based detector (no training) is a third, simple baseline.

## Folder map

```
DSP391m/
├── README.md              <- you are here
├── params.yaml            shared settings: frame size, flow, tracking, rule thresholds
├── dvc.yaml               the whole pipeline (dvc repro), in order
│
├── data/                  DATA (not code)
│   ├── raw/SHANGHAI/        ShanghaiTech frames + labels (DVC)
│   ├── raw/manual/          our own phone videos (DVC, Google Drive)
│   ├── interim/frames/      videos cut into frames (manual/, other/ = one-off videos)
│   └── features/            per-video results of src/features: farneback/ tracking/ events/
│
├── models/                MODEL CODE, one folder per model, each with its own README
│   ├── thesis_astt/         1. thesis model, unchanged
│   │   ├── original_repo/     the GitHub repo, byte-for-byte (do not edit)
│   │   ├── model.py           joins the repo's parts into the final thesis model
│   │   ├── train.py  evaluate.py  config.yaml
│   │   └── README.md
│   └── crowdformer/         2. our own model
│       ├── model.py  dataset.py  train.py  evaluate.py  config.yaml
│       └── README.md
│
├── runs/                  TRAINING OUTPUT, one folder per run (created automatically)
│   └── <model>/<run_id>/    config.yaml, split.json, log.csv, checkpoints, metrics.json
│
├── results/               EVALUATION OUTPUT (created automatically)
│   ├── <model>/<run_id>/    metrics_test.json, scores/, videos/ (labelled result videos)
│   ├── comparison/          comparison.md, comparison.csv, roc.png
│   └── visualize/           arrow videos from src.visualize
│
├── src/                   SHARED PIPELINE (used by both models)
│   ├── data/                extract_frames.py (video -> frames), split.py (train/val/test)
│   ├── features/            optical_flow.py, tracking.py (YOLO + ByteTrack), direction.py
│   ├── baseline_rules.py    rules: LOITERING, ABNORMAL (against crowd / too fast)
│   ├── metrics.py           frame AUC + EER, identical for every model
│   ├── compare.py           comparison table + ROC chart
│   ├── predict.py           any video -> labelled result video
│   ├── visualize.py  utils.py
│
├── configs/               extra settings files (features/, detection/)
└── scripts/               dataset download helpers
```

The rule of thumb: **code** lives in `models/` and `src/`, **what training makes** goes to
`runs/`, **what evaluation makes** goes to `results/`. Nothing in `runs/` or `results/` is
written by hand.

## Run order

```bash
# 0. check which videos are train / val / test (same for both models)
python -m src.data.split --config models/thesis_astt/config.yaml

# 1. features + rule baseline
dvc repro track_shanghai          # people tracking (~1 h on GPU)
dvc repro rules_shanghai

# 2. thesis model
python -m models.thesis_astt.train    --config models/thesis_astt/config.yaml
python -m models.thesis_astt.evaluate --run runs/thesis_astt/<run_id>

# 3. own model
python -m models.crowdformer.train    --config models/crowdformer/config.yaml
python -m models.crowdformer.evaluate --run runs/crowdformer/<run_id>

# 4. compare (table + ROC in results/comparison/)
python -m src.compare --runs runs/thesis_astt/<run_id> runs/crowdformer/<run_id>

# 5. result video for any clip (people boxed with ABNORMAL / LOITERING)
python -m src.predict --run runs/crowdformer/<run_id> --video path/to/clip.MOV
```
Add `--smoke` to a train command for a 2-minute test run.
`<run_id>` is printed when training starts (a timestamp unless you pass `--run-name`).
Or run steps 1-4 with DVC: `dvc repro compare` (its runs are called `dvc`).

## Metrics (same as the thesis, Chapter 5)
- **Frame-level AUC-ROC**, main metric, higher is better. Reported per-video normalised
  ("norm", as most papers) and raw (what the thesis repo prints).
- **EER**, equal error rate, lower is better.
CrowdFormer's frame score = its highest person score in that frame, so both models are measured
on exactly the same frames and labels.
