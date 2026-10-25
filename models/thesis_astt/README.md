# Model 1: thesis model (ASTT), unchanged

Tran Minh Tung's PhD thesis, Chapter 3 (eq. 3.6-3.12), and the paper
"Transformer-based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos",
IEEE TCSVT 34 (2024) 8292-8309. Code: https://github.com/Tungufm/ASTT (commit 1d05305).

**This is the baseline to improve. Keep it unchanged; put improvements in a new folder
(e.g. `models/thesis_astt_improved/`) so both can be compared.**

| File | What it is |
|---|---|
| `original_repo/` | The GitHub repo, byte-for-byte. Prove it: `cd original_repo && sha256sum -c ../original_repo_SHA256SUMS` |
| `original_repo_SOURCE.md` | Where it came from, licence note, checksums |
| `model.py` | The final thesis model. The repo never uploaded it (`model.py` is imported by its train script but missing), so this file takes the repo's `model_ste_tte.VisionTransformer` and switches on the cross-attention that is commented out there (line 351), applied as in `model_baseline_ca.py` (line 386). Nothing else changes. |
| `train.py` | The repo's training loop and data loader (`original_repo/data_utils.py`) |
| `evaluate.py` | The repo's test procedure (per-frame MSE), plus AUC / EER |
| `config.yaml` | All repo defaults (`original_repo/config.py`) |

## Run
```bash
python -m src.data.split --config models/thesis_astt/config.yaml          # check train/val/test
python -m models.thesis_astt.train --config models/thesis_astt/config.yaml --smoke   # 2-min test
python -m models.thesis_astt.train --config models/thesis_astt/config.yaml --run-name full
python -m models.thesis_astt.evaluate --run runs/thesis_astt/full         # AUC / EER on test
```

## If training stops (shutdown, crash)
A checkpoint is saved every `checkpoint_minutes` (20) as `resume.pth`, and after every epoch as
`current.pth`. Continue with:
```bash
python -m models.thesis_astt.train --resume runs/thesis_astt/full
```
It picks the newest checkpoint. After a mid-epoch resume, the rest of that epoch is drawn in a new
random order (the repo has no resume at all; this is a procedure change only, not a model change).

## How it works
4 past frames -> each through a ViT-B/32 (one spatial token per frame) -> temporal transformer
(2 layers) -> cross-attention -> decoder draws the next frame (256x256). Trained on normal
videos only, so a badly predicted frame (high MSE) is abnormal. One score per frame.

## Parameters: 150,579,876
STE ViT-B/32-384 87,528,192 · TTE 9,460,224 · cross-attention 2,360,064 · decoder 51,230,627 ·
unused sphere params kept from the repo 769.

## Settings (repo defaults)
pretrained=False, BGR input in [-1, 1], frames concatenated newest-first, 384 px in / 256 px out,
SGD lr 1e-4 momentum 0.9 wd 1e-4, OneCycleLR (500 warm-up), batch 8, 5 epochs, fp32.

## What differs from the repo (procedure only, not the model)
- `.cuda()` in forward -> the model's device (same on GPU)
- data = our ShanghaiTech split; best epoch chosen on validation videos (the repo uses a test video)
- `train_steps: auto` (the repo's fixed 38000 crashes on bigger datasets)
- checkpoints every 20 min + `--resume` (the repo can't continue an interrupted run)
- out of GPU memory? set `batch_size: 4` in config.yaml (keeps the model the same)

## Ideas for the improved version (none applied here)
ImageNet-pretrained ViT (`pretrained=True`); RGB input; oldest-to-newest frame order; bf16 mixed
precision; drop the unused sphere parameters; score each tracked person by ASTT's error inside
their box (person-level output); optical flow as an extra input; per-scene thresholds.
