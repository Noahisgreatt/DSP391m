# reports/

Material for the written report: the results, the figures and where each one comes from.
Everything here is produced by code in this repository, so it can be checked and remade.

## Results (v4.2: 107 ShanghaiTech test videos, 40,791 frames, 17,326 abnormal)

| Model | Parameters | AUC micro | AUC per-video | AUC macro | EER | Frames/s |
|---|---|---|---|---|---|---|
| 1 ASTT, the thesis model (4 epochs; results so far: after epoch 2) | 150.6M | 47.0% | 63.9% | 67.7% | 40.2% | 35.5 |
| 2 ASTT improved (ours, 4 epochs) | 100.4M | 57.6% | 72.6% | **81.0%** | 33.0% | **84.2** |
| 3 MULDE, CVPR 2024 | 70.7M | **66.7%** | **76.2%** | 75.1% | **29.9%** | 15.3 |
| 4 MULDE improved (ours) | 58.1M | 66.3% | 71.4% | 73.5% | 33.5% | 7.9 |
| 5 ASTT lite (ours) | 64.9M | to come | | | | |
| 6 ASTT lite + smoothing (ours) | 64.9M | to come | | | | |
| Behavior rules (no training) | 0 | 59.9% | - | - | 40.1% | - |

Model 1 is being trained to 4 epochs (= model 2's budget); its row will be replaced. Frames/s:
RTX 4050 Laptop, `python -m src.evaluate --benchmark`. Interpretation: `docs/models.md`.

## Figures and tables

| File | What | Made by |
|---|---|---|
| `../outputs/comparison/comparison.md` | the comparison table above | `python -m src.evaluate --compare` (DVC stage `compare`) |
| `../outputs/comparison/roc.png` | ROC curves of all models | same |
| `../outputs/comparison/benchmark.md` | parameters, ms / frame, frames/s, GPU memory | `python -m src.evaluate --benchmark` |
| `figures/eda_*.png` | the data: videos per camera, video length, abnormal share, label timeline, the 13 cameras | `notebooks/01_eda.ipynb` |
| `figures/err_*.png` | error analysis: AUC per camera, raw scores per camera (why micro < per-video), score curves, score distributions, smoothing gain (model 5 vs 6) | `notebooks/02_error_analysis.ipynb` |

Remake every figure (after the models are evaluated):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_error_analysis.ipynb
```

## Checking the numbers
`python -m src.verify` recomputes every AUC / EER from the saved per-frame scores and the dataset's
own labels, with scikit-learn and by hand, plus sanity checks (random scores ~50%, the labels
100%), and compares them with `outputs/evaluation/<model>/metrics.json`. Put its output in the
report's appendix.

## Versions
Every result belongs to a version (`releases/README.md`, `python -m src.release --list`);
`python -m src.release --diff v4.2 v5.0` says what changed between two of them.
