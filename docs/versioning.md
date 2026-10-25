# Versions (DVC + git tags)

A version = one git tag. The tag pins the code, `dvc.lock` pins the exact data and model files
(by md5, stored on the DVC remote), and `releases/<version>.json` is its metadata card.

| Version | Contents | Main question it answers |
|---|---|---|
| **v1.0** | data (ShanghaiTech official split + manual videos, data cards) + model 1 + behavior rules | how good is the thesis model on our data? |
| **v2.0** | + model 2 | does our ASTT upgrade beat the thesis model? |
| **v3.0** | + model 3 | how does a strong, interpretable published method (AI-VAD) do? |
| **v4.0** | + model 4, full comparison + benchmark | can AI-VAD be made much lighter without losing accuracy? |
| v5.0 | TensorRT engines of models 2 and 4 | how much faster in deployment? |

## What the metadata of a version contains (`releases/<version>.json`)
- `git_commit`, creation time, note
- **data**: DVC md5 of `data/raw/shanghaitech` (the two zips) and `data/raw/manual`; the data
  cards (`data/metadata/*.json`: source, videos, frames, abnormal frames, checks); split sizes
- **models**: per model the config file + hash, commit it was trained from, parameters per part,
  epochs, final loss, and the test metrics (AUC micro / norm / macro, EER, threshold, speed)
- **behavior**: event counts and the rules' frame-level result
- **comparison**: the table of all models (if made)
- **software**: Python, PyTorch, torchvision, CUDA, GPU, DVC versions

## Making a version (example: v1.0)
```bash
dvc repro split behavior_shanghaitech train@astt_baseline evaluate@astt_baseline
python -m src.release v1.0 --note "data + model 1 (thesis ASTT)"
git add releases dvc.lock data/metadata outputs
git commit -m "release v1.0"
git tag -a v1.0 -m "data + model 1"
git push --follow-tags && dvc push
```
v2.0: `dvc repro train@astt_improved evaluate@astt_improved`, `python -m src.evaluate --compare`,
then the same release steps with `v2.0`. v3.0 / v4.0 likewise with `train@aivad` / `train@aivad_lite`
(for AI-VAD, "training" = extracting the attributes and fitting the density models);
from v4.0 on, `dvc repro compare benchmark` covers everything.

## Comparing versions
```bash
dvc metrics diff v1.0 v2.0                 # every number that changed
dvc params diff v1.0 v2.0                  # every setting that changed
dvc plots diff v1.0 v2.0                   # ROC curves / training loss, as an HTML page
git diff v1.0 v2.0 -- releases/            # the metadata cards
git checkout v1.0 && dvc pull              # exactly the code, data and models of v1.0
```
