# Versions: v1.0, v1.1, v2.0 ... (DVC + git)

**Every time something changes, the project gets a new version**, with a metadata card that
says exactly what it is and what changed. One command does it:

```bash
python -m src.release --note "what this version is"
```

`run_all.sh` calls it after each model, so a full run makes v1.0 ... v4.0 by itself.

## 1. How the number is chosen (automatically)

`src.release` compares the project now with the last version:

| What changed since the last version | New version | Example |
|---|---|---|
| a model was **added** | next MAJOR: v1.0 -> **v2.0** | model 2 trained and evaluated |
| same models, but the **data**, the **code**, a **setting** or the **trained weights** changed | next MINOR: v2.0 -> **v2.1** | learning rate changed and model 1 retrained; new phone videos added |
| nothing | no new version | running `run_all.sh` again |

The plan for this project:

| Version | Contents | Question it answers |
|---|---|---|
| **v1.0** | data (ShanghaiTech official split + phone videos) + behavior rules + **model 1** (thesis ASTT) | how good is the thesis model on our data? |
| **v2.0** | + **model 2** (improved ASTT) | does our ASTT upgrade beat the thesis model? |
| **v3.0** | + **model 3** (AI-VAD) | how does a strong published method do? |
| **v4.0** | + **model 4** (AI-VAD Lite) + comparison + benchmark | can AI-VAD be much lighter without losing accuracy? |
| v4.1, v4.2 ... | any change after that (tuning, more data, bug fixes) | did the change help? |
| v5.0 | TensorRT versions of models 2 and 4 | how much faster in deployment? |

## 2. What a version is made of

| Part | Where | Pins |
|---|---|---|
| git tag `v1.0` | git | the code and the settings (configs/) |
| `dvc.lock` (inside that commit) | git | md5 of every data file and model file |
| the files themselves | DVC remote (Google Drive) | the data zips, the phone videos, every `model.pth` |
| `releases/v1.0.json` | git | the **metadata card** (below) |
| `releases/README.md` | git | one table of all versions (made automatically) |

### The metadata card (`releases/<version>.json`)
| Field | Content |
|---|---|
| `version`, `note`, `created_utc`, `git_commit` | which version, why, when, built from which commit |
| `changes` | the list of what changed since the previous version (same text as `--diff`) |
| `data` | per DVC-tracked dataset: md5, size, number of files; videos and frames; train / test split sizes |
| `code_md5` | fingerprint of all code in `src/` and `api/` (any code change -> new md5) |
| `settings` | every value of every `configs/*.yaml` |
| `models` | per model: parameters, md5 of the trained weights, test metrics (AUC micro / per-video / macro, EER, frames/s) |
| `behavior` | events found by the rules and their frame-level result |
| `software` | Python, PyTorch, CUDA, GPU |

## 3. Commands

```bash
python -m src.release --note "model 1 trained"   # make the next version (if anything changed)
python -m src.release --list                     # all versions, one line each
python -m src.release --diff v1.0 v2.1           # what changed between two versions, in words
git push --follow-tags && dvc push               # upload: code + tags to GitHub, data + models to Drive
```

Example of `--diff`:
```
v1.0 -> v2.1:
  setting astt_baseline.train.lr: 0.0001 -> 0.0002
  astt_baseline: retrained (new weights)
  astt_baseline: auc_raw 61.0% -> 63.2% (+2.2)
  astt_improved: ADDED (100,379,299 parameters, AUC 68.4%)
```

DVC's own comparisons work between any two versions too:
```bash
dvc metrics diff v1.0 v2.0        # every number that changed (AUC, EER, loss, data counts)
dvc params diff v1.0 v2.0         # every setting that changed
dvc plots diff v1.0 v2.0          # ROC curves of both versions, as an HTML page
```

## 4. Going back to a version
```bash
git checkout v1.0      # the code, settings and dvc.lock of v1.0
dvc pull               # the exact data and model files of v1.0 (from Google Drive)
git checkout main      # back to the newest
```

## 5. What happens in the background
- `dvc repro` runs only the stages whose inputs changed (code, settings, data: md5 in `dvc.lock`).
- `src.release` writes the card, updates `releases/README.md`, then `git add` + `git commit` +
  `git tag -a <version>` (code, `dvc.lock`, data cards, small results). Big files (data, `model.pth`)
  are not in git: `dvc push` uploads them, and `dvc.lock` remembers their md5.
- It warns if a result is older than its code or settings (`dvc status`), so a version never
  mixes new settings with old results by accident.
