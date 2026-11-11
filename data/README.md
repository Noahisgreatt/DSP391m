# data/

Nothing heavy is in git: the videos are versioned with DVC (`*.dvc` files here, files on our
Google Drive remote). The scripts that download and prepare them are in `src/dataset/`.

| Path | What | In git? |
|---|---|---|
| `raw/shanghaitech.dvc` | ShanghaiTech Campus, the two Kaggle zips (437 videos as frames, 21 GB) | pointer only (DVC) |
| `raw/manual.dvc` | our own phone videos | pointer only (DVC) |
| `processed/<dataset>/frames/<video>/` | unpacked frames, one folder per video | no (made by the scripts) |
| `processed/shanghaitech/labels/<video>.npy` | ground truth, 0 / 1 per test frame | no (made by the scripts) |
| `metadata/shanghaitech.json`, `manual.json` | data cards: source, counts, checks | yes |
| `metadata/split.json` | official split: 330 train / 107 test videos | yes |
| `features/` | cached Hiera-B features (models 3, 4) | no |

Get the data (one of):
```bash
dvc pull data/raw/shanghaitech.dvc        # from our DVC remote (access needed)
python -m src.dataset download            # or from Kaggle (needs ~/.kaggle/kaggle.json)
```
Then prepare it (what `bash run_all.sh` / `dvc repro split` do):
```bash
python -m src.dataset prepare             # unzip + data card
python -m src.dataset manual              # phone videos -> frames
python -m src.dataset split               # official train / test split
```
