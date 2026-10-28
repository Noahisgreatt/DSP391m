# Training a model on Kaggle while the laptop trains another

Kaggle gives a free GPU (T4 or P100, about 30 hours a week, 12 hours per run), and the ShanghaiTech
data is already there, so nothing big has to be uploaded. `kaggle.sh` does everything on Kaggle.

## 1. Make the notebook (once)
1. kaggle.com -> **Create -> New Notebook**. Your account must be phone-verified (Settings) to get a GPU and internet.
2. Right panel, **Settings**: Accelerator = **GPU** (T4 x2 or P100), Internet = **On**.
3. Right panel, **Input -> Add Input**: add the datasets `nikanvasei/shanghaitech-campus-dataset` and
   `nikanvasei/shanghaitech-campus-dataset-test`.
4. Put this in the first cell (and delete the other cells):
   ```
   !git clone -q https://github.com/Noahisgreatt/DSP391m /tmp/DSP391m && bash /tmp/DSP391m/kaggle.sh astt_improved
   ```
5. **Save Version -> Save & Run All (Commit)**. It runs in the background (you can close the browser)
   for up to 11 hours, then saves `astt_improved/` in the notebook's **Output**.

## 2. Continue (if 11 hours were not enough)
Open the notebook -> **Input -> Add Input -> Your Work -> Notebooks** -> this notebook (its last
output) -> **Save & Run All** again. It finds `checkpoint.pth` in that output and continues.

## 3. Bring the result to the laptop
Notebook page -> **Output** -> download the `astt_improved` folder, then in Ubuntu:
```bash
mkdir -p outputs/training/astt_improved
cp /mnt/c/Users/anish/Downloads/astt_improved/* outputs/training/astt_improved/
```
When `run_all.sh` reaches model 2 it finds this run: finished -> no training (straight to evaluation);
not finished -> it continues from Kaggle's checkpoint on the laptop. Copy it before `run_all.sh`
reaches model 2 (watch for `Running stage 'train@astt_improved'` in `run_all.log`).

Note: Kaggle's GPUs have no bf16, so there model 2 trains in fp16 (`src/models/astt.py: half_dtype`);
the settings and the model are the same.
