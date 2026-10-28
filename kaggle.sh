#!/usr/bin/env bash
# Train ONE model on a free Kaggle GPU while the laptop trains another (docs/kaggle.md).
# Kaggle notebook, one cell:
#   !git clone -q https://github.com/Noahisgreatt/DSP391m /tmp/DSP391m && bash /tmp/DSP391m/kaggle.sh astt_improved
# Notebook settings: Accelerator = GPU, Internet = on, Input = the two ShanghaiTech datasets
# (nikanvasei/shanghaitech-campus-dataset, nikanvasei/shanghaitech-campus-dataset-test) and, to
# continue a run, this notebook's previous output. Then "Save Version" -> "Save & Run All".
# Result: /kaggle/working/<model>/ (model.pth when finished, checkpoint.pth to continue).
set -eo pipefail
MODEL=${1:-astt_improved}
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
pip install -q safetensors gdown
python third_party/fetch.py
python -m src.models.nets download "$MODEL"
python -m src.data link /kaggle/input/*/                        # the unpacked videos: linked, not copied
python -m src.data split
OUT=outputs/training/$MODEL
PREV=$(ls -d /kaggle/input/*/"$MODEL" 2>/dev/null | head -1 || true)
if [[ -n "$PREV" && ! -f "$OUT/checkpoint.pth" ]]; then         # continue the previous session's run
    mkdir -p "$OUT" && cp "$PREV"/* "$OUT"/ && echo "continuing from $PREV"
fi
python -m src.train --config "configs/$MODEL.yaml" --max-hours 11   # Kaggle stops a session at 12 h
mkdir -p "/kaggle/working/$MODEL" && cp "$OUT"/* "/kaggle/working/$MODEL"/
ls -lh "/kaggle/working/$MODEL"
