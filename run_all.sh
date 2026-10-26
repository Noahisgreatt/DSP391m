#!/usr/bin/env bash
# ONE command for everything: setup -> data -> 4 models (train + evaluate, a version after each)
# -> comparison + benchmark.
#
#   bash run_all.sh            full run (days on one GPU). Stopped? Run it again: it continues.
#   bash run_all.sh --smoke    quick check that every step works (tiny training runs)
#
# Needs: Linux or WSL2 Ubuntu, an NVIDIA GPU (>= 6 GB), ~100 GB free disk, conda or Python >= 3.10,
# and the data: either access to our DVC remote, or a Kaggle API token (~/.kaggle/kaggle.json).
set -eo pipefail
cd "$(dirname "$0")"
MODE=full; [[ "${1:-}" == "--smoke" ]] && MODE=smoke
MODELS=(astt_baseline astt_improved aivad aivad_lite)
step() { printf '\n========== %s\n' "$*"; }
die()  { printf '\nSTOPPED: %s\nFix that, then run the same command again (finished steps are skipped).\n' "$*" >&2; exit 1; }

step "1/7 Python environment"
if [[ "${CONDA_DEFAULT_ENV:-}" == "dsp391m" || -n "${VIRTUAL_ENV:-}" ]]; then
    echo "using the active environment"
elif command -v conda >/dev/null; then
    eval "$(conda shell.bash hook)"
    conda env list | grep -q '^dsp391m ' || conda create -y -q -n dsp391m python=3.10
    conda activate dsp391m
else
    [[ -d .venv ]] || python3 -m venv .venv || die "no conda and no python3-venv (sudo apt install python3-venv)"
    source .venv/bin/activate
fi
python -c "import sys; assert sys.version_info >= (3, 10)" || die "Python 3.10 or newer is needed"
echo "python: $(command -v python)"

step "2/7 Packages"
python -m pip install -q -r requirements.txt
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
    if ! command -v gcc >/dev/null; then          # PyTorch's Triton kernels compile a small C file
        echo "installing gcc (needs your Ubuntu password once)"
        sudo apt-get update -qq && sudo apt-get install -y -qq build-essential || die "install gcc: sudo apt install build-essential"
    fi
else
    [[ $MODE == full ]] && die "no NVIDIA GPU visible to PyTorch: the full run needs one (try --smoke)"
    echo "no GPU: fine for --smoke, slow"
fi

step "3/7 Code of the papers (third_party/, pinned + checksummed)"
python third_party/fetch.py

step "4/7 Pretrained weights (weights/): download + check each one loads"
python -m src.models.nets download astt_improved aivad aivad_lite behavior \
    || die "a weight file could not be downloaded: get it with the browser link printed above and save it in weights/"

step "5/7 Data (ShanghaiTech Campus + our phone videos)"
if [[ -f data/raw/shanghaitech/train.zip && -f data/raw/shanghaitech/test.zip ]]; then
    echo "ShanghaiTech zips already here"
elif dvc pull -q data/raw/shanghaitech.dvc 2>/dev/null; then
    echo "ShanghaiTech pulled from the DVC remote"
elif [[ -f ~/.kaggle/kaggle.json || -n "${KAGGLE_KEY:-}" ]]; then
    echo "no access to the DVC remote: downloading from Kaggle (~21 GB)"
    python -m src.data download || die "Kaggle download failed"
else
    die "no data. Either get access to the DVC remote (dvc pull), or put a Kaggle API token in
~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)"
fi
MANUAL=no
if [[ -d data/raw/manual ]] || dvc pull -q data/raw/manual.dvc 2>/dev/null; then MANUAL=yes; fi
echo "phone videos: $MANUAL (only used by the behavior rules)"
dvc repro split                                   # unpack + official split: 330 train / 107 test videos

step "6/7 Behavior rules + models: train + evaluate ($MODE)"
if [[ $MODE == smoke ]]; then
    python -m src.behavior --dataset shanghaitech --limit 1
    for m in "${MODELS[@]}"; do
        echo "--- $m"
        python -m src.train    --config "configs/$m.yaml" --smoke
        python -m src.evaluate --config "configs/$m.yaml" --smoke
    done
    step "7/7 Done: every step works. Full run: bash run_all.sh"
    exit 0
fi
dvc repro behavior@shanghaitech                   # the rules need no training
if [[ $MANUAL == yes ]]; then dvc repro behavior@manual; fi
for m in "${MODELS[@]}"; do                       # model by model; a version after each:
    dvc repro "evaluate@$m"                       # v1.0 = model 1, v2.0 = + model 2, ... (docs/versioning.md)
    python -m src.release --note "$m trained + evaluated" || true    # "nothing changed" = no new version
done
dvc repro compare benchmark

step "7/7 Results"
cat outputs/comparison/comparison.md outputs/comparison/benchmark.md
python -m src.release --list
echo "upload code, versions, data and models:  git push --follow-tags && dvc push"
