"""Check the reported accuracy YOURSELF, independently of src/evaluate.py.

    python -m src.verify                     # all evaluated models
    python -m src.verify --model astt_improved

Reads only two kinds of files and recomputes everything from them:
  outputs/evaluation/<model>/scores/<video>.npz   the model's anomaly score for every test frame
                                                  (written by the evaluation; "score" = raw score)
  data/processed/shanghaitech/labels/<video>.npy  the dataset's own ground truth (0 normal / 1 abnormal
                                                  per frame, copied unchanged from ShanghaiTech's
                                                  test_frame_mask/)
AUC is computed twice: with scikit-learn, and by hand from its definition (the probability that a
random abnormal frame scores higher than a random normal one = Mann-Whitney U / (n_abnormal x
n_normal)). Two sanity checks show the code is not rigged: random scores give ~50%, the labels
themselves give 100%. Finally the numbers are compared with outputs/evaluation/<model>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve

from src.models import MODELS                                 # the one list of models (names only)
LABELS = Path("data/processed/shanghaitech/labels")


def auc_by_hand(y, s):
    """AUC from its definition: share of (abnormal, normal) frame pairs where the abnormal frame has
    the higher score (ties count half). Computed with ranks so 40,000 frames take no time."""
    r = rankdata(s)                                     # ties get the average rank
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def eer(y, s):
    """Equal error rate: the point where false alarms = missed anomalies."""
    fpr, tpr, _ = roc_curve(y, s)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def normalise(s, mse):
    """Per-video normalisation of the usual ShanghaiTech protocol: ASTT (scores = prediction MSE) via
    PSNR = 10 log10(4 / MSE) min-max scaled inside the video; MULDE min-max of the score itself."""
    if mse:
        s = -10 * np.log10(4.0 / np.maximum(s, 1e-10))  # minus: low PSNR = abnormal
    return (s - s.min()) / max(s.max() - s.min(), 1e-12)


def check(model):
    folder = Path(f"outputs/evaluation/{model}/scores")
    files = sorted(folder.glob("*.npz"))
    if not files:
        print(f"{model}: no {folder}/ (evaluate it first: python -m src.evaluate --config configs/{model}.yaml)")
        return
    mse = model.startswith("astt")
    ys, raw, norm, per_video = [], [], [], []
    for f in files:
        s = np.load(f)["score"].astype(np.float64)
        y = np.load(LABELS / f"{f.stem}.npy").astype(int).ravel()   # ground truth straight from the dataset
        n = min(len(s), len(y))
        s, y = s[:n], y[:n]
        ys.append(y); raw.append(s); norm.append(normalise(s, mse))
        # macro AUC (Georgescu et al. 2021, as MULDE reports it): per video, with one dummy normal and one
        # dummy abnormal frame so videos with a single class still count
        per_video.append(roc_auc_score(np.r_[0, y, 1], np.r_[0, s, np.finfo(np.float64).max]))
    y, r, nm = np.concatenate(ys), np.concatenate(raw), np.concatenate(norm)
    rng = np.random.default_rng(0)
    mine = {"auc_raw": roc_auc_score(y, r), "auc_norm": roc_auc_score(y, nm), "auc_macro": float(np.mean(per_video)),
            "eer": eer(y, nm)}
    print(f"\n=== {model}: {len(files)} test videos, {len(y):,} frames, {int(y.sum()):,} abnormal ===")
    print(f"  AUC micro   sklearn {mine['auc_raw']:.4f}   by hand {auc_by_hand(y, r):.4f}")
    print(f"  AUC norm    sklearn {mine['auc_norm']:.4f}   by hand {auc_by_hand(y, nm):.4f}")
    print(f"  AUC macro   {mine['auc_macro']:.4f}   (mean of {len(per_video)} per-video AUCs)")
    print(f"  EER         {mine['eer']:.4f}")
    print(f"  sanity: random scores -> {roc_auc_score(y, rng.random(len(y))):.4f} (should be ~0.5), "
          f"labels as scores -> {roc_auc_score(y, y):.4f} (should be 1.0)")
    m = Path(f"outputs/evaluation/{model}/metrics.json")
    if m.exists():
        rep = json.loads(m.read_text())
        print("  vs metrics.json: " + ", ".join(
            f"{k} {rep[k]:.4f} {'OK' if abs(rep[k] - v) < 5e-4 else f'DIFFERENT (here {v:.4f})'}"
            for k, v in mine.items() if rep.get(k) is not None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), help="one model (default: all)")
    a = ap.parse_args()
    for m in [a.model] if a.model else MODELS:
        check(m)


if __name__ == "__main__":
    main()
