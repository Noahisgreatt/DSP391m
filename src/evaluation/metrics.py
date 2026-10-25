"""Frame-level metrics, computed the same way for every method (models 1-4 and the rules).

Score per frame: models 1-2 = prediction error (MSE) of the frame they had to predict;
models 3-4 = AI-VAD's calibrated, smoothed attribute score. Higher = more abnormal for all.
  auc_raw    AUC on the raw scores of all test frames pooled together ("micro" AUC; what the
             thesis repo prints, and AI-VAD's main number, 85.9% on ShanghaiTech)
  auc_norm   AUC after min-max normalising the score PER VIDEO (for MSE: of its PSNR). Videos
             differ in brightness / crowd size, so raw errors of different videos are not
             comparable; normalising removes that. This is why the old ASTT run gave 0.518 raw but
             0.616 normalised with the SAME model.
  auc_macro  mean of the per-video AUCs (AI-VAD's "macro" AUC, 89.6% on ShanghaiTech; computed
             with AI-VAD's trick of adding one normal and one abnormal dummy frame per video)
  eer        equal error rate on the normalised score (lower is better)
  threshold_raw_at_eer   MSE threshold for single new videos (demo / API), where per-video
                         normalisation is impossible (it always reaches 1.0 in every video)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def psnr_norm(mse: np.ndarray) -> np.ndarray:
    """Per-video anomaly score in [0, 1] from PSNR (1 = most abnormal). Pixels are in [-1, 1]."""
    p = 10 * np.log10(4.0 / np.maximum(mse, 1e-10))
    lo, hi = np.nanmin(p), np.nanmax(p)
    return 1.0 - (p - lo) / max(hi - lo, 1e-8)


def eer_point(y, s) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / max(hi - lo, 1e-12)


def macro_auc(per_video) -> float:
    """AI-VAD evaluate.py macro_auc: per-video AUC with a dummy normal (score 0) and abnormal
    (score +max) frame added, so videos without anomalies still have an AUC; then the mean."""
    aucs = [roc_auc_score(np.concatenate(([0], y, [1])), np.concatenate(([0], s, [np.finfo(np.float64).max])))
            for s, y in per_video]
    return float(np.mean(aucs))


def frame_metrics(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], kind: str = "mse"):
    """scores: video -> score per frame (NaN where there is none, e.g. ASTT's first 4 frames).
    kind: "mse" (models 1-2, normalised via PSNR) or "score" (models 3-4, normalised directly).

    Returns (metrics dict, per_video {video: (mse, norm, labels)}). Frames without a prediction
    get the video's lowest score, like the repo (they still count, as normal-looking frames).
    """
    ys, raw, norm, per_video = [], [], [], {}
    for vid, mse in scores.items():
        valid = ~np.isnan(mse)
        if not valid.any():
            continue
        m = mse.copy()
        m[~valid] = np.nanmin(mse)
        n = psnr_norm(m) if kind == "mse" else minmax(m)
        ys.append(labels[vid]); raw.append(m); norm.append(n)
        per_video[vid] = (m, n, labels[vid])
    y, r, n = np.concatenate(ys), np.concatenate(raw), np.concatenate(norm)
    res = {"videos": len(per_video), "frames": int(len(y)), "abnormal_frames": int(y.sum())}
    if 0 < y.sum() < len(y):
        res["auc_raw"] = float(roc_auc_score(y, r))
        res["auc_norm"] = float(roc_auc_score(y, n))
        res["auc_macro"] = macro_auc([(p[0], p[2]) for p in per_video.values()])
        res["eer"], res["threshold_norm_at_eer"] = eer_point(y, n)
        res["eer_raw"], res["threshold_raw_at_eer"] = eer_point(y, r)
    else:
        res["note"] = "need both normal and abnormal frames for AUC"
    return res, per_video


def binary_metrics(y: np.ndarray, flags: np.ndarray) -> dict:
    """For yes/no detectors (the behavior rules): one operating point."""
    tpr = float((flags & (y == 1)).sum() / max((y == 1).sum(), 1))
    fpr = float((flags & (y == 0)).sum() / max((y == 0).sum(), 1))
    out = {"frames": int(len(y)), "abnormal_frames": int(y.sum()), "tpr": tpr, "fpr": fpr,
           "eer": (fpr + 1 - tpr) / 2}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, flags.astype(float)))
    return out
