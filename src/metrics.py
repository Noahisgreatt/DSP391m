"""Shared metric code: frame-level AUC / EER, per-video normalisation, mixed-precision helper.

Used by every model (thesis ASTT, CrowdFormer) so their numbers are computed the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.split import video_labels


def autocast(cfg, device):
    if device.type == "cuda" and cfg["train"].get("amp", True):
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dt)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()


def psnr_norm(mse: np.ndarray) -> np.ndarray:
    """Per-video min-max normalised anomaly score in [0, 1] from PSNR (1 = most abnormal)."""
    p = 10 * np.log10(4.0 / np.maximum(mse, 1e-10))        # pixel range [-1, 1] -> peak^2 = 4
    lo, hi = np.nanmin(p), np.nanmax(p)
    return 1.0 - (p - lo) / max(hi - lo, 1e-8)


def frame_metrics(cfg, scores: dict[str, np.ndarray], events_dir: Path | None = None) -> dict:
    ys, raw, norm, base = [], [], [], []
    per_video = {}
    for vid, mse in scores.items():
        valid = ~np.isnan(mse)
        if not valid.any():
            continue
        lab = video_labels(cfg, vid, len(mse))
        m = mse.copy()
        m[~valid] = np.nanmin(mse)
        n = psnr_norm(m)
        ys.append(lab); raw.append(m); norm.append(n)
        if events_dir is not None and (events_dir / f"{vid}.npz").exists():
            ff = np.load(events_dir / f"{vid}.npz")["frame_flags"]
            base.append(np.pad(ff, (0, max(0, len(m) - len(ff))))[: len(m)] > 0)
        per_video[vid] = (m, n, lab)
    y, r, n = np.concatenate(ys), np.concatenate(raw), np.concatenate(norm)
    res = {"frames": int(len(y)), "abnormal_frames": int(y.sum()), "videos": len(per_video)}
    if 0 < y.sum() < len(y):
        res["auc_raw"] = float(roc_auc_score(y, r))
        res["auc_norm"] = float(roc_auc_score(y, n))
        fpr, tpr, thr = roc_curve(y, n)
        i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
        res["eer"] = float((fpr[i] + 1 - tpr[i]) / 2)
        res["threshold_norm_at_eer"] = float(thr[i])
        # raw-MSE threshold (EER point): usable on a single new video, unlike the per-video
        # normalised score, which always reaches 1.0 in every video
        fpr, tpr, thr = roc_curve(y, r)
        i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
        res["eer_raw"] = float((fpr[i] + 1 - tpr[i]) / 2)
        res["threshold_raw_at_eer"] = float(thr[i])
        if base and len(base) == len(ys):
            res["auc_rule_baseline"] = float(roc_auc_score(y, np.concatenate(base)))
    else:
        res["note"] = "need both normal and abnormal frames for AUC"
    return res, per_video
