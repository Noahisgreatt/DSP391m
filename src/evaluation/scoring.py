"""Anomaly score of every frame of one video = error of the model's prediction of that frame.

Same procedure as the thesis repo's test_all_scenes: slide over the video, predict frame t from
frames t-4..t-1, score = MSE(predicted, real) at 256x256. The first 4 frames have no score (NaN).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from src.data.dataset import video_dataset


def autocast(cfg: dict, device):
    if device.type == "cuda" and cfg["train"].get("amp", False):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def score_video(model, cfg: dict, video_dir: Path, device, return_maps: bool = False,
                num_workers: int | None = None):
    """-> mse (n_frames,) [, error maps (n_frames, 256, 256) float16 for drawing]."""
    ds, n = video_dataset(cfg, Path(video_dir))
    dl = data.DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                         num_workers=cfg["train"]["num_workers"] if num_workers is None else num_workers)
    T = cfg["model"]["num_frames"]
    mse = np.full(n, np.nan, np.float32)
    maps = np.zeros((n, 256, 256), np.float16) if return_maps else None
    model.eval()
    t = T
    for b in dl:
        target, inp = b["256"][:, T].to(device).float(), b["standard"][:, :T].to(device)
        with autocast(cfg, device):
            pred = model(inp)
        err = (pred.float() - target) ** 2                       # (b, 3, 256, 256)
        k = len(err)
        mse[t:t + k] = err.mean(dim=(1, 2, 3)).cpu().numpy()     # = nn.MSELoss per frame
        if return_maps:
            maps[t:t + k] = err.mean(1).cpu().numpy()
        t += k
    return (mse, maps) if return_maps else mse


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Image-gradient difference loss (Liu et al., CVPR 2018): keeps predicted edges sharp."""
    def grads(x):
        return (x[..., :, 1:] - x[..., :, :-1]).abs(), (x[..., 1:, :] - x[..., :-1, :]).abs()
    (px, py), (tx, ty) = grads(pred), grads(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


class ASTTScorer:
    """Models 1 / 2: per-frame prediction error. `details=True` also returns error maps."""
    kind = "mse"

    def __init__(self, cfg: dict, checkpoint, device):
        from src.models import load_trained
        self.cfg, self.device = cfg, device
        self.model = load_trained(cfg, checkpoint, device)

    def __call__(self, video_dir, vid=None, details=False, num_workers=None, **_):
        if details:
            mse, maps = score_video(self.model, self.cfg, video_dir, self.device, True, num_workers)
            return mse, {"maps": maps}
        return score_video(self.model, self.cfg, video_dir, self.device, False, num_workers)


def build_scorer(cfg: dict, checkpoint, device, smoke=False):
    """The per-video scorer of any of the 4 models (config decides)."""
    if cfg["model"].get("family") == "aivad":
        from src.models.aivad.pipeline import VideoScorer
        return VideoScorer(cfg, checkpoint, device, smoke=smoke)
    return ASTTScorer(cfg, checkpoint, device)
