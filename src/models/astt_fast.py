"""Faster scoring of models 2, 5, 6 for the API and the app (the evaluation keeps the reference
code, astt.score_video). Same numbers, less work: predicting frame t needs the ViT tokens of
frames t-4 .. t-1, so astt.score_video computes every frame's ViT 4 times (once per window).
Here each frame goes through the ViT ONCE and its tokens are reused by the 4 windows: ~3-4x
less compute, which matters most on a CPU. tests/test_models.py checks the scores are equal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def predict_from_tokens(model, tok, last):
    """ASTTImproved.forward after its ViT: tok (b, T, n, d) ViT tokens of the T past frames,
    last (b, 3, S, S) the most recent frame -> predicted next frame (b, 3, 256, 256)."""
    b, t, n, d = tok.shape
    seq = tok.permute(0, 2, 1, 3).reshape(b * n, t, d)
    with torch.autocast(tok.device.type, enabled=False):           # as in forward(): fp32 attention
        seq = torch.cat([model.cls_token.expand(b * n, -1, -1), seq.float()], 1)
        tok = model.temporal_transformer(seq)[:, 0].view(b, n, d)
        tok = tok + model.cross_att(tok)
    fmap = tok[:, 1:].transpose(1, 2).reshape(b, d, model.grid, model.grid)
    out = model.decoder(model.to_map(F.interpolate(fmap, size=(16, 16), mode="bilinear", align_corners=False)))
    if model.residual:
        last = F.interpolate(last.float(), size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        out = (last + 2 * out).clamp(-1, 1)
    return out


@torch.no_grad()
def score_video_cached(model, cfg, video_dir, device, maps=False):
    """Same output as astt.score_video (mse per frame, NaN for the first T; error maps if maps)."""
    from src.dataset.loaders import VideoWindows
    from src.models.astt import autocast
    m = cfg["model"]
    T, B = m["num_frames"], cfg["eval"]["batch_size"]
    ds = VideoWindows([Path(video_dir)], T, m["image_size"], m["output_size"], cfg["input"]["color"] == "rgb")
    paths = ds.frames[0]
    n = len(paths)
    mse = np.full(n, np.nan, np.float32)
    emaps = np.zeros((n, 256, 256), np.float16) if maps else None
    model.eval()
    cache = {}                                          # frame -> (ViT tokens, ViT-size image, 256 target)
    for t0 in range(T, n, B):
        ts = list(range(t0, min(t0 + B, n)))
        new = [i for i in range(t0 - T, ts[-1] + 1) if i not in cache]
        loaded = [ds._load(paths[i]) for i in new]
        x = torch.from_numpy(np.stack([a for a, _ in loaded])).to(device)
        with autocast(cfg, device):
            tok = model.spatial_transformer.forward_features(x.float())
        for j, i in enumerate(new):
            cache[i] = (tok[j], x[j], torch.from_numpy(loaded[j][1]).to(device))
        tok = torch.stack([torch.stack([cache[i][0] for i in range(t - T, t)]) for t in ts])
        last = torch.stack([cache[t - 1][1] for t in ts])
        target = torch.stack([cache[t][2] for t in ts]).float()
        with autocast(cfg, device):
            pred = predict_from_tokens(model, tok, last)
        err = (pred.float() - target) ** 2
        mse[ts[0]:ts[-1] + 1] = err.mean(dim=(1, 2, 3)).cpu().numpy()
        if maps:
            emaps[ts[0]:ts[-1] + 1] = err.mean(1).cpu().numpy()
        for i in [k for k in cache if k < ts[-1] + 1 - T]:          # no longer needed by any window
            del cache[i]
    return (mse, emaps) if maps else mse


class CachedScorer:
    """Wraps astt.Scorer (models 2, 5, 6): same model, same smoothing, scored with the cache."""
    kind = "mse"

    def __init__(self, scorer):
        self.inner, self.model, self.cfg, self.device = scorer, scorer.model, scorer.cfg, scorer.device

    def __call__(self, video_dir, vid=None, details=False, **_):
        from src.models.astt import smooth
        if details:
            mse, emaps = score_video_cached(self.model, self.cfg, video_dir, self.device, True)
            return smooth(mse, self.inner.sigma), {"maps": emaps}
        return smooth(score_video_cached(self.model, self.cfg, video_dir, self.device), self.inner.sigma)
