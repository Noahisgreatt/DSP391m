"""LOITERING: a person who stays inside a small area for a long time.

A person is loitering at frame t if, over the last `seconds`, every position of their box centre
stays within `radius_boxes` x their body height of the average position, and they were visible
for at least `min_coverage` of that window.
"""
from __future__ import annotations

import numpy as np


def detect_loitering(tracks: np.ndarray, fps: float, c: dict) -> np.ndarray:
    """-> loitering (N,) bool, one per tracking row."""
    out = np.zeros(len(tracks), bool)
    if len(tracks) == 0:
        return out
    cx, cy = (tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2
    box_h = tracks[:, 5] - tracks[:, 3]
    window = c["seconds"] * fps
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        f = tracks[idx, 0]
        pts = np.stack([cx[idx], cy[idx]], 1)
        radius = c["radius_boxes"] * float(np.median(box_h[idx]))
        start = 0
        for k in range(len(idx)):
            if f[k] - f[0] < window:
                continue
            while f[k] - f[start] > window:
                start += 1
            win = pts[start:k + 1]
            if len(win) >= c["min_coverage"] * window and \
                    np.linalg.norm(win - win.mean(0), axis=1).max() <= radius:
                out[idx[k]] = True
    return out
