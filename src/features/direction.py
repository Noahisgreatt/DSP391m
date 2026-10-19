"""Turn a dense flow field into per-cell direction features.

For a flow field of shape (H, W, 2) and a grid of R x C cells, this produces:
  hof      (R, C, n_bins)  Histogram of Oriented Flow, weighted by speed, normalised per cell
  mean     (R, C, 2)       mean (dx, dy) of the moving pixels in each cell, px/frame
  moving   (R, C)          share of pixels in the cell that move faster than min_magnitude
  dominant (R, C)          index of the strongest bin, or -1 if the cell is not moving

Image coordinates: x to the right, y DOWN. So bin 0 = right, 2 = down, 4 = left, 6 = up.
"""
from __future__ import annotations

import cv2
import numpy as np

DIRECTION_NAMES_8 = ["right", "down-right", "down", "down-left",
                     "left", "up-left", "up", "up-right"]


def angle_to_bin(ang: np.ndarray, n_bins: int) -> np.ndarray:
    """Radians -> bin index, with bin 0 centred on 'right' (0 rad)."""
    width = 2 * np.pi / n_bins
    return (np.floor((ang + width / 2) / width).astype(np.int64)) % n_bins


def flow_to_grid_features(flow: np.ndarray, grid=(8, 8), n_bins: int = 8,
                          min_magnitude: float = 1.0,
                          min_moving_fraction: float = 0.02) -> dict:
    H, W = flow.shape[:2]
    R, C = grid
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    moving_px = mag > min_magnitude
    bins = angle_to_bin(ang, n_bins)

    # cell index of every pixel
    rows = np.minimum((np.arange(H) * R) // H, R - 1)
    cols = np.minimum((np.arange(W) * C) // W, C - 1)
    cell = rows[:, None] * C + cols[None, :]               # (H, W)

    m = moving_px.ravel()
    cell_m = cell.ravel()[m]
    bins_m = bins.ravel()[m]
    w = mag.ravel()[m]

    # speed-weighted histogram per cell, in one bincount.
    # .astype(float64): with no moving pixels, bincount returns int64 instead of float.
    hof = np.bincount(cell_m * n_bins + bins_m, weights=w,
                      minlength=R * C * n_bins).astype(np.float64).reshape(R, C, n_bins)
    totals = hof.sum(-1, keepdims=True)
    hof = np.divide(hof, totals, out=np.zeros_like(hof), where=totals > 0)

    px_per_cell = np.bincount(cell.ravel(), minlength=R * C)
    n_moving = np.bincount(cell_m, minlength=R * C)
    moving = (n_moving / np.maximum(px_per_cell, 1)).reshape(R, C)

    sum_dx = np.bincount(cell_m, weights=flow[..., 0].ravel()[m],
                         minlength=R * C).astype(np.float64)
    sum_dy = np.bincount(cell_m, weights=flow[..., 1].ravel()[m],
                         minlength=R * C).astype(np.float64)
    mean = np.stack([sum_dx, sum_dy], -1) / np.maximum(n_moving, 1)[:, None]
    mean = mean.reshape(R, C, 2)

    dominant = hof.argmax(-1)
    dominant[moving < min_moving_fraction] = -1
    mean[moving < min_moving_fraction] = 0.0

    return {"hof": hof.astype(np.float32), "mean": mean.astype(np.float32),
            "moving": moving.astype(np.float32), "dominant": dominant.astype(np.int8)}


def track_velocities(tracks: np.ndarray, window: int = 8, smooth: int = 5,
                     min_speed: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """One smoothed velocity per tracked box.

    tracks: (N, 7) rows of frame_idx, track_id, x1, y1, x2, y2, conf.
    Returns velocity (N, 2) in px/frame and moving (N,) bool, aligned with `tracks` rows.
    Velocity at frame t = (centre[t] - centre[t - window]) / frames elapsed, after a
    moving-average over `smooth` frames, so leg and arm swing doesn't flip the direction.
    """
    vel = np.zeros((len(tracks), 2), dtype=np.float32)
    if len(tracks) == 0:
        return vel, np.zeros(0, dtype=bool)
    centres = np.stack([(tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2], 1)
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        f = tracks[idx, 0]
        c = centres[idx]
        if smooth > 1 and len(c) > 1:
            k = min(smooth, len(c))
            kernel = np.ones(k) / k
            pad = np.pad(c, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
            c = np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(2)], 1)
        for j in range(1, len(idx)):
            i0 = max(0, j - window)
            dt = f[j] - f[i0]
            if dt > 0:
                vel[idx[j]] = (c[j] - c[i0]) / dt
        if len(idx) > 1:
            vel[idx[0]] = vel[idx[1]]
    moving = np.linalg.norm(vel, axis=1) > min_speed
    return vel, moving


def summarize_video(dominant: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Most common direction per cell over a whole video, -1 where it rarely moves.

    dominant: (T, R, C) from flow_to_grid_features stacked over time.
    """
    T, R, C = dominant.shape
    out = np.full((R, C), -1, dtype=np.int8)
    for r in range(R):
        for c in range(C):
            d = dominant[:, r, c]
            d = d[d >= 0]
            if len(d) >= max(1, T // 20):
                out[r, c] = np.bincount(d, minlength=n_bins).argmax()
    return out
