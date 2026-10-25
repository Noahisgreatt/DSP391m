"""COUNTER-FLOW: a person moving against the people around them, or much faster than them.

For every tracked person in every frame, look at the k nearest moving people (same frame):
  against crowd   their direction is more than `angle_deg` away from where that crowd goes.
                  Nobody usable nearby? Compare with the camera's normal flow at that spot
                  (learned by optical_flow.py), but only where that flow is one-way.
  too fast        >= `fast_bh_per_s` body-heights per second AND >= `fast_ratio` x the
                  neighbours' speed (running, cycling, skating).
Both must last `min_seconds` for the same person. Body heights make it work at any distance.
"""
from __future__ import annotations

import numpy as np


def sustain(flag_rows: np.ndarray, idx: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only runs of >= min_run consecutive True rows of one track (idx sorted by time)."""
    out = np.zeros(len(idx), bool)
    run = 0
    for k, i in enumerate(idx):
        run = run + 1 if flag_rows[i] else 0
        if run >= min_run:
            out[k - run + 1:k + 1] = True
    return out


def crowd_reference(tracks, vel, moving, box_h, c):
    """Per row: mean direction (unit), agreement 0..1, median speed (body-heights/frame) and
    number of the k nearest moving neighbours in the same frame."""
    n = len(tracks)
    unit_c, cons = np.zeros((n, 2), np.float32), np.zeros(n, np.float32)
    spd_c, n_nb = np.full(n, np.nan, np.float32), np.zeros(n, np.int32)
    cx, cy = (tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2
    speed_bh = np.linalg.norm(vel, axis=1) / np.maximum(box_h, 1)
    unit = vel / np.maximum(np.linalg.norm(vel, axis=1), 1e-6)[:, None]
    frames = tracks[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(frames[order])) + 1):
        if len(rows) < 2:
            continue
        pts = np.stack([cx[rows], cy[rows]], 1)
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        reach = c["crowd_radius_boxes"] * box_h[rows][:, None]
        for a, r in enumerate(rows):
            ok = moving[rows] & (dist[a] <= reach[a]) & (np.arange(len(rows)) != a)
            nb = rows[ok]
            if len(nb) == 0:
                continue
            nb = nb[np.argsort(dist[a][ok])][: c["crowd_k"]]
            m = unit[nb].mean(0)
            unit_c[r], cons[r] = m / max(float(np.linalg.norm(m)), 1e-6), float(np.linalg.norm(m))
            spd_c[r], n_nb[r] = float(np.median(speed_bh[nb])), len(nb)
    return unit_c, cons, spd_c, n_nb


def detect_counter_flow(tracks, vel, moving, normal, consistency, size, fps, c):
    """-> (against_crowd (N,) bool, too_fast (N,) bool), sustained for min_seconds."""
    n = len(tracks)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    W, H = size
    R, C = consistency.shape
    cx, cy = (tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2
    box_h = tracks[:, 5] - tracks[:, 3]
    col = np.clip((cx * C / W).astype(int), 0, C - 1)
    row = np.clip((cy * R / H).astype(int), 0, R - 1)
    speed = np.linalg.norm(vel, axis=1)
    unit = vel / np.maximum(speed, 1e-6)[:, None]
    cos_lim = np.cos(np.deg2rad(c["angle_deg"]))

    unit_c, cons_c, spd_c, n_nb = crowd_reference(tracks, vel, moving, box_h, c)
    has_crowd = (n_nb >= c["min_crowd"]) & (cons_c >= c["crowd_consistency"])
    against_crowd = has_crowd & ((unit * unit_c).sum(1) <= cos_lim)
    against_scene = (~has_crowd & (consistency[row, col] >= c["min_flow_consistency"])
                     & ((unit * normal[row, col]).sum(1) <= cos_lim))
    against = moving & (against_crowd | against_scene)

    speed_bhs = speed / np.maximum(box_h, 1) * fps
    crowd_bhs = spd_c * fps
    fast = moving & (speed_bhs >= c["fast_bh_per_s"]) & (
        np.isnan(crowd_bhs) | (speed_bhs >= c["fast_ratio"] * np.nan_to_num(crowd_bhs)))

    min_run = max(1, int(round(c["min_seconds"] * fps)))
    out_a, out_f = np.zeros(n, bool), np.zeros(n, bool)
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        out_a[idx[sustain(against, idx, min_run)]] = True
        out_f[idx[sustain(fast, idx, min_run)]] = True
    return out_a, out_f
