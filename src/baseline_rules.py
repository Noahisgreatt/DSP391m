"""Rule-based anomaly events per person: LOITERING and ABNORMAL (different from the crowd).

This is the BASELINE, not a trained model. It combines the two feature stages:
  1. Normal direction map: from the optical-flow features, the usual walking direction of
     every grid cell. ShanghaiTech: one map per camera scene, learned from all of that
     scene's videos. Manual clips: one map per video.
  2. Per person (from tracking), compared with the k nearest moving people in the same frame:
       ABNORMAL, reason "moving against the crowd": direction more than `counter_angle_deg`
                   away from where the nearby crowd goes (if nobody usable is nearby: away from
                   the scene's normal flow at that spot)
       ABNORMAL, reason "much faster than the crowd": speed >= `fast_bh_per_s` body-heights/s
                   and >= `fast_ratio` x the neighbours' speed (running, cycling, skating)
       Both must last `counter_min_seconds`.
       LOITERING = staying within `loiter_radius_boxes` x body height of one spot for
                   `loiter_seconds`.

Output per video:
  data/features/events/<dataset>/<video>.npz
      flags        (N,) uint8, aligned with the tracking rows. bit 1 = LOITERING, 2 = against crowd, 4 = fast
      frame_flags  (T,) uint8, OR of all people in each frame (for frame-level evaluation)
  data/features/events/<dataset>/<video>.json   human-readable list of events with times
  data/features/normal_maps/<dataset>/<scene>.npz

    python -m src.baseline_rules --dataset shanghai
    python -m src.baseline_rules --dataset manual
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import features_path, list_videos, load_config, video_fps

LOITER, AGAINST, FAST = 1, 2, 4
ABNORMAL = AGAINST | FAST
EVENT_NAMES = {LOITER: "LOITERING", AGAINST: "ABNORMAL", FAST: "ABNORMAL"}
REASONS = {LOITER: "stays in one place", AGAINST: "moving against the crowd",
           FAST: "much faster than the crowd"}


def scene_key(cfg: dict, dataset: str, video_id: str) -> str:
    mode = cfg["detection"]["scene_from"][dataset]
    return video_id.split("_")[0] if mode == "prefix" else video_id


def build_normal_map(flow_files, min_samples: int):
    """Average unit direction per cell over many frames.

    Returns normal (R, C, 2) unit vectors, consistency (R, C) in 0..1, samples (R, C).
    consistency ~1: everyone goes the same way. ~0: mixed or two-way traffic.
    """
    acc = cnt = None
    for p in flow_files:
        d = np.load(p)
        mean = d["mean"].astype(np.float32)
        valid = d["dominant"] >= 0
        speed = np.linalg.norm(mean, axis=-1, keepdims=True)
        unit = np.divide(mean, speed, out=np.zeros_like(mean), where=speed > 0)
        s = (unit * valid[..., None]).sum(0)
        n = valid.sum(0).astype(np.float64)
        acc, cnt = (s, n) if acc is None else (acc + s, cnt + n)
    avg = acc / np.maximum(cnt, 1)[..., None]
    consistency = np.linalg.norm(avg, axis=-1)
    normal = np.divide(avg, consistency[..., None], out=np.zeros_like(avg),
                       where=consistency[..., None] > 0)
    consistency[cnt < min_samples] = 0.0
    return normal.astype(np.float32), consistency.astype(np.float32), cnt


def _sustain(flag_rows: np.ndarray, idx: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only runs of >= min_run consecutive True rows of one track (idx sorted by time)."""
    out = np.zeros(len(idx), bool)
    run = 0
    for k, i in enumerate(idx):
        run = run + 1 if flag_rows[i] else 0
        if run >= min_run:
            out[k - run + 1:k + 1] = True
    return out


def crowd_reference(tracks, vel, moving, box_h, d):
    """For every row: the motion of the nearby crowd in the same frame.

    Returns crowd_unit (N, 2) mean direction of the k nearest moving neighbours,
    crowd_consistency (N,) 0..1 (1 = they all go the same way), crowd_speed (N,) their median
    speed in body-heights/frame, n_neighbours (N,).
    """
    n = len(tracks)
    unit_c = np.zeros((n, 2), np.float32)
    cons = np.zeros(n, np.float32)
    spd_c = np.full(n, np.nan, np.float32)
    n_nb = np.zeros(n, np.int32)
    cx = (tracks[:, 2] + tracks[:, 4]) / 2
    cy = (tracks[:, 3] + tracks[:, 5]) / 2
    speed_bh = np.linalg.norm(vel, axis=1) / np.maximum(box_h, 1)
    unit = vel / np.maximum(np.linalg.norm(vel, axis=1), 1e-6)[:, None]
    frames = tracks[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    bounds = np.flatnonzero(np.diff(frames[order])) + 1
    for rows in np.split(order, bounds):
        if len(rows) < 2:
            continue
        pts = np.stack([cx[rows], cy[rows]], 1)
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        reach = d["crowd_radius_boxes"] * box_h[rows][:, None]
        for a, r in enumerate(rows):
            ok = moving[rows] & (dist[a] <= reach[a]) & (np.arange(len(rows)) != a)
            nb = rows[ok]
            if len(nb) == 0:
                continue
            nb = nb[np.argsort(dist[a][ok])][: d["crowd_k"]]
            m = unit[nb].mean(0)
            c = float(np.linalg.norm(m))
            unit_c[r] = m / max(c, 1e-6)
            cons[r] = c
            spd_c[r] = float(np.median(speed_bh[nb]))
            n_nb[r] = len(nb)
    return unit_c, cons, spd_c, n_nb


def detect_events(tracks, vel, moving, normal, consistency, size, fps, d) -> np.ndarray:
    """flags (N,) for every tracking row: LOITER | AGAINST | FAST bits."""
    n = len(tracks)
    flags = np.zeros(n, np.uint8)
    if n == 0:
        return flags
    W, H = size
    R, C = consistency.shape
    cx = (tracks[:, 2] + tracks[:, 4]) / 2
    cy = (tracks[:, 3] + tracks[:, 5]) / 2
    box_h = tracks[:, 5] - tracks[:, 3]
    col = np.clip((cx * C / W).astype(int), 0, C - 1)
    row = np.clip((cy * R / H).astype(int), 0, R - 1)

    speed = np.linalg.norm(vel, axis=1)
    unit = vel / np.maximum(speed, 1e-6)[:, None]
    speed_bhs = speed / np.maximum(box_h, 1) * fps            # body-heights per second
    cos_lim = np.cos(np.deg2rad(d["counter_angle_deg"]))

    # 1) direction: against the nearby crowd; if no usable crowd, against the scene's flow
    unit_c, cons_c, spd_c, n_nb = crowd_reference(tracks, vel, moving, box_h, d)
    has_crowd = (n_nb >= d["min_crowd"]) & (cons_c >= d["crowd_consistency"])
    against_crowd = has_crowd & ((unit * unit_c).sum(1) <= cos_lim)
    against_scene = (~has_crowd & (consistency[row, col] >= d["min_flow_consistency"])
                     & ((unit * normal[row, col]).sum(1) <= cos_lim))
    against = moving & (against_crowd | against_scene)

    # 2) speed: much faster than the people around (running, cycling, skating)
    crowd_bhs = spd_c * fps
    fast = moving & (speed_bhs >= d["fast_bh_per_s"]) & (
        np.isnan(crowd_bhs) | (speed_bhs >= d["fast_ratio"] * np.nan_to_num(crowd_bhs)))

    min_run = max(1, int(round(d["counter_min_seconds"] * fps)))
    loiter_len = d["loiter_seconds"] * fps

    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        f = tracks[idx, 0]
        flags[idx[_sustain(against, idx, min_run)]] |= AGAINST
        flags[idx[_sustain(fast, idx, min_run)]] |= FAST

        # LOITERING: every position in the last `loiter_seconds` stays near one spot
        pts = np.stack([cx[idx], cy[idx]], 1)
        radius = d["loiter_radius_boxes"] * float(np.median(box_h[idx]))
        start = 0
        for k in range(len(idx)):
            if f[k] - f[0] < loiter_len:
                continue
            while f[k] - f[start] > loiter_len:
                start += 1
            win = pts[start:k + 1]
            if len(win) < d["min_coverage"] * loiter_len:
                continue
            if np.linalg.norm(win - win.mean(0), axis=1).max() <= radius:
                flags[idx[k]] |= LOITER
    return flags


def summarize(tracks, flags, fps: float, max_gap: int) -> list[dict]:
    """Turn per-row flags into event segments with start and end times."""
    events = []
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        for bit, name in EVENT_NAMES.items():
            frames = tracks[idx[(flags[idx] & bit) > 0], 0].astype(int)
            if len(frames) == 0:
                continue
            seg_start = prev = frames[0]
            for fr in list(frames[1:]) + [None]:
                if fr is None or fr - prev > max_gap:
                    events.append({"track_id": int(tid), "event": name, "reason": REASONS[bit],
                                   "start_frame": int(seg_start), "end_frame": int(prev),
                                   "start_s": round(float(seg_start) / fps, 2),
                                   "end_s": round(float(prev) / fps, 2)})
                    seg_start = fr
                if fr is not None:
                    prev = fr
    return sorted(events, key=lambda e: e["start_frame"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghai", "manual"])
    ap.add_argument("--config", help="YAML that overrides params.yaml")
    ap.add_argument("--video", help="only this video (the normal map still uses its whole scene)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["detection"]
    method = cfg["flow"]["method"]
    feat_root = Path(cfg["data"]["features_dir"])
    events_dir = feat_root / "events" / args.dataset
    maps_dir = feat_root / "normal_maps" / args.dataset
    events_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # videos that have both flow and tracking features
    ready = [v for v, _ in list_videos(cfg, args.dataset)
             if features_path(cfg, method, args.dataset, v).exists()
             and features_path(cfg, "tracking", args.dataset, v).exists()]
    if not ready:
        print("No videos with both flow and tracking features yet.")
        return

    by_scene = defaultdict(list)
    for v, _ in list_videos(cfg, args.dataset):
        fp = features_path(cfg, method, args.dataset, v)
        if fp.exists():
            by_scene[scene_key(cfg, args.dataset, v)].append(fp)

    targets = [args.video] if args.video else ready
    maps = {}
    for vid in targets:
        key = scene_key(cfg, args.dataset, vid)
        if key not in maps:
            maps[key] = build_normal_map(by_scene[key], d["min_cell_samples"])
            np.savez_compressed(maps_dir / f"{key}.npz", normal=maps[key][0],
                                consistency=maps[key][1], samples=maps[key][2])
        normal, consistency, _ = maps[key]

        tr = np.load(features_path(cfg, "tracking", args.dataset, vid))
        fl = np.load(features_path(cfg, method, args.dataset, vid))
        tracks = tr["tracks"]
        fps = video_fps(cfg, args.dataset, vid)
        flags = detect_events(tracks, tr["velocity"], tr["moving"], normal, consistency,
                              tuple(fl["size"]), fps, d)

        T = len(tr["frame_names"])
        frame_flags = np.zeros(T, np.uint8)
        np.bitwise_or.at(frame_flags, tracks[:, 0].astype(int), flags)
        events = summarize(tracks, flags, fps, max_gap=int(fps * 0.5))

        np.savez_compressed(events_dir / f"{vid}.npz", flags=flags, frame_flags=frame_flags)
        (events_dir / f"{vid}.json").write_text(json.dumps(
            {"video": vid, "scene": key, "fps": fps, "events": events}, indent=2))
        n_l = sum(e["event"] == "LOITERING" for e in events)
        n_a = sum(e["event"] == "ABNORMAL" for e in events)
        print(f"{vid}: {n_l} loitering, {n_a} abnormal (different from the crowd)")


if __name__ == "__main__":
    main()
