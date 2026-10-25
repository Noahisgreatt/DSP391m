"""Step 5, BEHAVIOR: who is LOITERING, who moves AGAINST the crowd (counter-flow), who is much
FASTER than the crowd. No training: people tracking + optical flow + rules.

    python -m src.behavior track  --dataset shanghaitech   # people + tracks   (DVC: tracking_*)
    python -m src.behavior flow   --dataset shanghaitech   # normal directions (DVC: optical_flow_*)
    python -m src.behavior detect --dataset shanghaitech   # the events        (DVC: behavior_*)
(--dataset manual for our phone videos; ShanghaiTech: tracking on the test videos, flow on the
normal training videos.) Thresholds: configs/behavior.yaml.

Out: data/features/tracking/<dataset>/<video>.npz        boxes, track ids, velocities
     data/features/normal_maps/<dataset>/<scene>.npz     each camera's normal walking direction
     outputs/behavior/<dataset>/events/<video>.json|npz  events with times / flags per person
     outputs/behavior/<dataset>/metrics.json             counts (+ frame AUC on ShanghaiTech)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.utils import features_path, frames_dir, list_frames, list_videos, load_config, read_frame, video_fps

LOITER, COUNTER, FAST = 1, 2, 4
EVENTS = {LOITER: ("LOITERING", "stays in one place"), COUNTER: ("COUNTER_FLOW", "moving against the crowd"),
          FAST: ("TOO_FAST", "much faster than the crowd")}
DIRECTIONS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]


# ====================================================================== 1. people and tracks
def iou(a, b):
    """(n, 4) x (m, 4) -> (n, m) intersection over union."""
    tl, br = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


class IoUTracker:
    """Our tracker: each track's box is moved by its last velocity, then detections are matched
    to tracks by box overlap (Hungarian algorithm), confident detections first (the ByteTrack
    idea). Unmatched confident detections start new tracks; tracks unseen for max_lost frames end."""

    def __init__(self, match_iou=0.3, max_lost=30, high=0.6):
        self.match_iou, self.max_lost, self.high = match_iou, max_lost, high
        self.reset()

    def reset(self):
        self.tracks, self.next_id = {}, 1                 # id -> {box, vel, lost}

    def update(self, boxes, scores) -> np.ndarray:
        ids = -np.ones(len(boxes), np.int64)
        free = list(self.tracks)
        for stage in (scores >= self.high, scores < self.high):
            dets = np.flatnonzero(stage)
            if len(dets) and free:
                pred = np.array([self.tracks[t]["box"] + self.tracks[t]["vel"] for t in free])
                m = iou(boxes[dets], pred)
                rows, cols = linear_sum_assignment(-m)
                for r, c in zip(rows, cols):
                    if m[r, c] >= self.match_iou:
                        t = free[c]
                        tr = self.tracks[t]
                        tr["vel"] = 0.5 * tr["vel"] + 0.5 * (boxes[dets[r]] - tr["box"])
                        tr["box"], tr["lost"], ids[dets[r]] = boxes[dets[r]].copy(), 0, t
                free = [t for t in free if t not in set(ids)]
        for d in np.flatnonzero((ids < 0) & (scores >= self.high)):
            self.tracks[self.next_id] = {"box": boxes[d].copy(), "vel": np.zeros(4), "lost": 0}
            ids[d], self.next_id = self.next_id, self.next_id + 1
        for t in free:
            self.tracks[t]["lost"] += 1
            if self.tracks[t]["lost"] > self.max_lost:
                del self.tracks[t]
        return ids


def track_velocities(tracks, window=8, smooth=5, min_speed=0.4):
    """Per box: (centre[t] - centre[t - window]) / frames, after a moving average over `smooth`
    frames (so arm / leg swing doesn't flip the direction). -> velocity (n, 2), moving (n,)."""
    vel = np.zeros((len(tracks), 2), np.float32)
    if len(tracks) == 0:
        return vel, np.zeros(0, bool)
    centres = np.stack([(tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2], 1)
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        f, c = tracks[idx, 0], centres[idx]
        if smooth > 1 and len(c) > 1:
            k = min(smooth, len(c))
            pad = np.pad(c, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
            c = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(2)], 1)
        for j in range(1, len(idx)):
            i0 = max(0, j - window)
            if f[j] > f[i0]:
                vel[idx[j]] = (c[j] - c[i0]) / (f[j] - f[i0])
        if len(idx) > 1:
            vel[idx[0]] = vel[idx[1]]
    return vel, np.linalg.norm(vel, axis=1) > min_speed


class PeopleTracker:
    """Keypoint R-CNN (our network, COCO weights) person boxes + IoUTracker, per video."""

    def __init__(self, cfg, device=None, weights=True):
        from src.models.nets import RCNN, to_rgb_tensor
        t = cfg["tracking"]
        self.cfg, self.to_rgb = cfg, to_rgb_tensor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = RCNN(2, 17, min_size=t["detector_size"], score_thresh=t["detector_conf"])
        if weights:
            self.net.load_pretrained("keypointrcnn")
        self.net = self.net.to(self.device).eval()

    def video(self, frame_paths) -> dict:
        t, (W, H) = self.cfg["tracking"], self.cfg["frame_size"]
        tracker, rows = IoUTracker(t["match_iou"], t["max_lost"]), []
        for k, p in enumerate(frame_paths):
            img = cv2.imread(str(p))
            det = self.net([self.to_rgb(img, self.device)])[0]
            boxes, scores = det["boxes"].cpu().numpy(), det["scores"].cpu().numpy()
            ids = tracker.update(boxes, scores)
            sx, sy = W / img.shape[1], H / img.shape[0]       # store boxes in frame_size coords
            for b, i, s in zip(boxes, ids, scores):
                if i > 0:
                    rows.append([k, i, b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy, s])
        tracks = np.array(rows, np.float32).reshape(-1, 7)
        vel, moving = track_velocities(tracks, t["window"], t["smooth"], t["min_speed"])
        return {"tracks": tracks, "velocity": vel, "moving": moving,
                "frame_names": np.array([p.name for p in frame_paths])}


# ====================================================================== 2. each camera's normal direction
def angle_to_bin(ang, n_bins):
    w = 2 * np.pi / n_bins
    return (np.floor((ang + w / 2) / w).astype(np.int64)) % n_bins


def grid_features(flow, grid=(8, 8), n_bins=8, min_magnitude=1.0, min_moving_fraction=0.02):
    """Dense flow -> per cell: mean motion of the moving pixels, and its main direction."""
    H, W = flow.shape[:2]
    R, C = grid
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    cell = np.minimum((np.arange(H) * R) // H, R - 1)[:, None] * C + np.minimum((np.arange(W) * C) // W, C - 1)[None]
    m = (mag > min_magnitude).ravel()
    cm, bins, w = cell.ravel()[m], angle_to_bin(ang, n_bins).ravel()[m], mag.ravel()[m]
    hof = np.bincount(cm * n_bins + bins, weights=w, minlength=R * C * n_bins).astype(np.float64).reshape(R, C, n_bins)
    n_mov = np.bincount(cm, minlength=R * C)
    moving = (n_mov / np.maximum(np.bincount(cell.ravel(), minlength=R * C), 1)).reshape(R, C)
    mean = (np.stack([np.bincount(cm, weights=flow[..., j].ravel()[m], minlength=R * C) for j in range(2)], -1)
            / np.maximum(n_mov, 1)[:, None]).reshape(R, C, 2)
    dominant = hof.argmax(-1)
    dominant[moving < min_moving_fraction] = -1
    mean[moving < min_moving_fraction] = 0
    return mean.astype(np.float32), dominant.astype(np.int8)


def flow_video(vid, frame_dir, out, cfg):
    cv2.setNumThreads(1)
    o, fb = cfg["optical_flow"], cfg["optical_flow"]["farneback"]
    frames, means, doms = list_frames(frame_dir), [], []
    prev = cv2.cvtColor(read_frame(frames[0], cfg["frame_size"]), cv2.COLOR_BGR2GRAY)
    for p in frames[1:]:
        gray = cv2.cvtColor(read_frame(p, cfg["frame_size"]), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, fb["pyr_scale"], fb["levels"], fb["winsize"],
                                            fb["iterations"], fb["poly_n"], fb["poly_sigma"], 0)
        mean, dom = grid_features(flow, tuple(o["grid"]), o["n_bins"], o["min_magnitude"], o["min_moving_fraction"])
        means.append(mean); doms.append(dom)
        prev = gray
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mean=np.stack(means).astype(np.float16), dominant=np.stack(doms))
    return vid


def normal_map(flow_files, min_samples):
    """Average unit direction per cell -> normal (R,C,2), consistency (R,C): ~1 = everyone goes
    the same way there, ~0 = mixed or two-way traffic (then the rule doesn't use that cell)."""
    acc = cnt = None
    for p in flow_files:
        d = np.load(p)
        mean, valid = d["mean"].astype(np.float32), d["dominant"] >= 0
        speed = np.linalg.norm(mean, axis=-1, keepdims=True)
        unit = np.divide(mean, speed, out=np.zeros_like(mean), where=speed > 0)
        s, n = (unit * valid[..., None]).sum(0), valid.sum(0).astype(np.float64)
        acc, cnt = (s, n) if acc is None else (acc + s, cnt + n)
    avg = acc / np.maximum(cnt, 1)[..., None]
    cons = np.linalg.norm(avg, axis=-1)
    normal = np.divide(avg, cons[..., None], out=np.zeros_like(avg), where=cons[..., None] > 0)
    cons[cnt < min_samples] = 0
    return normal.astype(np.float32), cons.astype(np.float32)


def scene_of(dataset, vid):                     # ShanghaiTech: camera 01..13; manual: each video
    return vid.split("_")[0] if dataset == "shanghaitech" else vid


# ====================================================================== 3. the rules
def _sustain(flag, idx, min_run):
    """Keep runs of >= min_run consecutive flagged rows of one track (idx sorted by time)."""
    out, run = np.zeros(len(idx), bool), 0
    for k, i in enumerate(idx):
        run = run + 1 if flag[i] else 0
        if run >= min_run:
            out[k - run + 1:k + 1] = True
    return out


def counter_flow(tracks, vel, moving, normal, cons, size, fps, c):
    """COUNTER_FLOW: direction > angle_deg from where the k nearest moving people go (no usable
    crowd nearby: from the camera's normal flow, where that flow is one-way).
    TOO_FAST: >= fast_bh_per_s body heights per second and >= fast_ratio x the neighbours.
    Both must last min_seconds. Body heights make it work at any distance from the camera."""
    n = len(tracks)
    against, fast = np.zeros(n, bool), np.zeros(n, bool)
    if n == 0:
        return against, fast
    W, H = size
    R, C = cons.shape
    cx, cy = (tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2
    bh = tracks[:, 5] - tracks[:, 3]
    speed = np.linalg.norm(vel, axis=1)
    unit = vel / np.maximum(speed, 1e-6)[:, None]
    crowd_u, crowd_c = np.zeros((n, 2), np.float32), np.zeros(n, np.float32)
    crowd_s, n_nb = np.full(n, np.nan, np.float32), np.zeros(n, np.int32)
    frames = tracks[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(frames[order])) + 1):     # people of one frame
        if len(rows) < 2:
            continue
        pts = np.stack([cx[rows], cy[rows]], 1)
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        for a, r in enumerate(rows):
            ok = moving[rows] & (dist[a] <= c["crowd_radius_boxes"] * bh[r]) & (np.arange(len(rows)) != a)
            nb = rows[ok][np.argsort(dist[a][ok])][: c["crowd_k"]]
            if len(nb):
                m = unit[nb].mean(0)
                crowd_c[r] = np.linalg.norm(m)
                crowd_u[r] = m / max(crowd_c[r], 1e-6)
                crowd_s[r], n_nb[r] = np.median(speed[nb] / np.maximum(bh[nb], 1)), len(nb)
    cos_lim = np.cos(np.deg2rad(c["angle_deg"]))
    col, row = np.clip((cx * C / W).astype(int), 0, C - 1), np.clip((cy * R / H).astype(int), 0, R - 1)
    has_crowd = (n_nb >= c["min_crowd"]) & (crowd_c >= c["crowd_consistency"])
    vs_crowd = has_crowd & ((unit * crowd_u).sum(1) <= cos_lim)
    vs_scene = ~has_crowd & (cons[row, col] >= c["min_flow_consistency"]) & ((unit * normal[row, col]).sum(1) <= cos_lim)
    a_raw = moving & (vs_crowd | vs_scene)
    bhs, crowd_bhs = speed / np.maximum(bh, 1) * fps, crowd_s * fps
    f_raw = moving & (bhs >= c["fast_bh_per_s"]) & (np.isnan(crowd_bhs) | (bhs >= c["fast_ratio"] * np.nan_to_num(crowd_bhs)))
    run = max(1, int(round(c["min_seconds"] * fps)))
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        against[idx[_sustain(a_raw, idx, run)]] = True
        fast[idx[_sustain(f_raw, idx, run)]] = True
    return against, fast


def loitering(tracks, fps, c):
    """LOITERING at frame t: over the last `seconds`, every box centre stays within radius_boxes
    body heights of their average, and the person was visible >= min_coverage of that time."""
    out = np.zeros(len(tracks), bool)
    cx, cy = (tracks[:, 2] + tracks[:, 4]) / 2, (tracks[:, 3] + tracks[:, 5]) / 2
    win = c["seconds"] * fps
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        f, pts = tracks[idx, 0], np.stack([cx[idx], cy[idx]], 1)
        radius, start = c["radius_boxes"] * float(np.median(tracks[idx, 5] - tracks[idx, 3])), 0
        for k in range(len(idx)):
            if f[k] - f[0] < win:
                continue
            while f[k] - f[start] > win:
                start += 1
            w = pts[start:k + 1]
            if len(w) >= c["min_coverage"] * win and np.linalg.norm(w - w.mean(0), axis=1).max() <= radius:
                out[idx[k]] = True
    return out


def detect_events(cfg, tracks, vel, moving, normal, cons, fps):
    """flags (n,) per tracking row: 1 LOITERING | 2 COUNTER_FLOW | 4 TOO_FAST."""
    flags = np.zeros(len(tracks), np.uint8)
    if len(tracks):
        against, fast = counter_flow(tracks, vel, moving, normal, cons, tuple(cfg["frame_size"]), fps, cfg["counter_flow"])
        flags[against] |= COUNTER
        flags[fast] |= FAST
        flags[loitering(tracks, fps, cfg["loitering"])] |= LOITER
    return flags


def summarize(tracks, flags, fps):
    """Per-row flags -> one event per person and type, with start / end times."""
    events, gap = [], int(fps * 0.5)
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        for bit, (name, reason) in EVENTS.items():
            fr = tracks[idx[(flags[idx] & bit) > 0], 0].astype(int)
            if not len(fr):
                continue
            start = prev = fr[0]
            for x in list(fr[1:]) + [None]:
                if x is None or x - prev > gap:
                    events.append({"track_id": int(tid), "event": name, "reason": reason, "start_frame": int(start),
                                   "end_frame": int(prev), "start_s": round(start / fps, 2), "end_s": round(prev / fps, 2)})
                    start = x
                if x is not None:
                    prev = x
    return sorted(events, key=lambda e: e["start_frame"])


# ====================================================================== command line (DVC stages)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["track", "flow", "detect"])
    ap.add_argument("--dataset", required=True, choices=["shanghaitech", "manual"])
    ap.add_argument("--limit", type=int, help="only N videos (quick test)")
    a = ap.parse_args()
    cfg = load_config("configs/behavior.yaml")
    vids = list_videos(frames_dir(cfg, a.dataset))
    if a.dataset == "shanghaitech":
        from src.data import load_split
        split = load_split(cfg)
        keep = set(split["train"]) if a.step == "flow" else set(split["test"]) | set(split["val"])
        vids = [(v, d) for v, d in vids if v in keep]
    vids = vids[: a.limit]
    t0 = time.time()

    if a.step == "track":                                    # test videos: who is where
        todo = [(v, d) for v, d in vids if not features_path(cfg, "tracking", a.dataset, v).exists()]
        features_path(cfg, "tracking", a.dataset, "x").parent.mkdir(parents=True, exist_ok=True)
        tracker = PeopleTracker(cfg) if todo else None
        for k, (v, d) in enumerate(todo, 1):
            r = tracker.video(list_frames(d))
            np.savez_compressed(features_path(cfg, "tracking", a.dataset, v), **r)
            print(f"[{k}/{len(todo)}] {v}: {len(r['frame_names'])} frames, "
                  f"{len(np.unique(r['tracks'][:, 1]))} people ({time.time() - t0:.0f}s)")

    elif a.step == "flow":                                   # normal videos: usual directions
        todo = [(v, d) for v, d in vids if not features_path(cfg, "optical_flow", a.dataset, v).exists()]
        with ProcessPoolExecutor(max_workers=cfg["run"]["workers"]) as pool:
            jobs = [pool.submit(flow_video, v, d, features_path(cfg, "optical_flow", a.dataset, v), cfg) for v, d in todo]
            for k, j in enumerate(as_completed(jobs), 1):
                print(f"[{k}/{len(todo)}] {j.result()} ({time.time() - t0:.0f}s)")
        scenes = defaultdict(list)
        for v, _ in vids:
            scenes[scene_of(a.dataset, v)].append(features_path(cfg, "optical_flow", a.dataset, v))
        for sc, files in scenes.items():
            normal, cons = normal_map(files, cfg["optical_flow"]["min_cell_samples"])
            p = features_path(cfg, "normal_maps", a.dataset, sc)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, normal=normal, consistency=cons)
        print(f"{len(scenes)} normal direction maps")

    else:                                                    # the events
        out = Path(cfg["paths"]["outputs"]) / "behavior" / a.dataset
        (out / "events").mkdir(parents=True, exist_ok=True)
        R, C = cfg["optical_flow"]["grid"]
        counts, ys, ff = {n: 0 for n, _ in EVENTS.values()}, [], []
        for f in sorted(features_path(cfg, "tracking", a.dataset, "x").parent.glob("*.npz")):
            vid, tr = f.stem, np.load(f)
            mp = features_path(cfg, "normal_maps", a.dataset, scene_of(a.dataset, vid))
            normal, cons = ((np.load(mp)["normal"], np.load(mp)["consistency"]) if mp.exists()
                            else (np.zeros((R, C, 2), np.float32), np.zeros((R, C), np.float32)))
            fps = video_fps(cfg, a.dataset, vid)
            flags = detect_events(cfg, tr["tracks"], tr["velocity"], tr["moving"], normal, cons, fps)
            frame_flags = np.zeros(len(tr["frame_names"]), np.uint8)
            np.bitwise_or.at(frame_flags, tr["tracks"][:, 0].astype(int), flags)
            events = summarize(tr["tracks"], flags, fps)
            np.savez_compressed(out / "events" / f"{vid}.npz", flags=flags, frame_flags=frame_flags)
            (out / "events" / f"{vid}.json").write_text(json.dumps({"video": vid, "fps": fps, "events": events}, indent=2))
            for e in events:
                counts[e["event"]] += 1
            if a.dataset == "shanghaitech":
                from src.data import video_labels
                ys.append(video_labels(cfg, vid, len(frame_flags)))
                ff.append(frame_flags > 0)
        metrics = {"videos": len(list((out / "events").glob("*.json"))), "events": counts}
        if ys:
            from src.evaluate import binary_metrics
            metrics["frame_level"] = binary_metrics(np.concatenate(ys), np.concatenate(ff))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
