"""Stage 2b: YOLO + ByteTrack -> person tracks -> loitering / counter-flow pseudo-labels.

This is the notebook logic moved into a reusable module, with these fixes:
  * Loitering used "distance from the first position" -> anyone who walked in and then
    stopped never triggered, and anyone who drifted 50 px "reset". Now: sliding window over
    the last `window_sec`, radius relative to box height (works for near and far people).
  * Counter-flow used a raw dot product (`dot < -50`) -> depended on speed and image size.
    Now: cosine to the leave-one-out crowd mean, only when the crowd is coherent, and it must
    persist `min_frames` frames.
  * Scores and labels share one scale: score >= 0.5 <=> rule fires.

Outputs
  data/features/tracking/<dataset>/<video_id>.npz   tracks: frame, id, cx, cy, w, h, conf
  data/labels/tracking/<dataset>/<video_id>.npy      int8 per frame: 0 normal, 1 loitering, 2 counter-flow
  data/labels/tracking/<dataset>/<video_id>.score.npy float32 per frame 0..1

Usage:
    python -m src.features.tracking --dataset shanghaitech --config configs/features/tracking.yaml
    python -m src.features.tracking --dataset shanghaitech --labels-only   # re-run rules only
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.utils import list_frame_dirs, list_frames, load_params, load_yaml, resolve

NORMAL, LOITERING, COUNTERFLOW = 0, 1, 2
CLASS_NAMES = {NORMAL: "normal", LOITERING: "loitering", COUNTERFLOW: "counter-flow"}


# --------------------------------------------------------------------------- tracking
def track_video(model, frame_dir: Path, det: dict) -> dict[str, np.ndarray]:
    rows = []
    for fi, path in enumerate(list_frames(frame_dir)):
        frame = cv2.imread(str(path))
        res = model.track(frame, persist=True, tracker=det["tracker"], classes=det["classes"],
                          imgsz=det["imgsz"], conf=det["conf"], iou=det["iou"],
                          device=det.get("device"), verbose=False)[0]
        if res.boxes.id is None:
            continue
        xywh = res.boxes.xywh.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        for (x, y, w, h), i, c in zip(xywh, ids, conf):
            rows.append((fi, i, x, y, w, h, c))
    a = np.asarray(rows, np.float32).reshape(-1, 7)
    return {"frame": a[:, 0].astype(np.int32), "id": a[:, 1].astype(np.int32),
            "cx": a[:, 2], "cy": a[:, 3], "w": a[:, 4], "h": a[:, 5], "conf": a[:, 6],
            "n_frames": np.int32(len(list_frames(frame_dir)))}


# --------------------------------------------------------------------------- rules
def apply_rules(tracks: dict, n_frames: int, fps: float, rules: dict) -> dict[str, np.ndarray]:
    """Return per-detection and per-frame scores/labels.

    det_loiter, det_counter: (D,) scores in [0, 1] aligned with tracks rows.
    frame_score: (n_frames,) max over detections, median-smoothed.
    frame_label: (n_frames,) 0 / 1 / 2.
    """
    D = len(tracks["frame"])
    frame, tid = tracks["frame"], tracks["id"]
    cx, cy, h = tracks["cx"], tracks["cy"], np.maximum(tracks["h"], 1.0)
    lag = rules["velocity_lag"]
    lo, co = rules["loitering"], rules["counterflow"]
    win = max(1, int(round(lo["window_sec"] * fps)))
    min_age = int(round(lo["min_track_sec"] * fps))

    vel = np.zeros((D, 2), np.float32)       # box-heights / frame
    det_loiter = np.zeros(D, np.float32)
    det_counter = np.zeros(D, np.float32)

    by_track: dict[int, np.ndarray] = defaultdict(list)
    for k in np.argsort(frame, kind="stable"):
        by_track[int(tid[k])].append(k)

    for rows in by_track.values():
        rows = np.asarray(rows)
        f = frame[rows]
        pts = np.stack([cx[rows], cy[rows]], 1)
        start = f[0]
        for j, k in enumerate(rows):
            # velocity over `lag` frames (handles gaps via real frame difference)
            p = np.searchsorted(f, f[j] - lag)
            if p < j:
                vel[k] = (pts[j] - pts[p]) / max(f[j] - f[p], 1) / h[k]
            # loitering: max distance to current position within the sliding window
            age = f[j] - start
            if age >= min_age:
                w0 = np.searchsorted(f, f[j] - win)
                r = np.linalg.norm(pts[w0:j + 1] - pts[j], axis=1).max() / (lo["radius_box_heights"] * h[k])
                det_loiter[k] = np.clip(1.5 - r, 0, 1) * min(1.0, age / win)

    # counter-flow, frame by frame
    speed = np.linalg.norm(vel, axis=1)
    moving = speed >= rules["min_speed"]
    unit = vel / np.maximum(speed, 1e-6)[:, None]
    raw_cos = np.ones(D, np.float32)
    order = np.argsort(frame, kind="stable")
    bounds = np.flatnonzero(np.diff(frame[order])) + 1
    for grp in np.split(order, bounds):
        m = grp[moving[grp]]
        if len(m) < co["min_movers"]:
            continue
        total = unit[m].sum(0)
        for k in m:
            loo = total - unit[k]                      # leave-one-out crowd direction
            n = np.linalg.norm(loo)
            coherence = n / (len(m) - 1)
            if coherence < 0.3:                        # crowd has no dominant direction
                continue
            raw_cos[k] = float(unit[k] @ (loo / n))

    thr = co["cos_threshold"]
    for rows in by_track.values():                     # persistence per track
        run = 0
        for k in rows:
            run = run + 1 if raw_cos[k] < thr else 0
            det_counter[k] = np.clip(0.5 * raw_cos[k] / thr, 0, 1) * min(1.0, run / co["min_frames"]) \
                if raw_cos[k] < 0 else 0.0

    s_loiter = np.zeros(n_frames, np.float32)
    s_counter = np.zeros(n_frames, np.float32)
    np.maximum.at(s_loiter, frame, det_loiter)
    np.maximum.at(s_counter, frame, det_counter)
    k = rules["smooth_frames"]
    if k > 1:
        s_loiter = median_filter(s_loiter, size=k, mode="nearest")
        s_counter = median_filter(s_counter, size=k, mode="nearest")
    score = np.maximum(s_loiter, s_counter)
    label = np.where(score < rules["label_threshold"], NORMAL,
                     np.where(s_counter >= s_loiter, COUNTERFLOW, LOITERING)).astype(np.int8)
    return {"det_loiter": det_loiter, "det_counter": det_counter, "velocity": vel,
            "frame_score": score, "frame_label": label}


def dataset_fps(params: dict, dataset: str) -> float:
    return float(params["datasets"][dataset].get("fps") or 25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/features/tracking.yaml")
    ap.add_argument("--labels-only", action="store_true", help="Skip YOLO, re-apply rules to saved tracks.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    params = load_params()
    cfg = load_yaml(resolve(args.config))
    fps = dataset_fps(params, args.dataset)
    frame_root = resolve(params["paths"]["frames"]) / args.dataset
    feat_dir = resolve(params["paths"]["features"]) / cfg["method"] / args.dataset
    label_dir = resolve(params["paths"]["labels"]) / cfg["method"] / args.dataset
    feat_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    if not args.labels_only:
        from ultralytics import YOLO

        for d in tqdm([d for d in list_frame_dirs(frame_root) if not d.name.startswith("_")], desc="track"):
            out = feat_dir / f"{d.name}.npz"
            if out.exists() and not args.overwrite:
                continue
            model = YOLO(cfg["detector"]["weights"])     # fresh tracker state per video
            np.savez(out, **track_video(model, d, cfg["detector"]))

    for f in tqdm(sorted(feat_dir.glob("*.npz")), desc="rules"):
        t = dict(np.load(f))
        r = apply_rules(t, int(t["n_frames"]), fps, cfg["rules"])
        np.save(label_dir / f"{f.stem}.npy", r["frame_label"])
        np.save(label_dir / f"{f.stem}.score.npy", r["frame_score"])
    print(f"Tracks -> {feat_dir}\nLabels -> {label_dir}")


if __name__ == "__main__":
    main()
