"""Find and follow every person (YOLO + ByteTrack) and measure how each one moves.

    python -m src.behavior.tracking --dataset shanghaitech     (DVC stage: tracking@shanghaitech)
    python -m src.behavior.tracking --dataset manual           (DVC stage: tracking@manual)
    python -m src.behavior.tracking --dataset shanghaitech --limit 2   # quick test

ShanghaiTech: only the 107 TEST videos are tracked (that is where behavior is judged).
Out: data/features/tracking/<dataset>/<video>.npz
    tracks      (N, 7)  frame, track_id, x1, y1, x2, y2, confidence  (boxes in frame_size coords)
    velocity    (N, 2)  smoothed box-centre movement, px/frame
    moving      (N,)    speed > min_speed
    frame_names (T,)
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.config import features_path, frames_dir, load_config
from src.data.frames import list_frames, list_videos
from src.data.split import load_split

PERSON = 0  # COCO class id


def track_video(frame_paths, t: dict, size) -> np.ndarray:
    from ultralytics import YOLO          # imported here: the rest of the project works without it
    model = YOLO(t["model"])              # fresh model per video = fresh tracker (ids start at 1)
    rows = []
    for k, p in enumerate(frame_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        sx, sy = size[0] / w, size[1] / h          # detect at full size, store in frame_size
        r = model.track(img, persist=True, tracker=t["tracker"], classes=[PERSON], conf=t["conf"],
                        imgsz=t["imgsz"], device=t["device"], verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy() * np.array([sx, sy, sx, sy])
        for b, i, c in zip(xyxy, r.boxes.id.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            rows.append([k, i, *b, c])
    return np.array(rows, dtype=np.float32).reshape(-1, 7)


def track_velocities(tracks: np.ndarray, window: int = 8, smooth: int = 5,
                     min_speed: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """One smoothed velocity per box: (centre[t] - centre[t - window]) / frames elapsed,
    after a moving average over `smooth` frames so arm / leg swing doesn't flip the direction."""
    vel = np.zeros((len(tracks), 2), dtype=np.float32)
    if len(tracks) == 0:
        return vel, np.zeros(0, dtype=bool)
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


def track_and_save(cfg: dict, frame_paths, out) -> np.ndarray:
    t = cfg["tracking"]
    tracks = track_video(frame_paths, t, cfg["frame_size"])
    vel, moving = track_velocities(tracks, t["window"], t["smooth"], t["min_speed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, tracks=tracks, velocity=vel, moving=moving,
                        frame_names=np.array([f.name for f in frame_paths]))
    tmp.replace(out)
    return tracks


def videos_for(cfg: dict, dataset: str):
    vids = list_videos(frames_dir(cfg, dataset))
    if dataset == "shanghaitech":
        test = set(load_split(cfg)["test"]) | set(load_split(cfg)["val"])
        vids = [(v, d) for v, d in vids if v in test]
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghaitech", "manual"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config("configs/behavior.yaml")
    vids = videos_for(cfg, args.dataset)[: args.limit]
    features_path(cfg, "tracking", args.dataset, "x").parent.mkdir(parents=True, exist_ok=True)
    todo = [(v, d) for v, d in vids if not features_path(cfg, "tracking", args.dataset, v).exists()]
    print(f"{args.dataset}: {len(vids)} videos, {len(vids) - len(todo)} done, {len(todo)} to track")
    t0 = time.time()
    for k, (v, d) in enumerate(todo, 1):
        frames = list_frames(d)
        tracks = track_and_save(cfg, frames, features_path(cfg, "tracking", args.dataset, v))
        n_ids = len(np.unique(tracks[:, 1])) if len(tracks) else 0
        print(f"[{k}/{len(todo)}] {v}: {len(frames)} frames, {n_ids} people ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
