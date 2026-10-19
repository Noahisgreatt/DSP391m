"""Detect and track people (YOLO + ByteTrack), one file per video.

Output: data/features/tracking/<dataset>/<video_id>.npz
    tracks      (N, 7) float32  columns: frame_idx, track_id, x1, y1, x2, y2, conf
                                boxes are in the RESIZED frame coordinates (frames.resize),
                                so they line up with the flow features
    velocity    (N, 2) float32  smoothed (vx, vy) of the box centre, px/frame
    moving      (N,)   bool     speed > tracking.min_speed
    frame_names (T,)            frame file names, in order

Detection runs on the full-resolution frame (better for small people); boxes are then
scaled to the resized size.

    python -m src.features.tracking --dataset shanghai
    python -m src.features.tracking --dataset manual --limit 2
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.features.direction import track_velocities
from src.utils import features_path, list_videos, load_config, video_frames

PERSON = 0  # COCO class id


def track_video(model_name: str, frame_paths, tcfg: dict, resize) -> np.ndarray:
    from ultralytics import YOLO  # imported here so the rest of the project works without it

    model = YOLO(model_name)      # fresh model per video = fresh tracker state (ids start at 1)
    rows = []
    for t, p in enumerate(frame_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        sx, sy = (resize[0] / w, resize[1] / h) if resize else (1.0, 1.0)
        r = model.track(img, persist=True, tracker=tcfg["tracker"], classes=[PERSON],
                        conf=tcfg["conf"], imgsz=tcfg["imgsz"], device=tcfg["device"],
                        verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy() * np.array([sx, sy, sx, sy])
        ids = r.boxes.id.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        for b, i, c in zip(xyxy, ids, conf):
            rows.append([t, i, *b, c])
    return np.array(rows, dtype=np.float32).reshape(-1, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghai", "manual"])
    ap.add_argument("--config", help="YAML that overrides params.yaml")
    ap.add_argument("--limit", type=int, help="only the first N videos (for testing)")
    ap.add_argument("--max-frames", type=int, help="only the first N frames per video (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["tracking"]
    resize = cfg["frames"]["resize"]
    overwrite = args.overwrite or cfg["run"]["overwrite"]
    videos = list_videos(cfg, args.dataset)[: args.limit]
    out_dir = features_path(cfg, "tracking", args.dataset, "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [(v, d) for v, d in videos
            if overwrite or not features_path(cfg, "tracking", args.dataset, v).exists()]
    print(f"{args.dataset}: {len(videos)} videos, {len(videos) - len(todo)} already done, "
          f"{len(todo)} to process with {tcfg['model']}")

    t0 = time.time()
    for k, (vid, d) in enumerate(todo, 1):
        frames = video_frames(cfg, args.dataset, d)[: args.max_frames]
        tracks = track_video(tcfg["model"], frames, tcfg, resize)
        vel, moving = track_velocities(tracks, tcfg["window"], tcfg["smooth"], tcfg["min_speed"])
        out = features_path(cfg, "tracking", args.dataset, vid)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, tracks=tracks, velocity=vel, moving=moving,
                            frame_names=np.array([f.name for f in frames]))
        tmp.replace(out)
        n_ids = len(np.unique(tracks[:, 1])) if len(tracks) else 0
        print(f"[{k}/{len(todo)}] {vid}: {len(frames)} frames, {n_ids} people tracked "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
