"""The attribute files released by AI-VAD's authors, cut into our per-video format.

Their files hold one entry per frame, all videos of a split one after another (videos in sorted
name order, as in their video_dataset.py):
    <dir>/train/velocity.npy  deep_features.npy  pose.npy
    <dir>/test/velocity.npy   deep_features.npy  pose.npy
(Google Drive links: src/models/aivad/weights.py). Used with `features: released` in
configs/aivad.yaml to reproduce the paper's numbers exactly, without running the networks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import frames_dir
from src.data.frames import IMAGE_EXTS


def _per_frame(path: Path) -> list:
    a = np.load(path, allow_pickle=True)
    return list(a) if a.dtype == object or a.ndim == 1 else [x for x in a]


def frame_counts(cfg: dict, videos: list[str]) -> list[int]:
    root = frames_dir(cfg, "shanghaitech")
    return [sum(1 for f in (root / v).iterdir() if f.suffix.lower() in IMAGE_EXTS) for v in videos]


def load(cfg: dict, split: str, videos: list[str]) -> dict[str, dict]:
    d = Path(cfg["model"]["released_dir"]) / split
    counts = frame_counts(cfg, videos)
    vel = _per_frame(d / "velocity.npy")
    pose = _per_frame(d / "pose.npy")
    deep = _per_frame(d / "deep_features.npy") if (d / "deep_features.npy").exists() else None
    total = sum(counts)
    if len(vel) != total or len(pose) != total:
        raise SystemExit(
            f"released {split} features have {len(vel)} frames (pose {len(pose)}), our {len(videos)} "
            f"{split} videos have {total}. The frame counts differ (the Kaggle copy was cut from the "
            "videos differently), so the features can't be matched to videos. Use features: extract.")
    out, start = {}, 0
    for v, n in zip(videos, counts):
        fr = range(start, start + n)
        rows = [np.asarray(vel[i], np.float32).reshape(-1, 8) for i in fr]
        prow = [np.asarray(pose[i], np.float32).reshape(-1, 34) for i in fr]
        f = {"n_frames": np.array(n),
             "frame": np.concatenate([np.full(len(r), t, np.int32) for t, r in enumerate(rows)]),
             "velocity": np.concatenate(rows),
             "pose_frame": np.concatenate([np.full(len(p), t, np.int32) for t, p in enumerate(prow)]),
             "pose": np.concatenate(prow)}
        f["ids"] = -np.ones(len(f["frame"]), np.int64)
        f["mean_flow"] = np.zeros((len(f["frame"]), 2), np.float32)       # not released
        f["pose_object"] = -np.ones(len(f["pose"]), np.int64)
        if deep is not None:
            f["deep"] = np.concatenate([np.asarray(deep[i], np.float32).reshape(-1, 512) for i in fr]).astype(np.float16)
        out[v] = f
        start += n
    return out
