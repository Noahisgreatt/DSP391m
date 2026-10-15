"""Stage 1: raw videos / frame folders -> data/interim/frames/<dataset>/<video_id>/000001.jpg

Video ids are prefixed with the split ("train__01_001", "test__01_0014") so every later
stage knows the split without extra bookkeeping. Ground truth for test videos is copied to
data/interim/frames/<dataset>/_gt/<video_id>.npy.

Usage:
    python -m src.data.extract_frames --dataset shanghaitech
    python -m src.data.extract_frames --dataset manual
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import list_frame_dirs, list_frames, list_videos, load_params, resolve


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def video_to_frames(video: Path, out_dir: Path, max_side: int, quality: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), _resize(frame, max_side),
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
    cap.release()
    return i


def copy_frames(src_dir: Path, out_dir: Path, max_side: int, quality: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = list_frames(src_dir)
    for i, f in enumerate(frames, 1):
        img = cv2.imread(str(f))
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), _resize(img, max_side),
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
    return len(frames)


def copy_gt(gt_dir: str | None, raw_id: str, dst: Path) -> None:
    if not gt_dir:
        return
    src = resolve(gt_dir) / f"{raw_id}.npy"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    params = load_params(args.params)
    ds = params["datasets"][args.dataset]
    ex = params["extract"]
    out_root = resolve(params["paths"]["frames"]) / args.dataset
    gt_root = out_root / "_gt"

    jobs: list[tuple[str, Path, str, str | None]] = []  # (video_id, source, kind, gt_dir)
    policy = ds.get("train_with_anomaly", "keep")

    def add_train(name: str, src: Path, kind: str) -> None:
        gt = resolve(ds["train_gt"]) / f"{name}.npy" if ds.get("train_gt") else None
        if gt is not None and gt.exists() and np.load(gt).any() and policy != "keep":
            if policy == "move_to_test":
                jobs.append((f"test__{name}", src, kind, ds["train_gt"]))
            return
        jobs.append((f"train__{name}", src, kind, None))

    for v in list_videos(ds.get("train_videos") or "__none__"):
        add_train(v.stem, v, "video")
    for d in list_frame_dirs(ds.get("train_frames") or "__none__"):
        add_train(d.name, d, "frames")
    for v in list_videos(ds.get("test_videos") or "__none__"):
        jobs.append((f"test__{v.stem}", v, "video", ds.get("test_gt")))
    for d in list_frame_dirs(ds.get("test_frames") or "__none__"):
        jobs.append((f"test__{d.name}", d, "frames", ds.get("test_gt")))

    if not jobs:
        raise SystemExit(f"No input found for dataset '{args.dataset}'. Check params.yaml paths.")

    for vid, src, kind, gt in tqdm(jobs, desc=f"extract {args.dataset}"):
        out = out_root / vid
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            continue
        if kind == "video":
            video_to_frames(src, out, ex["max_side"], ex["jpeg_quality"])
        else:
            copy_frames(src, out, ex["max_side"], ex["jpeg_quality"])
        copy_gt(gt, vid.split("__", 1)[1], gt_root / f"{vid}.npy")

    print(f"Done: {len(jobs)} videos -> {out_root}")


if __name__ == "__main__":
    main()
