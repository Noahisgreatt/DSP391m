"""Stage 2a: frames -> Farneback optical flow + direction histograms.

Output: data/features/farneback/<dataset>/<video_id>.npz with
    flow      (M, H, W, 2) float16   flow arriving at frame_idx[k], in px/frame of compute_size
    hist      (M, gr, gc, bins) float32  direction histograms (see direction.py)
    frame_idx (M,) int32             0-based index of the frame each flow ends at
    n_frames  ()  int                total frames of the video

Usage:
    python -m src.features.optical_flow --dataset shanghaitech --config configs/features/farneback.yaml
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.features.direction import direction_histograms
from src.utils import list_frame_dirs, list_frames, load_params, load_yaml, resolve


def read_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)


def resize_flow(flow: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize spatially but keep values in compute_size pixel units (consistent across videos)."""
    return cv2.resize(flow, tuple(size), interpolation=cv2.INTER_AREA)


def compute_video(frame_dir: Path, out_path: Path, cfg: dict) -> int:
    frames = list_frames(frame_dir)
    stride = cfg["frame_stride"]
    fb = cfg["farneback"]
    flows, idx = [], []
    prev = read_gray(frames[0], cfg["compute_size"]) if frames else None
    for i in range(stride, len(frames), stride):
        cur = read_gray(frames[i], cfg["compute_size"])
        f = cv2.calcOpticalFlowFarneback(prev, cur, None, fb["pyr_scale"], fb["levels"],
                                         fb["winsize"], fb["iterations"], fb["poly_n"],
                                         fb["poly_sigma"], fb["flags"]) / stride
        flows.append(resize_flow(f, cfg["store_size"]).astype(np.float16))
        idx.append(i)
        prev = cur
    if not flows:
        return 0
    flow = np.stack(flows)
    d = cfg["direction"]
    hist = direction_histograms(flow, tuple(d["grid"]), d["bins"], d["min_magnitude"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, flow=flow, hist=hist, frame_idx=np.asarray(idx, np.int32),
             n_frames=len(frames))
    return len(flows)


def _job(a):
    return compute_video(*a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/features/farneback.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    params = load_params()
    cfg = load_yaml(resolve(args.config))
    frame_root = resolve(params["paths"]["frames"]) / args.dataset
    out_root = resolve(params["paths"]["features"]) / cfg["method"] / args.dataset

    jobs = []
    for d in list_frame_dirs(frame_root):
        if d.name.startswith("_"):
            continue
        out = out_root / f"{d.name}.npz"
        if out.exists() and not args.overwrite:
            continue
        jobs.append((d, out, cfg))

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            list(tqdm(ex.map(_job, jobs), total=len(jobs), desc="farneback"))
    else:
        for j in tqdm(jobs, desc="farneback"):
            _job(j)
    print(f"Flow for {len(jobs)} videos -> {out_root}")


if __name__ == "__main__":
    main()
