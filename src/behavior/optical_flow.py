"""Learn each camera's NORMAL walking direction from optical flow (Farneback).

    python -m src.behavior.optical_flow --dataset shanghaitech   (DVC stage: optical_flow@shanghaitech)
    python -m src.behavior.optical_flow --dataset manual         (DVC stage: optical_flow@manual)

1. Flow per frame pair -> per grid cell: direction histogram, mean motion, share of moving pixels.
2. Normal direction map = average direction of every cell over many frames.
   ShanghaiTech: one map per camera (01..13) from its TRAINING videos only (normal behaviour).
   Manual clips: one map per video (every phone clip is a different view).
Out: data/features/optical_flow/<dataset>/<video>.npz   per-frame cell features
     data/features/normal_maps/<dataset>/<scene>.npz    normal (R,C,2), consistency (R,C), samples
     consistency ~1: everyone goes the same way there; ~0: mixed or two-way traffic.
Image coordinates: x right, y DOWN, so bin 0 = right, 2 = down, 4 = left, 6 = up.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from src.config import features_path, frames_dir, load_config
from src.data.frames import list_frames, list_videos, read_frame
from src.data.split import load_split

DIRECTION_NAMES_8 = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]


def angle_to_bin(ang: np.ndarray, n_bins: int) -> np.ndarray:
    """Radians -> bin index, bin 0 centred on 'right' (0 rad)."""
    width = 2 * np.pi / n_bins
    return (np.floor((ang + width / 2) / width).astype(np.int64)) % n_bins


def grid_features(flow: np.ndarray, grid=(8, 8), n_bins=8, min_magnitude=1.0,
                  min_moving_fraction=0.02) -> dict:
    H, W = flow.shape[:2]
    R, C = grid
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    moving_px = mag > min_magnitude
    bins = angle_to_bin(ang, n_bins)
    cell = np.minimum((np.arange(H) * R) // H, R - 1)[:, None] * C + np.minimum((np.arange(W) * C) // W, C - 1)[None]
    m = moving_px.ravel()
    cell_m, bins_m, w = cell.ravel()[m], bins.ravel()[m], mag.ravel()[m]
    hof = np.bincount(cell_m * n_bins + bins_m, weights=w,
                      minlength=R * C * n_bins).astype(np.float64).reshape(R, C, n_bins)
    tot = hof.sum(-1, keepdims=True)
    hof = np.divide(hof, tot, out=np.zeros_like(hof), where=tot > 0)
    n_moving = np.bincount(cell_m, minlength=R * C)
    moving = (n_moving / np.maximum(np.bincount(cell.ravel(), minlength=R * C), 1)).reshape(R, C)
    mean = np.stack([np.bincount(cell_m, weights=flow[..., j].ravel()[m], minlength=R * C)
                     for j in range(2)], -1).astype(np.float64) / np.maximum(n_moving, 1)[:, None]
    mean = mean.reshape(R, C, 2)
    dominant = hof.argmax(-1)
    dominant[moving < min_moving_fraction] = -1
    mean[moving < min_moving_fraction] = 0.0
    return {"hof": hof.astype(np.float32), "mean": mean.astype(np.float32),
            "moving": moving.astype(np.float32), "dominant": dominant.astype(np.int8)}


def process_video(vid, frame_dir, out_path, cfg) -> tuple[str, int]:
    cv2.setNumThreads(1)
    frames, fb, o = list_frames(frame_dir), cfg["optical_flow"]["farneback"], cfg["optical_flow"]
    if len(frames) < 2:
        return vid, 0
    acc = defaultdict(list)
    prev = cv2.cvtColor(read_frame(frames[0], cfg["frame_size"]), cv2.COLOR_BGR2GRAY)
    for p in frames[1:]:
        gray = cv2.cvtColor(read_frame(p, cfg["frame_size"]), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, fb["pyr_scale"], fb["levels"],
                                            fb["winsize"], fb["iterations"], fb["poly_n"],
                                            fb["poly_sigma"], 0)
        for k, v in grid_features(flow, tuple(o["grid"]), o["n_bins"], o["min_magnitude"],
                                  o["min_moving_fraction"]).items():
            acc[k].append(v)
        prev = gray
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, hof=np.stack(acc["hof"]).astype(np.float16),
                        mean=np.stack(acc["mean"]).astype(np.float16),
                        moving=np.stack(acc["moving"]).astype(np.float16),
                        dominant=np.stack(acc["dominant"]), frame_names=np.array([f.name for f in frames]))
    tmp.replace(out_path)
    return vid, len(frames) - 1


def build_normal_map(flow_files, min_samples: int):
    """Average unit direction per cell -> normal (R,C,2), consistency (R,C), samples (R,C)."""
    acc = cnt = None
    for p in flow_files:
        d = np.load(p)
        mean, valid = d["mean"].astype(np.float32), d["dominant"] >= 0
        speed = np.linalg.norm(mean, axis=-1, keepdims=True)
        unit = np.divide(mean, speed, out=np.zeros_like(mean), where=speed > 0)
        s, n = (unit * valid[..., None]).sum(0), valid.sum(0).astype(np.float64)
        acc, cnt = (s, n) if acc is None else (acc + s, cnt + n)
    avg = acc / np.maximum(cnt, 1)[..., None]
    consistency = np.linalg.norm(avg, axis=-1)
    normal = np.divide(avg, consistency[..., None], out=np.zeros_like(avg), where=consistency[..., None] > 0)
    consistency[cnt < min_samples] = 0.0
    return normal.astype(np.float32), consistency.astype(np.float32), cnt


def scene_of(dataset: str, vid: str) -> str:
    return vid.split("_")[0] if dataset == "shanghaitech" else vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghaitech", "manual"])
    ap.add_argument("--limit", type=int, help="only N videos (quick test)")
    args = ap.parse_args()
    cfg = load_config("configs/behavior.yaml")
    vids = list_videos(frames_dir(cfg, args.dataset))
    if args.dataset == "shanghaitech":                           # normal behaviour only
        train = set(load_split(cfg)["train"])
        vids = [(v, d) for v, d in vids if v in train]
    vids = vids[: args.limit]
    todo = [(v, d) for v, d in vids if not features_path(cfg, "optical_flow", args.dataset, v).exists()]
    print(f"{args.dataset}: {len(vids)} videos, {len(todo)} to process")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg["run"]["workers"]) as pool:
        futs = [pool.submit(process_video, v, d, features_path(cfg, "optical_flow", args.dataset, v), cfg)
                for v, d in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            v, n = fut.result()
            print(f"[{k}/{len(todo)}] {v}: {n} flow frames ({time.time() - t0:.0f}s)")

    by_scene = defaultdict(list)
    for v, _ in vids:
        by_scene[scene_of(args.dataset, v)].append(features_path(cfg, "optical_flow", args.dataset, v))
    for scene, files in sorted(by_scene.items()):
        normal, cons, samples = build_normal_map(files, cfg["optical_flow"]["min_cell_samples"])
        out = features_path(cfg, "normal_maps", args.dataset, scene)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, normal=normal, consistency=cons, samples=samples)
    print(f"{len(by_scene)} normal direction maps written")


if __name__ == "__main__":
    main()
