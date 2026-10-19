"""Compute optical flow for every video and save per-cell direction features.

Reads frame folders (ShanghaiTech directly, manual from data/interim/frames/manual),
runs Farneback flow on each consecutive pair, and writes one file per video:

    data/features/farneback/<dataset>/<video_id>.npz
        hof      (T-1, R, C, n_bins) float16
        mean     (T-1, R, C, 2)      float16   mean (dx, dy) px/frame
        moving   (T-1, R, C)         float16
        dominant (T-1, R, C)         int8      -1 = not moving
        small_flow (T-1, h, w, 2)    float16   only if flow.save_small_flow
        frame_names, size, grid, n_bins

Row t describes the motion from frame t to frame t+1.

    python -m src.features.optical_flow --dataset shanghai
    python -m src.features.optical_flow --dataset manual
    python -m src.features.optical_flow --dataset shanghai --limit 3    # quick test
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from src.features.direction import flow_to_grid_features
from src.utils import features_path, list_videos, load_config, read_frame, video_frames


def compute_flow(prev_gray: np.ndarray, gray: np.ndarray, fb: dict) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, fb["pyr_scale"], fb["levels"], fb["winsize"],
        fb["iterations"], fb["poly_n"], fb["poly_sigma"], 0)


def process_video(video_id: str, frame_dir, out_path, cfg: dict,
                  dataset: str) -> tuple[str, int]:
    cv2.setNumThreads(1)  # parallelism comes from processes, not OpenCV threads
    frames = video_frames(cfg, dataset, frame_dir)
    if len(frames) < 2:
        return video_id, 0

    resize = cfg["frames"]["resize"]
    fb = cfg["flow"]["farneback"]
    dcfg = cfg["direction"]
    save_small = cfg["flow"]["save_small_flow"]
    ds = cfg["flow"]["small_flow_downscale"]

    keys = ("hof", "mean", "moving", "dominant")
    acc = {k: [] for k in keys}
    small = []

    prev = cv2.cvtColor(read_frame(frames[0], resize), cv2.COLOR_BGR2GRAY)
    for p in frames[1:]:
        gray = cv2.cvtColor(read_frame(p, resize), cv2.COLOR_BGR2GRAY)
        flow = compute_flow(prev, gray, fb)
        feats = flow_to_grid_features(flow, tuple(dcfg["grid"]), dcfg["n_bins"],
                                      dcfg["min_magnitude"], dcfg["min_moving_fraction"])
        for k in keys:
            acc[k].append(feats[k])
        if save_small:
            h, w = flow.shape[:2]
            small.append(cv2.resize(flow, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
                         .astype(np.float16))
        prev = gray

    out = {
        "hof": np.stack(acc["hof"]).astype(np.float16),
        "mean": np.stack(acc["mean"]).astype(np.float16),
        "moving": np.stack(acc["moving"]).astype(np.float16),
        "dominant": np.stack(acc["dominant"]),
        "frame_names": np.array([f.name for f in frames]),
        "size": np.array(prev.shape[::-1]),            # (width, height)
        "grid": np.array(dcfg["grid"]),
        "n_bins": np.array(dcfg["n_bins"]),
    }
    if save_small:
        out["small_flow"] = np.stack(small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(out_path)                              # only finished files get the real name
    return video_id, len(frames) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghai", "manual"])
    ap.add_argument("--config", help="YAML that overrides params.yaml")
    ap.add_argument("--limit", type=int, help="only the first N videos (for testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    method = cfg["flow"]["method"]
    overwrite = args.overwrite or cfg["run"]["overwrite"]
    videos = list_videos(cfg, args.dataset)[: args.limit]
    # create the output folder up front so DVC finds it even when there is nothing to do
    features_path(cfg, method, args.dataset, "x").parent.mkdir(parents=True, exist_ok=True)

    todo = []
    for vid, d in videos:
        out = features_path(cfg, method, args.dataset, vid)
        if out.exists() and not overwrite:
            continue
        todo.append((vid, d, out))
    print(f"{args.dataset}: {len(videos)} videos, {len(videos) - len(todo)} already done, "
          f"{len(todo)} to process with {cfg['run']['workers']} workers")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=cfg["run"]["workers"]) as pool:
        futs = [pool.submit(process_video, vid, d, out, cfg, args.dataset)
                for vid, d, out in todo]
        for fut in as_completed(futs):
            vid, n = fut.result()
            done += 1
            print(f"[{done}/{len(todo)}] {vid}: {n} flow frames  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
