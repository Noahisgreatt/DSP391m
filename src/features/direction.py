"""Flow -> direction histograms / grid map, scene direction priors, flow-based counter-flow.

Three entry points:
  * direction_histograms(): called by optical_flow.py, stored in the same .npz as `hist`.
  * fit_scene_priors(): the dominant direction of every grid cell, learned from the
    NORMAL training videos of each camera/scene.
  * CLI: fits priors and writes flow-based pseudo-labels to
        data/labels/farneback/<dataset>/<video_id>.npy        (int8 class per frame: 0 / 2)
        data/labels/farneback/<dataset>/<video_id>.score.npy  (float32 0..1 per frame)

    python -m src.features.direction --dataset shanghaitech --config configs/features/farneback.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.utils import load_params, load_yaml, resolve, scene_of, split_of

COUNTERFLOW = 2  # class id shared with tracking.py (0 normal, 1 loitering, 2 counter-flow)


def bin_centers(bins: int) -> np.ndarray:
    ang = (np.arange(bins) + 0.5) * 2 * np.pi / bins
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (bins, 2)


def direction_histograms(flow: np.ndarray, grid: tuple[int, int], bins: int,
                         min_magnitude: float) -> np.ndarray:
    """flow: (N, H, W, 2) -> (N, gr, gc, bins) magnitude-weighted orientation histograms.

    Each cell's histogram is NOT normalized: its total mass = mean moving magnitude in that
    cell, so "how much" and "which way" are both kept.
    """
    flow = flow.astype(np.float32)
    n, h, w, _ = flow.shape
    gr, gc = grid
    mag = np.linalg.norm(flow, axis=-1)
    ang = np.mod(np.arctan2(flow[..., 1], flow[..., 0]), 2 * np.pi)
    b = np.minimum((ang / (2 * np.pi) * bins).astype(np.int64), bins - 1)
    weight = np.where(mag >= min_magnitude, mag, 0.0)

    rows = np.minimum(np.arange(h) * gr // h, gr - 1)
    cols = np.minimum(np.arange(w) * gc // w, gc - 1)
    cell = rows[:, None] * gc + cols[None, :]                    # (H, W)
    idx = cell[None] * bins + b                                  # (N, H, W)
    idx = idx + (np.arange(n) * gr * gc * bins)[:, None, None]
    out = np.bincount(idx.ravel(), weights=weight.ravel(), minlength=n * gr * gc * bins)
    pix_per_cell = np.bincount(cell.ravel(), minlength=gr * gc).reshape(gr, gc)
    return (out.reshape(n, gr, gc, bins) / pix_per_cell[None, :, :, None]).astype(np.float32)


def cell_vectors(hist: np.ndarray) -> np.ndarray:
    """(..., bins) histograms -> (..., 2) resultant motion vector per cell."""
    return hist @ bin_centers(hist.shape[-1])


def fit_scene_priors(hists: list[np.ndarray], min_magnitude: float) -> dict[str, np.ndarray]:
    """Dominant direction per cell from normal videos of one scene.

    Returns unit direction (gr, gc, 2) and coherence (gr, gc) in [0, 1]: how consistently
    that cell moves the same way (1 = one-way corridor, 0 = no preferred direction).
    """
    h = np.concatenate(hists, axis=0)                     # (N, gr, gc, bins)
    v = cell_vectors(h)                                   # (N, gr, gc, 2)
    mass = h.sum(-1)                                      # (N, gr, gc)
    moving = mass > min_magnitude * 0.05
    unit = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
    s = (unit * moving[..., None]).sum(0)                 # sum of unit vectors
    count = moving.sum(0)
    coherence = np.linalg.norm(s, axis=-1) / np.maximum(count, 1)
    dominant = s / (np.linalg.norm(s, axis=-1, keepdims=True) + 1e-6)
    return {"dominant": dominant.astype(np.float32), "coherence": coherence.astype(np.float32),
            "count": count.astype(np.int64)}


def counterflow_scores(hist: np.ndarray, prior: dict, cfg: dict,
                       min_coherence: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame flow-based counter-flow score (0..1) and per-cell cosine map.

    A cell counts only if the scene has a clear dominant direction there (coherence) and the
    cell is actually moving in this frame. Score maps cos == threshold to 0.5, cos == -1 to 1.
    """
    thr = cfg["counterflow_cos"]
    v = cell_vectors(hist)                                       # (N, gr, gc, 2)
    mass = hist.sum(-1)
    unit = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
    cos = (unit * prior["dominant"][None]).sum(-1)               # (N, gr, gc)
    valid = (mass > cfg["min_magnitude"] * 0.05) & (prior["coherence"][None] >= min_coherence)
    cell_score = np.clip(0.5 * cos / thr, 0, 1) * valid          # thr < 0
    cos = np.where(valid, cos, 1.0)
    return cell_score.reshape(len(hist), -1).max(1), cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/features/farneback.yaml")
    ap.add_argument("--prior-dataset", default=None,
                    help="Use another dataset's priors (e.g. manual videos of a known scene).")
    ap.add_argument("--smooth", type=int, default=9)
    args = ap.parse_args()

    params = load_params()
    cfg = load_yaml(resolve(args.config))
    dcfg = cfg["direction"]
    feat_dir = resolve(params["paths"]["features"]) / cfg["method"] / args.dataset
    label_dir = resolve(params["paths"]["labels"]) / cfg["method"] / args.dataset
    label_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(feat_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No features in {feat_dir}. Run src.features.optical_flow first.")

    # 1) Priors per scene from training (normal) videos.
    prior_src = resolve(params["paths"]["features"]) / cfg["method"] / (args.prior_dataset or args.dataset)
    by_scene: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in sorted(prior_src.glob("*.npz")):
        if split_of(f.stem, params, args.dataset) == "train":
            by_scene[scene_of(f.stem)].append(np.load(f)["hist"])
    if not by_scene:  # no train split (e.g. manual set): fall back to all videos
        for f in files:
            by_scene[scene_of(f.stem)].append(np.load(f)["hist"])
    priors = {s: fit_scene_priors(h, dcfg["min_magnitude"]) for s, h in by_scene.items()}
    prior_dir = feat_dir / "_priors"
    prior_dir.mkdir(exist_ok=True)
    for s, p in priors.items():
        np.savez(prior_dir / f"{s}.npz", **p)
    global_prior = fit_scene_priors([h for hs in by_scene.values() for h in hs], dcfg["min_magnitude"])

    # 2) Frame-level pseudo-labels.
    for f in tqdm(files, desc="flow counter-flow labels"):
        d = np.load(f)
        prior = priors.get(scene_of(f.stem), global_prior)
        score, _ = counterflow_scores(d["hist"], prior, dcfg)
        n_frames = int(d["n_frames"])
        full = np.zeros(n_frames, np.float32)
        full[d["frame_idx"]] = score
        if args.smooth > 1:
            full = median_filter(full, size=args.smooth, mode="nearest")
        labels = np.where(full >= 0.5, COUNTERFLOW, 0).astype(np.int8)
        np.save(label_dir / f"{f.stem}.npy", labels)
        np.save(label_dir / f"{f.stem}.score.npy", full)
    print(f"Priors for scenes {sorted(priors)} -> {prior_dir}\nLabels -> {label_dir}")


if __name__ == "__main__":
    main()
