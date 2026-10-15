"""Torch dataset: windows of `seq_len` past flow fields -> (next flow field, pseudo-label).

Flow .npz files are unpacked once into memory-mappable .npy caches next to them
(data/features/<method>/<dataset>/_cache/), so random access across videos stays cheap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import resolve, split_of


def flow_cache(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (memmapped flow (M,H,W,2) float16, frame_idx (M,), n_frames)."""
    cache = npz_path.parent / "_cache"
    cache.mkdir(exist_ok=True)
    fpath, ipath = cache / f"{npz_path.stem}.flow.npy", cache / f"{npz_path.stem}.idx.npy"
    if not fpath.exists() or fpath.stat().st_mtime < npz_path.stat().st_mtime:
        d = np.load(npz_path)
        np.save(fpath, d["flow"])
        np.save(ipath, np.append(d["frame_idx"], int(d["n_frames"])).astype(np.int64))
    idx = np.load(ipath)
    return np.load(fpath, mmap_mode="r"), idx[:-1], int(idx[-1])


def load_pseudo_labels(label_methods: list[str] | str | None, labels_root: Path, dataset: str,
                       video_id: str, n_frames: int) -> np.ndarray:
    """Combine several label sources: per frame keep the label with the highest score."""
    if not label_methods:
        return np.zeros(n_frames, np.int64)
    if isinstance(label_methods, str):
        label_methods = [label_methods]
    best_s = np.full(n_frames, -1.0, np.float32)
    best_l = np.zeros(n_frames, np.int64)
    for m in label_methods:
        lp = labels_root / m / dataset / f"{video_id}.npy"
        if not lp.exists():
            continue
        lab = np.load(lp)[:n_frames].astype(np.int64)
        sp = lp.with_suffix(".score.npy")
        sc = np.load(sp)[:n_frames] if sp.exists() else lab.astype(np.float32)
        upd = (sc > best_s) & (lab > 0) | (best_s < 0)
        best_s = np.where(upd, sc, best_s)
        best_l = np.where(upd, lab, best_l)
    return best_l


def encode_flow(flow: torch.Tensor, clip: float) -> torch.Tensor:
    """(..., H, W, 2) raw flow -> (..., 3, H, W) channels (dx, dy, magnitude) scaled to ~[-1, 1]."""
    flow = flow.float().clamp(-clip, clip) / clip
    mag = flow.norm(dim=-1, keepdim=True) * 2 - 1
    return torch.cat([flow, mag], -1).movedim(-1, -3)


class FlowSequenceDataset(Dataset):
    def __init__(self, params: dict, cfg: dict, split: str = "train", video_ids: list[str] | None = None,
                 stride: int | None = None):
        dcfg = cfg["data"]
        self.seq_len = dcfg["seq_len"]
        self.clip = dcfg["flow_clip"]
        dataset = dcfg["dataset"]
        feat_dir = resolve(params["paths"]["features"]) / dcfg["feature_method"] / dataset
        labels_root = resolve(params["paths"]["labels"])
        stride = stride or dcfg["stride"]

        files = sorted(feat_dir.glob("*.npz"))
        if video_ids is None:
            files = [f for f in files if split_of(f.stem, params, dataset) == split]
        else:
            keep = set(video_ids)
            files = [f for f in files if f.stem in keep]
        if not files:
            raise RuntimeError(f"No '{split}' features found in {feat_dir}")

        self.videos, self.index = [], []
        for vi, f in enumerate(files):
            flow, fidx, n_frames = flow_cache(f)
            labels = load_pseudo_labels(dcfg.get("label_method"), labels_root, dataset, f.stem, n_frames)
            self.videos.append({"id": f.stem, "flow": flow, "frame_idx": fidx,
                                "labels": labels, "n_frames": n_frames})
            for t in range(self.seq_len, len(fidx), stride):  # predict flow[t] from flow[t-L:t]
                self.index.append((vi, t))

    def __len__(self) -> int:
        return len(self.index)

    def class_counts(self, num_classes: int) -> np.ndarray:
        lab = [self.videos[v]["labels"][self.videos[v]["frame_idx"][t]] for v, t in self.index]
        return np.bincount(np.asarray(lab, np.int64), minlength=num_classes)

    def __getitem__(self, i: int) -> dict:
        vi, t = self.index[i]
        v = self.videos[vi]
        seq = torch.from_numpy(np.array(v["flow"][t - self.seq_len:t]))   # (L,H,W,2)
        tgt = torch.from_numpy(np.array(v["flow"][t]))                    # (H,W,2)
        frame = int(v["frame_idx"][t])
        return {
            "x": encode_flow(seq, self.clip),                                # (L,3,H,W)
            "y_flow": (tgt.float().clamp(-self.clip, self.clip) / self.clip).movedim(-1, 0),  # (2,H,W)
            "y_cls": torch.tensor(int(v["labels"][frame]), dtype=torch.long),
            "frame": frame,
            "video": vi,
        }
