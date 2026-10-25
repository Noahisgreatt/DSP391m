"""Which ShanghaiTech videos are for training and which for testing (same for every model).

    python -m src.data.split                            (DVC stage: split)

Official ShanghaiTech Campus split, from the video names:
    train = XX_YYY  (330 videos, normal only: the models learn what "normal" looks like)
    test  = XX_YYYY (107 videos, with 0/1 labels per frame)
    val   = `val_fraction` of the test videos (0 by default = no validation, like papers)
Out: data/metadata/split.json
"""
from __future__ import annotations

import json
import re

import numpy as np

from src.config import frames_dir, labels_dir, load_config, split_file


def make_split(cfg: dict) -> dict:
    vids = sorted(d.name for d in frames_dir(cfg, "shanghaitech").iterdir() if d.is_dir())
    train = [v for v in vids if re.fullmatch(r"\d{2}_\d{3}", v)]
    test = [v for v in vids if re.fullmatch(r"\d{2}_\d{4}", v)]
    s = cfg["shanghaitech"]
    n_val = int(round(len(test) * s["val_fraction"]))
    rng = np.random.RandomState(s["seed"])
    val = sorted(str(x) for x in rng.choice(test, n_val, replace=False)) if n_val else []
    return {"train": train, "val": val, "test": sorted(set(test) - set(val))}


def load_split(cfg: dict) -> dict:
    f = split_file(cfg)
    if not f.exists():
        raise SystemExit(f"{f} missing. Run `python -m src.data.split` (or dvc repro split).")
    return json.loads(f.read_text())


def video_labels(cfg: dict, vid: str, n_frames: int) -> np.ndarray:
    """0/1 per frame; zeros for a training (normal) video without a label file."""
    f = labels_dir(cfg) / f"{vid}.npy"
    if not f.exists():
        return np.zeros(n_frames, np.int64)
    lab = np.load(f).astype(np.int64).ravel()
    if len(lab) != n_frames:
        print(f"warning: {vid} has {len(lab)} labels for {n_frames} frames; trimming/padding")
        lab = np.pad(lab, (0, max(0, n_frames - len(lab))))[:n_frames]
    return lab


def main():
    cfg = load_config()
    s = make_split(cfg)
    split_file(cfg).parent.mkdir(parents=True, exist_ok=True)
    split_file(cfg).write_text(json.dumps(s, indent=1))
    for k in ("train", "val", "test"):
        print(f"{k:6s} {len(s[k]):4d} videos  e.g. {s[k][:4]}")


if __name__ == "__main__":
    main()
