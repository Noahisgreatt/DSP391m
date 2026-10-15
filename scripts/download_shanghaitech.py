"""Download ShanghaiTech Campus from Kaggle and lay it out the way params.yaml expects.

Needs a Kaggle API token (~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY).
On a Kaggle notebook, skip the download and point --src at /kaggle/input/... instead:

    python scripts/download_shanghaitech.py
    python scripts/download_shanghaitech.py --src /kaggle/input/datasets/nikanvasei/shanghaitech-campus-dataset

Result:
    data/raw/shanghaitech/training/frames/<clip>/*.jpg     (+ training/test_frame_mask/<clip>.npy if present)
    data/raw/shanghaitech/testing/frames/<clip>/*.jpg
    data/raw/shanghaitech/testing/test_frame_mask/<clip>.npy
Frames are symlinked when possible (no copy).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "shanghaitech"
KAGGLE_SLUG = "nikanvasei/shanghaitech-campus-dataset"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    except OSError:
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)


def find(root: Path, *names: str) -> Path | None:
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            return p
    return None


def organize(src: Path) -> None:
    for split, keys in (("training", ("shanghai_train", "training", "train")),
                        ("testing", ("shanghai_test", "testing", "test"))):
        base = find(src, *keys)
        if base is None:
            print(f"[warn] no {split} folder found under {src}")
            continue
        frames = find(base, "frames") or base
        labels = find(base, "label", "labels", "test_frame_mask")
        n = 0
        for clip in sorted(d for d in frames.iterdir() if d.is_dir()):
            link(clip, DEST / split / "frames" / clip.name)
            n += 1
        if labels:
            for f in labels.glob("*.npy"):
                link(f, DEST / split / "test_frame_mask" / f.name)
        print(f"{split}: {n} clips, labels: {'yes' if labels else 'no'}")
        # original split ships training as .avi videos
        vids = find(base, "videos")
        if vids:
            for v in vids.glob("*.avi"):
                link(v, DEST / split / "videos" / v.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="Already-downloaded dataset folder.")
    args = ap.parse_args()
    src = Path(args.src) if args.src else ROOT / "data" / "raw" / "_download" / "shanghaitech"
    if not args.src:
        src.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(src), "--unzip"], check=True)
    organize(src)


if __name__ == "__main__":
    main()
