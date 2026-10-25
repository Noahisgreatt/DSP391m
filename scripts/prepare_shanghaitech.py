"""Download ShanghaiTech Campus (train + test) from Kaggle and build the layout the code reads.

    python scripts/prepare_shanghaitech.py                       # download both parts + prepare
    python scripts/prepare_shanghaitech.py --train-src DIR --test-src DIR   # already downloaded

Sources (two separate Kaggle datasets):
    "train": nikanvasei/shanghaitech-campus-dataset       SHANGHAI/SHANGHAI_TRAIN, 238 videos
    "test":  nikanvasei/shanghaitech-campus-dataset-test  SHANGHAI/SHANGHAI_Test,  199 videos
These two Kaggle parts are the WEAKLY-SUPERVISED split (238/199), all as frames + label .npy.
Together they hold all 437 ShanghaiTech videos, so the OFFICIAL Campus split is rebuilt from
the names: XX_YYY = 330 training videos (normal), XX_YYYY = 107 test videos.
Needs a Kaggle API token in ~/.kaggle/kaggle.json.

Result (what params.yaml -> data.shanghai_frames points at):
    data/raw/shanghaitech/frames/<video>/...jpg   all 437 videos side by side (linked, not copied)
    data/raw/shanghaitech/label/<video>.npy       0/1 per frame
(.avi videos, if a source has them, are cut into frames with OpenCV.)
Safe to re-run: finished videos are skipped.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "shanghaitech"
DL = ROOT / "data" / "raw" / "_download"
SLUGS = {"train": "nikanvasei/shanghaitech-campus-dataset",
         "test": "nikanvasei/shanghaitech-campus-dataset-test"}
IMG = {".jpg", ".jpeg", ".png"}


def download(part: str) -> Path:
    dst = DL / f"shanghaitech_{part}"
    if dst.exists() and any(dst.iterdir()):
        print(f"[{part}] already downloaded: {dst}")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[{part}] downloading {SLUGS[part]} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", SLUGS[part], "-p", str(dst), "--unzip"],
                   check=True)
    return dst


def frame_dirs(root: Path) -> dict[str, Path]:
    """Every folder that directly contains images, keyed by its name (= video id)."""
    out = {}
    for d in root.rglob("*"):
        if d.is_dir() and any(f.suffix.lower() in IMG for f in d.iterdir()):
            if d.name in out:
                print(f"[warn] two frame folders named {d.name}: {out[d.name]} and {d}")
            out[d.name] = d
    return out


def extract(video: Path, dst: Path, resize=None) -> int:
    done = dst / ".done"
    if done.exists():
        return int(done.read_text())
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / f"{n:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    done.write_text(str(n))
    return n


def link_dir(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        if f.suffix.lower() in IMG:
            t = dst / f.name
            if not t.exists():
                try:
                    os.symlink(f.resolve(), t)
                except OSError:
                    shutil.copy2(f, t)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-src", help="already-downloaded train part")
    ap.add_argument("--test-src", help="already-downloaded test part")
    ap.add_argument("--resize", help="e.g. 640x360 to save disk (default: keep original size)")
    args = ap.parse_args()
    resize = tuple(int(x) for x in args.resize.split("x")) if args.resize else None
    train_src = Path(args.train_src) if args.train_src else download("train")
    test_src = Path(args.test_src) if args.test_src else download("test")
    frames_out, label_out = OUT / "frames", OUT / "label"
    label_out.mkdir(parents=True, exist_ok=True)

    # ---- frames: .avi videos are cut into frames; frame folders are linked
    clips = {}
    for part, src in (("train", train_src), ("test", test_src)):
        videos = sorted(src.rglob("*.avi")) + sorted(src.rglob("*.mp4"))
        for k, v in enumerate(videos, 1):
            clips[v.stem] = extract(v, frames_out / v.stem, resize)
            if k % 20 == 0 or k == len(videos):
                print(f"  [{part}] {k}/{len(videos)} videos cut into frames")
        dirs = {k: v for k, v in frame_dirs(src).items() if "pixel" not in str(v).lower()}
        for name, d in dirs.items():
            clips[name] = link_dir(d, frames_out / name)
        print(f"[{part}] {len(videos)} videos + {len(dirs)} frame folders")
        masks = [m for m in src.rglob("*.npy") if "pixel" not in str(m).lower()]
        for m in masks:
            shutil.copy2(m, label_out / m.name)
        print(f"[{part}] {len(masks)} label files")

    # ---- checks (official Campus naming: train XX_YYY, test XX_YYYY)
    import re
    train_ids = [c for c in clips if re.fullmatch(r"\d{2}_\d{3}", c)]
    test_ids = [c for c in clips if re.fullmatch(r"\d{2}_\d{4}", c)]
    other = sorted(set(clips) - set(train_ids) - set(test_ids))
    problems = []
    for name, n in clips.items():
        lab = label_out / f"{name}.npy"
        if name in test_ids and not lab.exists():
            problems.append(f"test video {name} has no label file")
        elif lab.exists() and len(np.load(lab)) != n:
            problems.append(f"{name}: {len(np.load(lab))} labels vs {n} frames")
    bad_train = [c for c in train_ids if (label_out / f"{c}.npy").exists()
                 and np.load(label_out / f"{c}.npy").max() > 0]
    if bad_train:
        problems.append(f"{len(bad_train)} training videos contain abnormal frames, e.g. {bad_train[:3]}")
    if other:
        problems.append(f"{len(other)} folders with unexpected names, e.g. {other[:3]}")
    abnormal = sum(int(np.load(label_out / f"{c}.npy").max() > 0) for c in test_ids
                   if (label_out / f"{c}.npy").exists())
    print("\n=== summary (official ShanghaiTech Campus split)")
    print(f"all videos   : {len(clips)}   (official: 437)")
    print(f"train videos : {len(train_ids)}   (official: 330), {sum(clips[c] for c in train_ids)} frames")
    print(f"test videos  : {len(test_ids)}   (official: 107), {abnormal} contain anomalies, "
          f"{sum(clips[c] for c in test_ids)} frames (official: 40791)")
    print("problems     :", "none" if not problems else "")
    for p in problems[:20]:
        print("  -", p)


if __name__ == "__main__":
    main()
