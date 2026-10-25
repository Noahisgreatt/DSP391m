"""Unpack the two ShanghaiTech zips into one clean folder + write the data card.

    python -m src.data.prepare_shanghaitech            (DVC stage: prepare_shanghaitech)

In:   data/raw/shanghaitech/train.zip, test.zip  (Kaggle parts, 238 + 199 videos as frames)
Out:  data/processed/shanghaitech/frames/<video>/<n>.jpg   all 437 videos side by side
      data/processed/shanghaitech/labels/<video>.npy       0/1 per frame (pixel masks skipped)
      data/metadata/shanghaitech.json                      data card: source, counts, checks

The Kaggle parts are the weakly-supervised split (238/199). The OFFICIAL Campus split is
rebuilt from the names in src/data/split.py: XX_YYY = 330 train videos, XX_YYYY = 107 test.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

from src.config import frames_dir, labels_dir, load_config
from src.data.frames import IMAGE_EXTS

TRAIN_RE, TEST_RE = re.compile(r"\d{2}_\d{3}"), re.compile(r"\d{2}_\d{4}")


def unpack(zip_path: Path, frames: Path, labels: Path) -> tuple[set, int]:
    videos, n_labels = set(), 0
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir() and "pixel" not in m.filename.lower()]
        for k, m in enumerate(members, 1):
            p = PurePosixPath(m.filename)
            if p.suffix.lower() in IMAGE_EXTS:
                dst = frames / p.parent.name / p.name
                videos.add(p.parent.name)
            elif p.suffix == ".npy":
                dst = labels / p.name
                n_labels += 1
            else:
                continue
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
            if k % 20000 == 0 or k == len(members):
                print(f"  {zip_path.name}: {k}/{len(members)} files")
    return videos, n_labels


def dvc_hash(path: Path) -> str | None:
    """md5 that DVC recorded for the raw data (data/raw/shanghaitech.dvc), if it is tracked."""
    f = path.with_name(path.name + ".dvc")
    if f.exists():
        return yaml.safe_load(f.read_text())["outs"][0].get("md5")
    return None


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["paths"]["raw_shanghaitech"])
    frames, labels = frames_dir(cfg, "shanghaitech"), labels_dir(cfg)
    labels.mkdir(parents=True, exist_ok=True)
    parts = {}
    for part in ("train", "test"):
        zp = raw / f"{part}.zip"
        if not zp.exists():
            raise SystemExit(f"{zp} missing. Run `dvc pull` or `python -m src.data.download`.")
        vids, n_lab = unpack(zp, frames, labels)
        parts[part] = {"zip": zp.name, "zip_bytes": zp.stat().st_size, "videos": len(vids),
                       "label_files": n_lab}
        print(f"[{part}] {len(vids)} videos, {n_lab} label files")

    # ---- checks
    counts = {d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
              for d in frames.iterdir() if d.is_dir()}
    train = sorted(v for v in counts if TRAIN_RE.fullmatch(v))
    test = sorted(v for v in counts if TEST_RE.fullmatch(v))
    problems, abnormal_videos, abnormal_frames = [], 0, 0
    for v in sorted(counts):
        lab_f = labels / f"{v}.npy"
        if v in test:
            if not lab_f.exists():
                problems.append(f"test video {v} has no label file")
                continue
            lab = np.load(lab_f)
            if len(lab) != counts[v]:
                problems.append(f"{v}: {len(lab)} labels vs {counts[v]} frames")
            abnormal_videos += int(lab.max() > 0)
            abnormal_frames += int(lab.sum())
        elif v in train and lab_f.exists() and np.load(lab_f).max() > 0:
            problems.append(f"training video {v} has abnormal frames")
    other = sorted(set(counts) - set(train) - set(test))
    if other:
        problems.append(f"{len(other)} folders with unexpected names, e.g. {other[:3]}")

    card = {
        "dataset": "ShanghaiTech Campus",
        "source": {"kaggle_train": cfg["shanghaitech"]["kaggle_train"],
                   "kaggle_test": cfg["shanghaitech"]["kaggle_test"],
                   "raw_dvc_md5": dvc_hash(raw), "parts": parts},
        "videos": len(counts), "frames": int(sum(counts.values())),
        "official_split": {
            "train": {"videos": len(train), "frames": int(sum(counts[v] for v in train))},
            "test": {"videos": len(test), "frames": int(sum(counts[v] for v in test)),
                     "abnormal_videos": abnormal_videos, "abnormal_frames": abnormal_frames}},
        "expected": {"videos": 437, "train_videos": 330, "test_videos": 107, "test_frames": 40791},
        "fps": cfg["shanghaitech"]["fps"],
        "problems": problems,
    }
    meta = Path(cfg["paths"]["metadata"])
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "shanghaitech.json").write_text(json.dumps(card, indent=2))
    s = card["official_split"]
    print(f"\nvideos {card['videos']} (437)  train {s['train']['videos']} (330)  "
          f"test {s['test']['videos']} (107), {s['test']['frames']} frames (40791)")
    print("problems:", problems or "none")


if __name__ == "__main__":
    main()
