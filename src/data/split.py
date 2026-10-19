"""Which ShanghaiTech videos are used for training, validation and testing (shared by all models).

Split
-----
Both models are unsupervised: they must be TRAINED ON NORMAL VIDEOS ONLY. Your copy
(data/raw/SHANGHAI/SHANGHAI_TRAIN, 238 videos + SHANGHAI_train.txt + label/) looks like the
weakly-supervised ShanghaiTech split, whose "train" part also contains abnormal videos.
So every video is classified using, in this order:
  1. its frame labels in label/<video>.(npy|txt): any 1 -> abnormal, all 0 -> normal
  2. a 0/1 (or Normal/Abnormal) entry for it in SHANGHAI_train.txt
  3. the original ShanghaiTech naming: 01_001 (3 digits) = normal training video
Then:
  train = normal videos, minus a held-out share (`normal_holdout`)
  test  = abnormal videos + held-out normal videos   (frame labels needed for AUC)
  val   = `val_fraction` of the test videos, used only to pick the best epoch
Videos whose status can't be determined are left out and listed.
`python -m src.data.split` prints the split so you can check it.

"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.utils import list_frames, load_config

VID_RE = re.compile(r"(\d{2}_\d{3,4})")


# ---------------------------------------------------------------- split
def _read_txt_labels(path: Path) -> dict[str, int]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = VID_RE.search(line)
        if not m:
            continue
        toks = re.split(r"[\s,;]+", line.strip())
        lab = None
        for t in reversed(toks):
            tl = t.lower()
            if t in ("0", "1"):
                lab = int(t); break
            if tl.startswith("abnormal") or tl.startswith("anomal"):
                lab = 1; break
            if tl.startswith("normal"):
                lab = 0; break
        out.setdefault(m.group(1), lab)
    return out


def _load_frame_labels(label_dir: Path, vid: str):
    for ext in (".npy", ".txt", ".csv"):
        p = label_dir / f"{vid}{ext}"
        if p.exists():
            arr = np.load(p, allow_pickle=True) if ext == ".npy" else np.loadtxt(p, delimiter=None if ext == ".txt" else ",")
            return np.asarray(arr).astype(np.int64).ravel()
    return None


def discover_shanghai(cfg: dict) -> list[dict]:
    frames_dir = Path(cfg["data"]["shanghai_frames"])
    root = frames_dir.parent
    label_dir = root / "label"
    txt = next(iter(sorted(root.glob("*.txt"))), None)
    txt_labels = _read_txt_labels(txt) if txt else {}

    videos = []
    for d in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        vid = d.name
        labels = _load_frame_labels(label_dir, vid) if label_dir.exists() else None
        if labels is not None:
            normal, source = bool(labels.max() == 0), "frame labels"
        elif txt_labels.get(vid) is not None:
            normal, source = bool(txt_labels[vid] == 0), txt.name
        elif re.fullmatch(r"\d{2}_\d{3}", vid):
            normal, source = True, "name (3-digit = normal)"
        else:
            normal, source = None, "unknown"
        videos.append({"id": vid, "dir": str(d), "normal": normal, "source": source,
                       "has_labels": labels is not None, "_labels": labels})
    return videos


def make_split(cfg: dict) -> dict:
    d = cfg["data"]
    override = d.get("split_override")
    if override:                     # a json with {"train": [...], "val": [...], "test": [...]}
        return json.loads(Path(override).read_text())

    videos = discover_shanghai(cfg)
    rng = np.random.RandomState(d["seed"])
    normal = [v["id"] for v in videos if v["normal"] is True]
    abnormal = [v["id"] for v in videos if v["normal"] is False]
    unknown = [v["id"] for v in videos if v["normal"] is None]

    # hold out normal videos per scene so every camera stays in training
    by_scene: dict[str, list[str]] = {}
    for v in normal:
        by_scene.setdefault(v.split("_")[0], []).append(v)
    held = []
    for vs in by_scene.values():
        k = int(round(len(vs) * d["normal_holdout"]))
        if k and len(vs) > k:
            held += [str(x) for x in rng.choice(vs, k, replace=False)]
    train = sorted(set(normal) - set(held))
    test_all = sorted(abnormal + held)
    n_val = max(1, int(round(len(test_all) * d["val_fraction"]))) if test_all else 0
    val = sorted(str(x) for x in rng.choice(test_all, n_val, replace=False)) if n_val else []
    test = sorted(set(test_all) - set(val))
    return {"train": train, "val": list(val), "test": test, "unknown": unknown,
            "sources": {v["id"]: v["source"] for v in videos}}


def video_labels(cfg: dict, vid: str, n_frames: int) -> np.ndarray:
    """Frame labels for a video, zeros for a known-normal video without a label file."""
    lab = _load_frame_labels(Path(cfg["data"]["shanghai_frames"]).parent / "label", vid)
    if lab is None:
        return np.zeros(n_frames, np.int64)
    if len(lab) != n_frames:
        print(f"warning: {vid} has {len(lab)} labels for {n_frames} frames; trimming/padding")
        lab = np.pad(lab, (0, max(0, n_frames - len(lab))))[:n_frames]
    return lab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print the ShanghaiTech split")
    ap.add_argument("--config", default="models/thesis_astt/config.yaml")
    cfg = load_config(ap.parse_args().config)
    s = make_split(cfg)
    srcs = {}
    for v, src in s["sources"].items():
        srcs[src] = srcs.get(src, 0) + 1
    print("status decided by:", srcs)
    for k in ("train", "val", "test", "unknown"):
        print(f"{k:8s} {len(s[k]):4d} videos  e.g. {s[k][:5]}")
    if s["unknown"]:
        print("unknown videos are NOT used. Add labels or a split_override file to include them.")
