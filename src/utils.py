"""Small helpers used everywhere: settings, paths, reading frames.

    cfg = load_config("configs/astt_baseline.yaml")    # configs/data.yaml + that file
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- settings
def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(*paths) -> dict:
    """configs/data.yaml, then the given YAML files on top (later files win)."""
    cfg: dict = {}
    for p in ("configs/data.yaml", *paths):
        if p:
            cfg = _merge(cfg, yaml.safe_load(Path(p).read_text()) or {})
    return cfg


def config_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, check=True).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------- paths
def frames_dir(cfg, dataset):                 # data/processed/<dataset>/frames/<video>/
    return Path(cfg["paths"]["processed"]) / dataset / "frames"


def labels_dir(cfg):                          # data/processed/shanghaitech/labels/<video>.npy
    return Path(cfg["paths"]["processed"]) / "shanghaitech" / "labels"


def features_path(cfg, kind, dataset, video):  # data/features/<kind>/<dataset>/<video>.npz
    return Path(cfg["paths"]["features"]) / kind / dataset / f"{video}.npz"


def out_dir(cfg, stage, smoke=False):         # outputs/<training|evaluation>/<model>[_smoke]
    return Path(cfg["paths"]["outputs"]) / stage / (cfg["model"]["name"] + ("_smoke" if smoke else ""))


def video_fps(cfg, dataset, video) -> float:
    if dataset == "shanghaitech":
        return float(cfg["shanghaitech"]["fps"])
    meta = frames_dir(cfg, dataset) / video / "meta.json"
    fps = json.loads(meta.read_text()).get("fps") if meta.exists() else None
    return float(fps or cfg["manual"]["fps_fallback"])


# ---------------------------------------------------------------- frames
def _natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_frames(folder) -> list[Path]:
    """Images of one video in natural order (2.jpg before 10.jpg)."""
    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=_natural)


def list_videos(root) -> list[tuple[str, Path]]:
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"{root} not found. Prepare the data first (dvc repro split).")
    return [(d.name, d) for d in sorted((d for d in root.iterdir() if d.is_dir()), key=_natural)]


def read_frame(path, size=None) -> np.ndarray:
    """BGR uint8 image, resized to size = (width, height) if given."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    if size and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img
