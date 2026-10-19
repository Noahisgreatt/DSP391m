"""Shared helpers: config loading, frame listing, frame reading."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASETS = ("shanghai", "manual", "other")   # other = one-off videos given to src.predict


# ---------------------------------------------------------------- config
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(override_path: str | None = None, base_path: str = "params.yaml") -> dict:
    """Load params.yaml, then merge an optional override YAML on top of it."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f) or {}
    if override_path:
        with open(override_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


# ---------------------------------------------------------------- frames
def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_frames(folder: str | Path) -> list[Path]:
    """Image files in a folder, in natural order (2.jpg before 10.jpg)."""
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=_natural_key)


def frames_root(cfg: dict, dataset: str) -> Path:
    """Folder that holds one sub-folder of frames per video."""
    if dataset == "shanghai":
        return Path(cfg["data"]["shanghai_frames"])
    if dataset in ("manual", "other"):
        return Path(cfg["data"]["frames_dir"]) / dataset
    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")


def list_videos(cfg: dict, dataset: str) -> list[tuple[str, Path]]:
    """(video_id, frame_folder) pairs for a dataset, sorted by id."""
    root = frames_root(cfg, dataset)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. For 'manual', run src/data/extract_frames.py first."
        )
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=_natural_key)
    return [(d.name, d) for d in dirs]


def read_frame(path: str | Path, resize=None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    if resize:
        w, h = resize
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def frame_step(cfg: dict, dataset: str) -> int:
    """Manual videos were already sampled during extraction, so don't sample them twice."""
    return int(cfg["frames"]["step"]) if dataset == "shanghai" else 1


def video_frames(cfg: dict, dataset: str, folder: str | Path) -> list[Path]:
    return list_frames(folder)[:: frame_step(cfg, dataset)]


def video_fps(cfg: dict, dataset: str, video_id: str) -> float:
    """Frames per second of the frames the pipeline actually uses."""
    if dataset in ("manual", "other"):
        meta = frames_root(cfg, dataset) / video_id / "meta.json"
        if meta.exists():
            fps = json.loads(meta.read_text()).get("fps")
            if fps:
                return float(fps)
    fps = cfg["detection"]["fps"].get(dataset, cfg["detection"]["fps"]["manual"])
    return float(fps) / frame_step(cfg, dataset)


def features_path(cfg: dict, method: str, dataset: str, video_id: str) -> Path:
    return Path(cfg["data"]["features_dir"]) / method / dataset / f"{video_id}.npz"


def results_dir(run: str | Path) -> Path:
    """runs/<model>/<run_id>  ->  results/<model>/<run_id>  (metrics, scores, result videos)."""
    run = Path(run)
    return Path("results") / run.parent.name / run.name
