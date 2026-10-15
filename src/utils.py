"""Shared helpers: config loading, CLI overrides, paths, seeding. Nothing here is dataset-specific."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_params(path: str | os.PathLike | None = None) -> dict:
    return load_yaml(path or ROOT / "params.yaml")


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` overrides (value parsed as YAML) without touching the file.

    Example: --set train.epochs=1 model.pretrained=false
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got: {item}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve(path: str | os.PathLike) -> Path:
    """Paths in configs are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def list_videos(folder: str | os.PathLike) -> list[Path]:
    folder = resolve(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def list_frame_dirs(folder: str | os.PathLike) -> list[Path]:
    folder = resolve(folder)
    if not folder.exists():
        return []
    return sorted(d for d in folder.iterdir() if d.is_dir())


def list_frames(frame_dir: str | os.PathLike) -> list[Path]:
    d = Path(frame_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def split_of(video_id: str, params: dict, dataset: str) -> str:
    """Return 'train' or 'test'. Video ids carry a prefix written by extract_frames.py."""
    return "test" if video_id.startswith("test__") else "train"


def scene_of(video_id: str) -> str:
    """ShanghaiTech ids look like 01_0014 -> scene 01. Other datasets fall back to 'all'."""
    base = video_id.split("__", 1)[-1]
    head = base.split("_", 1)[0]
    return head if head.isdigit() else "all"
