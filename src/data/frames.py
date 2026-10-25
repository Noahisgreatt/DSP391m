"""Reading frames from disk: list a video's frames in order, read one, list the videos."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_frames(folder: str | Path) -> list[Path]:
    """Image files of one video, in natural order (2.jpg before 10.jpg)."""
    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS),
                  key=natural_key)


def list_videos(root: str | Path) -> list[tuple[str, Path]]:
    """(video_id, frame_folder) for every video folder under root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run the prepare step first (dvc repro).")
    return [(d.name, d) for d in sorted((d for d in root.iterdir() if d.is_dir()), key=natural_key)]


def read_frame(path: str | Path, size=None) -> np.ndarray:
    """BGR uint8 image, resized to size=(width, height) if given."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    if size and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img
