"""PyTorch datasets: windows of 4 past frames + the frame to predict.

Both return the same dict, so training and scoring code is shared:
    "standard": (T+1, 3, image_size, image_size)   model input (first T frames)
    "256":      (T+1, 3, 256, 256)                 target = the last frame
Sample i predicts frame i+T of its video.

  loader: repo     (model 1) the thesis repo's data_utils.DataLoader, UNCHANGED: BGR in [-1, 1].
                   For training it is given one folder with all training videos, so (like the
                   repo) a window can span the end of one video and the start of the next.
  loader: windows  (model 2) windows stay inside one video; RGB or BGR in [-1, 1];
                   each frame is read once and resized twice.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from src.config import load_config
from src.data.frames import list_frames
from third_party.ASTT.data_utils import DataLoader as RepoDataLoader

LINKS = Path("data/processed/_links")      # folders of links the repo loader can read


def jpg_view(src_dir: Path, dst_dir: Path) -> Path:
    """A folder of links to only the .jpg frames (the repo loader expects nothing else in it)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in list_frames(src_dir):
        link = dst_dir / f.name
        if not link.exists():
            os.symlink(f.resolve(), link)
    return dst_dir


class VideoWindows(data.Dataset):
    def __init__(self, video_dirs: list[Path], num_frames=4, image_size=384, out_size=256,
                 color="rgb"):
        self.T, self.size, self.out, self.rgb = num_frames, image_size, out_size, color == "rgb"
        self.frames = [list_frames(d) for d in video_dirs]
        self.index = [(v, i) for v, fr in enumerate(self.frames) for i in range(len(fr) - num_frames)]

    def __len__(self):
        return len(self.index)

    def _load(self, path):
        img = cv2.imread(str(path))
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        big = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 127.5 - 1.0
        small = cv2.resize(img, (self.out, self.out)).astype(np.float32) / 127.5 - 1.0
        return big.transpose(2, 0, 1), small.transpose(2, 0, 1)

    def __getitem__(self, k):
        v, i = self.index[k]
        pairs = [self._load(p) for p in self.frames[v][i:i + self.T + 1]]
        return {"standard": torch.from_numpy(np.stack([b for b, _ in pairs])),
                "256": torch.from_numpy(np.stack([s for _, s in pairs]))}


def _repo(folder: Path, image_size: int):
    return RepoDataLoader(str(folder), transforms.Compose([transforms.ToTensor()]),
                          resize_height=image_size, resize_width=image_size)


def train_dataset(cfg: dict, video_dirs: list[Path]) -> data.Dataset:
    m, i = cfg["model"], cfg["input"]
    if i["loader"] == "repo":
        root = LINKS / "train" / m["name"]
        for d in video_dirs:
            jpg_view(d, root / d.name)
        return _repo(root, m["image_size"])
    return VideoWindows(video_dirs, m["num_frames"], m["image_size"], m["output_size"], i["color"])


def video_dataset(cfg: dict, video_dir: Path) -> tuple[data.Dataset, int]:
    """Windows of ONE video (for scoring) and its number of frames."""
    m, i = cfg["model"], cfg["input"]
    if i["loader"] == "repo":
        ds = _repo(jpg_view(video_dir, LINKS / "videos" / video_dir.parent.parent.name / video_dir.name),
                   m["image_size"])
        return ds, len(ds.video_frames)
    ds = VideoWindows([video_dir], m["num_frames"], m["image_size"], m["output_size"], i["color"])
    return ds, len(ds.frames[0])


if __name__ == "__main__":       # quick look: python -m src.data.dataset configs/astt_improved.yaml
    import sys
    from src.config import frames_dir
    from src.data.split import load_split
    cfg = load_config(sys.argv[1])
    vids = load_split(cfg)["train"][:2]
    ds = train_dataset(cfg, [frames_dir(cfg, "shanghaitech") / v for v in vids])
    b = ds[0]
    print(len(ds), {k: (tuple(v.shape), float(v.min()), float(v.max())) for k, v in b.items()})
