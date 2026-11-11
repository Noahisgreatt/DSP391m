"""Step 1b, DATA -> MODELS: PyTorch datasets of frame windows (ASTT: models 1, 2, 5, 6).
Model 1 reads through the thesis repo's own loader; the others through VideoWindows."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as tud

from src.utils import list_frames


# ====================================================================== PyTorch datasets (models 1, 2)
LINKS = Path("data/processed/_links")      # folders of links the thesis repo's loader can read


def _jpg_view(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in list_frames(src):
        if not (dst / f.name).exists():
            os.symlink(f.resolve(), dst / f.name)
    return dst


def _repo_loader(folder, size):
    """The thesis repo's own data_utils.DataLoader (BGR, [-1, 1]), unchanged."""
    import torchvision.transforms as T
    from third_party.ASTT.data_utils import DataLoader
    return DataLoader(str(folder), T.Compose([T.ToTensor()]), resize_height=size, resize_width=size)


class VideoWindows(tud.Dataset):
    """Model 2: windows of T+1 frames that never cross two videos; each frame read once.
    Item: {"standard": (T+1, 3, S, S), "256": (T+1, 3, 256, 256)}, values in [-1, 1]."""

    def __init__(self, video_dirs, T=4, size=384, out=256, rgb=True):
        self.T, self.size, self.out, self.rgb = T, size, out, rgb
        self.frames = [list_frames(d) for d in video_dirs]
        self.index = [(v, i) for v, fr in enumerate(self.frames) for i in range(len(fr) - T)]

    def __len__(self):
        return len(self.index)

    def _load(self, p):
        img = cv2.imread(str(p))
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f = lambda s: (cv2.resize(img, (s, s)).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
        return f(self.size), f(self.out)

    def __getitem__(self, k):
        v, i = self.index[k]
        pairs = [self._load(p) for p in self.frames[v][i:i + self.T + 1]]
        return {"standard": torch.from_numpy(np.stack([a for a, _ in pairs])),
                "256": torch.from_numpy(np.stack([b for _, b in pairs]))}


def train_dataset(cfg, video_dirs):
    m = cfg["model"]
    if cfg["input"]["loader"] == "repo":          # model 1: one folder with all videos (repo quirk)
        root = LINKS / "train" / m["name"]
        for d in video_dirs:
            _jpg_view(d, root / d.name)
        return _repo_loader(root, m["image_size"])
    return VideoWindows(video_dirs, m["num_frames"], m["image_size"], m["output_size"], cfg["input"]["color"] == "rgb")


def video_dataset(cfg, video_dir):
    """Windows of ONE video (for scoring) and its number of frames."""
    m, video_dir = cfg["model"], Path(video_dir)
    if cfg["input"]["loader"] == "repo":
        ds = _repo_loader(_jpg_view(video_dir, LINKS / "videos" / video_dir.parent.parent.name / video_dir.name),
                          m["image_size"])
        return ds, len(ds.video_frames)
    ds = VideoWindows([video_dir], m["num_frames"], m["image_size"], m["output_size"], cfg["input"]["color"] == "rgb")
    return ds, len(ds.frames[0])
