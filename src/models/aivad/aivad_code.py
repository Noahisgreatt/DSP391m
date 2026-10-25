"""Access to AI-VAD's own code in third_party/AIVAD (unchanged, see third_party/fetch.py).

AI-VAD's scripts import each other as top-level modules (`from video_dataset import ...`,
`pre_processing.flownet_networks...`), so its folder is put on sys.path. Two things are plugged
in from OUTSIDE instead of editing its files:
  * FlowNet2's CUDA extensions -> pure PyTorch (flownet2_ops.py)
  * `object_detector` (needs detectron2) -> a stub, when only the box filters are needed (model 4)
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np

from src.models.aivad import flownet2_ops

ROOT = Path(__file__).resolve().parents[3] / "third_party" / "AIVAD"


def _path():
    if not ROOT.exists():
        raise SystemExit("third_party/AIVAD missing: run  python third_party/fetch.py")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def flownet2_class():
    _path()
    flownet2_ops.install()
    return importlib.import_module("pre_processing.flownet_networks.flownet2_models").FlowNet2


def bbox_functions(need_detector: bool):
    """AI-VAD's pre_processing/bboxes.py: Mask R-CNN wrapper + its box filters + moving blobs."""
    _path()
    sys.path.insert(0, str(ROOT / "pre_processing"))      # bboxes.py: `from object_detector import`
    if not need_detector and "object_detector" not in sys.modules:
        try:
            import detectron2  # noqa: F401
        except ImportError:                                # model 4 needs only the filters
            stub = types.ModuleType("object_detector")
            stub.Predictor = None
            sys.modules["object_detector"] = stub
    return importlib.import_module("bboxes")


def velocity_histogram(flow, magnitude, orientation, orientations=8, motion_threshold=0.):
    """Same numbers as AI-VAD's extract_velocity (which needs numpy < 2: it assigns a 1x1 array to
    one element). Its uniform_filter over the whole crop, read at the centre, IS the mean over
    the crop, so: per direction bin, the mean of the magnitudes falling in that bin (0 elsewhere)."""
    deg = orientation * (180 / np.pi)
    out = np.zeros(orientations)
    for i in range(orientations):
        m = (deg >= 360 / orientations * i) & (deg < 360 / orientations * (i + 1)) & (magnitude >= motion_threshold)
        out[i] = np.where(m, magnitude, 0).mean()
    return out


def extract_velocity():
    """feature_extraction.extract_velocity (8-bin histogram of flow magnitude by direction); falls
    back to velocity_histogram (identical result) on numpy >= 2, where AI-VAD's version fails."""
    _path()
    if "models" in sys.modules and not hasattr(sys.modules["models"], "CLIP"):
        del sys.modules["models"]
    try:
        fn = importlib.import_module("feature_extraction").extract_velocity
    except ImportError:                                    # `clip` not installed (model 4):
        sys.modules.setdefault("models", types.SimpleNamespace(CLIP=None))
        fn = importlib.import_module("feature_extraction").extract_velocity
    try:
        f = np.random.RandomState(0).randn(8, 8, 2).astype(np.float32)
        fn(f, np.abs(f).sum(-1), np.abs(f[..., 0]), orientations=8)
        return fn
    except ValueError:
        return velocity_histogram


def video_dataset():
    _path()
    return importlib.import_module("video_dataset")


def evaluate_functions():
    """evaluate.py: gaussian_video (temporal smoothing) and macro_auc (per-video AUC)."""
    _path()
    m = importlib.import_module("evaluate")
    return m.gaussian_video, m.macro_auc
