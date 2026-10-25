"""Settings and paths. Every script reads configs/data.yaml, then the YAML files it is given.

    cfg = load_config("configs/astt_baseline.yaml")       # data.yaml + model settings
    cfg = load_config("configs/behavior.yaml")            # data.yaml + behavior settings
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = "configs/data.yaml"
DATASETS = ("shanghaitech", "manual", "upload")   # upload = one-off videos (demo / API)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(*paths: str | Path) -> dict:
    """configs/data.yaml merged with the given YAML files (later files win)."""
    cfg: dict = {}
    for p in (DATA_CONFIG, *paths):
        if p:
            cfg = _deep_merge(cfg, yaml.safe_load(Path(p).read_text()) or {})
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, check=True).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------- paths
def frames_dir(cfg: dict, dataset: str) -> Path:
    """Folder with one sub-folder of frames per video."""
    return Path(cfg["paths"]["processed"]) / dataset / "frames"


def labels_dir(cfg: dict, dataset: str = "shanghaitech") -> Path:
    return Path(cfg["paths"]["processed"]) / dataset / "labels"


def split_file(cfg: dict) -> Path:
    return Path(cfg["paths"]["metadata"]) / "split.json"


def features_path(cfg: dict, kind: str, dataset: str, video: str) -> Path:
    """data/features/<kind>/<dataset>/<video>.npz   kind = tracking | optical_flow | normal_maps"""
    return Path(cfg["paths"]["features"]) / kind / dataset / f"{video}.npz"


def training_dir(cfg: dict, smoke: bool = False) -> Path:
    return Path(cfg["paths"]["outputs"]) / "training" / (cfg["model"]["name"] + ("_smoke" if smoke else ""))


def evaluation_dir(cfg: dict, smoke: bool = False) -> Path:
    return Path(cfg["paths"]["outputs"]) / "evaluation" / (cfg["model"]["name"] + ("_smoke" if smoke else ""))


def behavior_dir(cfg: dict, dataset: str) -> Path:
    return Path(cfg["paths"]["outputs"]) / "behavior" / dataset


def video_fps(cfg: dict, dataset: str, video: str) -> float:
    """Frames per second of the stored frames (manual/upload videos keep theirs in meta.json)."""
    if dataset == "shanghaitech":
        return float(cfg["shanghaitech"]["fps"])
    meta = frames_dir(cfg, dataset) / video / "meta.json"
    if meta.exists():
        fps = json.loads(meta.read_text()).get("fps")
        if fps:
            return float(fps)
    return float(cfg["manual"]["fps_fallback"])
