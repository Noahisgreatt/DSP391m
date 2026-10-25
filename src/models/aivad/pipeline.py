"""Train (fit) and score models 3 / 4 with the same commands as models 1 / 2:

    python -m src.training.train      --config configs/aivad.yaml        (or aivad_lite.yaml)
    python -m src.evaluation.evaluate --config configs/aivad.yaml

"Training" = attributes of every object in the 330 normal training videos (cached per video in
data/features/<model>/train/, so a stopped run continues where it was; test videos are cached
in data/features/<model>/test/ by the evaluation) -> fit the density estimators
and calibration -> outputs/training/<model>/model.pth.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import config_hash, frames_dir, git_commit, training_dir
from src.data.split import load_split
from src.models.aivad import released
from src.models.aivad.density import AttributeScorer
from src.models.aivad.networks import n_params


def feature_dir(cfg, split: str, smoke=False) -> Path:
    """data/features/<model>/<train|test>/  (two folders, so train and test are separate DVC outputs)"""
    return Path(cfg["paths"]["features"]) / (cfg["model"]["name"] + ("_smoke" if smoke else "")) / split


def count_parameters(extractor) -> dict:
    out = {f"{k} ({type(v).__name__})": n_params(v) for k, v in extractor.networks().items()}
    out["Total"] = sum(out.values())
    out["Trainable"] = 0                       # nothing is trained: pretrained networks + statistics
    return out


def fit(cfg: dict, smoke: bool, fresh: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = training_dir(cfg, smoke)
    if fresh and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    split = load_split(cfg)
    vids = split["train"][:2] if smoke else split["train"]
    src = cfg["model"]["features"]
    t0 = time.time()
    params = None
    if src == "released":
        feats = released.load(cfg, "train", vids)
    else:
        from src.models.aivad.features import Extractor
        ex = Extractor(cfg, device)
        params = count_parameters(ex)
        print(f"{cfg['model']['name']}: {params['Total']:,} parameters in its networks, on {device}")
        feats, log = {}, []
        for k, v in enumerate(vids, 1):
            f = ex.video(frames_dir(cfg, "shanghaitech") / v, feature_dir(cfg, "train", smoke) / f"{v}.npz",
                         max_frames=30 if smoke else None)
            feats[v] = f
            fps = float(f["n_frames"]) / max(float(f["seconds"]), 1e-9)
            log.append([k, v, int(f["n_frames"]), len(f["frame"]), round(fps, 2), round((time.time() - t0) / 60, 2)])
            print(f"[{k}/{len(vids)}] {v}: {int(f['n_frames'])} frames, {len(f['frame'])} objects, "
                  f"{len(f['pose'])} poses ({fps:.1f} fps)")
        with open(out / "log.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "video", "frames", "objects", "fps", "minutes"])
            w.writerows(log)
    scorer = AttributeScorer(cfg)
    print("fitting density estimators + leave-one-video-out calibration ...")
    calib = scorer.fit(feats)
    torch.save({"scorer": scorer.state_dict(), "config": cfg}, out / "model.pth")
    n_obj = sum(len(f["frame"]) for f in feats.values())
    (out / "run.json").write_text(json.dumps({
        "model": cfg["model"]["name"], "config_hash": config_hash({k: cfg[k] for k in ("model", "scoring")}),
        "git_commit": git_commit(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device), "features": src, "parameters": params or {"Total": cfg["model"].get("parameters_total")},
        "split": {k: len(v) for k, v in split.items()}, "config": cfg}, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps({
        "train_videos": len(feats), "train_frames": int(sum(int(f["n_frames"]) for f in feats.values())),
        "train_objects": int(n_obj), "train_poses": int(sum(len(f["pose"]) for f in feats.values())),
        "calibration": calib, "minutes": round((time.time() - t0) / 60, 1),
        "parameters_total": (params or {}).get("Total")}, indent=2))
    print(f"done: {out}/model.pth")


class VideoScorer:
    """Score one video with a fitted model 3 / 4 (features are cached in data/features/<model>/)."""
    kind = "score"

    def __init__(self, cfg: dict, checkpoint: Path, device, extractor=None, smoke=False):
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg, self.device = cfg, device
        self.scorer = AttributeScorer.from_state(sd["scorer"], cfg)
        self.extractor = extractor
        self._released = None
        self.smoke = smoke

    def features(self, video_dir: Path, vid: str, cache=True, max_frames=None):
        if self.cfg["model"]["features"] == "released" and vid in load_split(self.cfg)["test"]:
            if self._released is None:
                self._released = released.load(self.cfg, "test", load_split(self.cfg)["test"])
            return self._released[vid]
        if self.extractor is None:
            from src.models.aivad.features import Extractor
            self.extractor = Extractor(self.cfg, self.device)
        if cache and vid in load_split(self.cfg)["test"]:
            cache_file = feature_dir(self.cfg, "test", self.smoke) / f"{vid}.npz"
        else:
            cache_file = None
        return self.extractor.video(video_dir, cache_file, max_frames=30 if self.smoke else max_frames)

    def __call__(self, video_dir: Path, vid: str, cache=True, details=False, max_frames=None, **_):
        f = self.features(Path(video_dir), vid, cache, max_frames)
        r = self.scorer.score(vid, f)
        return (r["score"], {**r, "features": f}) if details else r["score"]
