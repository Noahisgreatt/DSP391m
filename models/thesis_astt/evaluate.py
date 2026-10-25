"""Evaluate the thesis model like the repo's test_all_scenes, on our ShanghaiTech test videos.

    python -m models.thesis_astt.evaluate --run runs/thesis_astt/<run_id>

Per frame: MSE between predicted and real frame (repo score). Labels aligned like the repo
(the first 4 frames have no prediction). Metrics from src/evaluate.py: auc_raw is what the repo
prints ("AUC@1"); auc_norm / eer are the per-video-normalised versions most papers report.
Writes results/thesis_astt/<run_id>/metrics_test.json and
results/thesis_astt/<run_id>/scores/<video>.npz (used by src.compare).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.metrics import frame_metrics
from models.thesis_astt.model import build_thesis_model
from models.thesis_astt.train import score_video
from src.utils import results_dir


def load_thesis_run(run: Path, device, which="best.pth"):
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    split = json.loads((run / "split.json").read_text())
    model = build_thesis_model(cfg).to(device)
    ck = torch.load(run / which, map_location=device)
    model.load_state_dict(ck["state_dict"])
    return cfg, split, model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    run = Path(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, split, model, ck = load_thesis_run(run, device, args.checkpoint)
    vids = split[args.split][: args.limit]
    print(f"{args.split}: {len(vids)} videos, checkpoint epoch {ck.get('epoch')}, {device}")
    scores = {}
    for k, v in enumerate(vids, 1):
        scores[v] = score_video(model, cfg, "shanghai", v, device)
        print(f"  [{k}/{len(vids)}] {v}")
    events_dir = Path(cfg["data"]["features_dir"]) / "events" / "shanghai"
    res, per_video = frame_metrics(cfg, scores, events_dir if events_dir.exists() else None)
    res.update({"split": args.split, "checkpoint": args.checkpoint, "epoch": ck.get("epoch")})
    out = results_dir(run) / "scores"
    out.mkdir(parents=True, exist_ok=True)
    for vid, (m, n, lab) in per_video.items():
        np.savez_compressed(out / f"{vid}.npz", mse=m, score_norm=n, labels=lab)
    (results_dir(run) / f"metrics_{args.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
