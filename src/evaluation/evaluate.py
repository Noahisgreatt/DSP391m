"""Evaluate a trained model on the ShanghaiTech test videos (frame-level AUC / EER).

    python -m src.evaluation.evaluate --config configs/astt_baseline.yaml
    python -m src.evaluation.evaluate --config configs/astt_improved.yaml
    python -m src.evaluation.evaluate --config configs/aivad.yaml          (models 3, 4 alike)
    (add --smoke after a smoke training run, --limit N to score only N videos)

In:   outputs/training/<model>/model.pth
Out:  outputs/evaluation/<model>/
        metrics.json        AUC raw (micro) / normalised / macro, EER, thresholds, per-scene AUC
        roc.csv             ROC curve points (DVC plots)
        scores/<video>.npz  score, score_norm, labels per frame (used by compare.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import evaluation_dir, frames_dir, load_config, training_dir
from src.data.split import load_split, video_labels
from src.evaluation.metrics import frame_metrics
from src.evaluation.scoring import build_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N test videos (quick check)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = training_dir(cfg, args.smoke) / "model.pth"
    scorer = build_scorer(cfg, ckpt, device, smoke=args.smoke)
    split = load_split(cfg)
    vids = split["test"][: (1 if args.smoke else args.limit)]
    print(f"{cfg['model']['name']}: {len(vids)} test videos on {device}")

    t0, scores = time.time(), {}
    for k, v in enumerate(vids, 1):
        scores[v] = scorer(frames_dir(cfg, "shanghaitech") / v, v)
        if k % 10 == 0 or k == len(vids):
            print(f"  [{k}/{len(vids)}] {time.time() - t0:.0f}s")
    labels = {v: video_labels(cfg, v, len(s)) for v, s in scores.items()}
    res, per_video = frame_metrics(scores, labels, scorer.kind)

    by_scene = {}                                            # AUC per camera (01..13)
    for v, (m, n, y) in per_video.items():
        by_scene.setdefault(v[:2], []).append((n, y))
    res["auc_norm_per_scene"] = {
        s: float(roc_auc_score(np.concatenate([y for _, y in l]), np.concatenate([n for n, _ in l])))
        for s, l in sorted(by_scene.items())
        if 0 < np.concatenate([y for _, y in l]).sum() < sum(len(y) for _, y in l)}
    res.update({"model": cfg["model"]["name"], "checkpoint_epoch": torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch"),
                "seconds": round(time.time() - t0), "fps_scored": round(res["frames"] / max(time.time() - t0, 1e-9), 1)})

    out = evaluation_dir(cfg, args.smoke)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    for v, (m, n, y) in per_video.items():
        np.savez_compressed(out / "scores" / f"{v}.npz", score=m, score_norm=n, labels=y)
    if "auc_norm" in res:
        y = np.concatenate([p[2] for p in per_video.values()])
        n = np.concatenate([p[1] for p in per_video.values()])
        fpr, tpr, _ = roc_curve(y, n)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 500)).astype(int))
        with open(out / "roc.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fpr", "tpr"])
            w.writerows([[round(float(fpr[i]), 5), round(float(tpr[i]), 5)] for i in keep])
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "auc_norm_per_scene"}, indent=2))


if __name__ == "__main__":
    main()
