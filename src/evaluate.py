"""Frame-level evaluation (AUC-ROC, EER as in the thesis, plus PR-AUC and best F1).

Model run:
    python -m src.evaluate --model vit_flow --run latest --dataset shanghaitech
Rule-only baselines (the notebook approach, now over ALL test videos, not one clip):
    python -m src.evaluate --baseline tracking  --dataset shanghaitech
    python -m src.evaluate --baseline farneback --dataset shanghaitech

Frame score (model) = alpha * per-video min-max normalized prediction error
                      + (1 - alpha) * P(loitering or counter-flow)
Prediction error = max over 8x8 patches of the squared flow error, so a single person moving
the wrong way is not averaged away by the rest of the image.
Results go into metrics.json of the run ("test_<dataset>"), per-video scores into scores/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FlowSequenceDataset
from src.models import build_model
from src.utils import load_params, resolve, save_json, split_of


def frame_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"error": "ground truth has a single class"}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    eer = float((fpr[i] + 1 - tpr[i]) / 2)
    p, r, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {"auc": float(roc_auc_score(y_true, y_score)), "eer": eer,
            "ap": float(average_precision_score(y_true, y_score)), "best_f1": float(f1.max()),
            "n_frames": int(len(y_true)), "anomaly_ratio": float(y_true.mean())}


def fill_series(values: np.ndarray, frames: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan, np.float32)
    out[frames] = values
    valid = np.flatnonzero(~np.isnan(out))
    if len(valid) == 0:
        return np.zeros(n, np.float32)
    return np.interp(np.arange(n), valid, out[valid]).astype(np.float32)


def minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def gt_for(params: dict, dataset: str, vid: str, n: int) -> np.ndarray | None:
    p = resolve(params["paths"]["frames"]) / dataset / "_gt" / f"{vid}.npy"
    if not p.exists():
        return None
    g = np.load(p).astype(int).ravel()
    return np.pad(g, (0, max(0, n - len(g))))[:n]


@torch.no_grad()
def score_video(model, ds: FlowSequenceDataset, cfg: dict, device, batch_size: int) -> dict:
    v = ds.videos[0]
    errs, probs, frames = [], [], []
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        out = model(b["x"].to(device), out_size=b["y_flow"].shape[-2:])
        e = (out["flow"].float() - b["y_flow"].to(device)).pow(2).sum(1, keepdim=True)
        errs.append(F.avg_pool2d(e, 8).flatten(1).max(1).values.cpu())
        probs.append(1 - out["logits"].float().softmax(1)[:, 0].cpu())
        frames.append(b["frame"])
    err = torch.cat(errs).numpy()
    prob = torch.cat(probs).numpy()
    fr = torch.cat(frames).numpy()
    n = v["n_frames"]
    alpha = cfg["eval"]["alpha"]
    e_full, p_full = minmax(fill_series(err, fr, n)), fill_series(prob, fr, n)
    score = alpha * e_full + (1 - alpha) * p_full
    k = cfg["eval"].get("smooth_frames", 1)
    if k > 1:
        score = median_filter(score, size=k, mode="nearest")
    return {"score": score, "error": e_full, "prob": p_full}


def resolve_run(params: dict, model_name: str, run: str) -> Path:
    base = resolve(params["paths"]["models"]) / model_name
    if run == "latest":
        run = (base / "latest.txt").read_text().strip()
    return base / run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shanghaitech")
    ap.add_argument("--model", help="Model name, e.g. vit_flow")
    ap.add_argument("--run", default="latest")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--baseline", choices=["tracking", "farneback"], help="Evaluate rule-only pseudo-label scores.")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    params = load_params()

    if args.baseline:
        label_dir = resolve(params["paths"]["labels"]) / args.baseline / args.dataset
        ys, ss, per = [], [], {}
        for sp in sorted(label_dir.glob("*.score.npy")):
            vid = sp.name[: -len(".score.npy")]
            if split_of(vid, params, args.dataset) != "test":
                continue
            s = np.load(sp)
            g = gt_for(params, args.dataset, vid, len(s))
            if g is None:
                continue
            ys.append(g); ss.append(s)
            per[vid] = frame_metrics(g, s).get("auc")
        if not ys:
            raise SystemExit("No test videos with ground truth found.")
        res = {"overall": frame_metrics(np.concatenate(ys), np.concatenate(ss)), "per_video_auc": per}
        out = resolve(params["paths"]["models"]) / f"baseline_{args.baseline}" / f"metrics_{args.dataset}.json"
        save_json(res, out)
        print(json.dumps(res["overall"], indent=2)); print("->", out)
        return

    run_dir = resolve_run(params, args.model, args.run)
    ckpt = torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["data"]["dataset"] = args.dataset
    cfg["data"]["label_method"] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["model"]["pretrained"] = False
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    feat_dir = resolve(params["paths"]["features"]) / cfg["data"]["feature_method"] / args.dataset
    vids = [f.stem for f in sorted(feat_dir.glob("*.npz")) if split_of(f.stem, params, args.dataset) == "test"]
    score_dir = run_dir / "scores" / args.dataset
    score_dir.mkdir(parents=True, exist_ok=True)

    ys, ss, per = [], [], {}
    for vid in tqdm(vids, desc="evaluate"):
        ds = FlowSequenceDataset(params, cfg, video_ids=[vid], stride=1)
        r = score_video(model, ds, cfg, device, args.batch_size)
        np.savez(score_dir / f"{vid}.npz", **r)
        g = gt_for(params, args.dataset, vid, len(r["score"]))
        if g is not None:
            ys.append(g); ss.append(r["score"])
            per[vid] = frame_metrics(g, r["score"]).get("auc")

    mpath = run_dir / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    res = {"overall": frame_metrics(np.concatenate(ys), np.concatenate(ss)) if ys else "no ground truth",
           "per_video_auc": per, "ckpt": args.ckpt}
    metrics[f"test_{args.dataset}"] = res
    save_json(metrics, mpath)
    print(json.dumps(res["overall"], indent=2) if ys else "Scores saved (no GT).", "\n->", mpath)


if __name__ == "__main__":
    main()
