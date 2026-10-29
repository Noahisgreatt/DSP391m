"""Step 4, EVALUATION: how good is each model, measured the same way for all of them.

    python -m src.evaluate --config configs/astt_baseline.yaml     # one model on the 107 test videos
    python -m src.evaluate --compare                               # all evaluated models + the rules
    python -m src.evaluate --benchmark                             # size and speed of every model
    (--smoke after a smoke training run, --limit N for only N test videos)

Score per frame: models 1-2 = prediction error (MSE); models 3-4 = MULDE's negative log-likelihood.
Higher = more abnormal for all.
  auc_raw    AUC of all test frames pooled ("micro"; what the thesis repo and MULDE report)
  auc_norm   AUC after normalising the score inside each video (videos differ in brightness and
             crowd size, so raw scores of different videos aren't comparable). This is why the
             old ASTT run gave 0.518 raw but 0.616 normalised with the SAME model.
  auc_macro  mean of the per-video AUCs ("macro", the second number MULDE reports)
  eer        equal error rate (lower is better); threshold_raw_at_eer is used by the demo / API
Out: outputs/evaluation/<model>/metrics.json, roc.csv, scores/<video>.npz
     outputs/comparison/comparison.md|json, roc.png, benchmark.md|json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils import frames_dir, load_config, out_dir

MODELS = {"astt_baseline": "Model 1: ASTT (thesis, unchanged)", "astt_improved": "Model 2: ASTT improved (ours)",
          "mulde": "Model 3: MULDE (Micorek et al.)", "mulde_improved": "Model 4: MULDE improved (ours)"}


# ====================================================================== metrics
def psnr_norm(mse):
    """Per-video score in [0, 1] from PSNR (pixels in [-1, 1]); 1 = most abnormal."""
    p = 10 * np.log10(4.0 / np.maximum(mse, 1e-10))
    return 1.0 - (p - np.nanmin(p)) / max(np.nanmax(p) - np.nanmin(p), 1e-8)


def minmax(s):
    return (s - np.nanmin(s)) / max(np.nanmax(s) - np.nanmin(s), 1e-12)


def eer_point(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def macro_auc(per_video):
    """Macro AUC as in Georgescu et al. 2021 (used by MULDE): per-video AUC with one dummy normal (0)
    and one dummy abnormal (max) frame, so videos with a single class still count."""
    return float(np.mean([roc_auc_score(np.concatenate(([0], y, [1])), np.concatenate(([0], s, [np.finfo(np.float64).max])))
                          for s, y in per_video]))


def frame_metrics(scores, labels, kind="mse"):
    """scores: video -> score per frame (NaN = no prediction, e.g. ASTT's first 4 frames; they get
    the video's lowest score, like the repo). -> (metrics, {video: (score, normalised, labels)})."""
    ys, raw, norm, per_video = [], [], [], {}
    for vid, s in scores.items():
        if np.isnan(s).all():
            continue
        m = np.where(np.isnan(s), np.nanmin(s), s)
        n = psnr_norm(m) if kind == "mse" else minmax(m)
        ys.append(labels[vid]); raw.append(m); norm.append(n)
        per_video[vid] = (m, n, labels[vid])
    y, r, n = np.concatenate(ys), np.concatenate(raw), np.concatenate(norm)
    res = {"videos": len(per_video), "frames": int(len(y)), "abnormal_frames": int(y.sum())}
    if 0 < y.sum() < len(y):
        res.update(auc_raw=float(roc_auc_score(y, r)), auc_norm=float(roc_auc_score(y, n)),
                   auc_macro=macro_auc([(p[0], p[2]) for p in per_video.values()]))
        res["eer"], res["threshold_norm_at_eer"] = eer_point(y, n)
        res["eer_raw"], res["threshold_raw_at_eer"] = eer_point(y, r)
    return res, per_video


def binary_metrics(y, flags):
    """For yes/no detectors (the behavior rules): one operating point."""
    tpr = float((flags & (y == 1)).sum() / max((y == 1).sum(), 1))
    fpr = float((flags & (y == 0)).sum() / max((y == 0).sum(), 1))
    out = {"frames": int(len(y)), "abnormal_frames": int(y.sum()), "tpr": tpr, "fpr": fpr, "eer": (fpr + 1 - tpr) / 2}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, flags.astype(float)))
    return out


# ====================================================================== one model
def evaluate(config, smoke=False, limit=None, max_hours=None):
    from src.data import load_split, video_labels
    from src.models import scorer as make_scorer
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = out_dir(cfg, "training", smoke) / "model.pth"
    scorer = make_scorer(cfg, ckpt, device, smoke)
    vids = load_split(cfg)["test"][: (1 if smoke else limit)]
    print(f"{cfg['model']['name']}: {len(vids)} test videos on {device}")
    t0, scores = time.time(), {}
    for k, v in enumerate(vids, 1):
        if max_hours and time.time() - t0 > 3600 * max_hours:     # models 3/4 cache each video: run again
            return print(f"stopped after {max_hours} h at video {k}/{len(vids)}: run again to continue")
        scores[v] = scorer(frames_dir(cfg, "shanghaitech") / v, v)
        if k % 10 == 0 or k == len(vids):
            print(f"  [{k}/{len(vids)}] {time.time() - t0:.0f}s")
    res, per_video = frame_metrics(scores, {v: video_labels(cfg, v, len(s)) for v, s in scores.items()}, scorer.kind)
    res.update(model=cfg["model"]["name"], seconds=round(time.time() - t0),
               fps_scored=round(res["frames"] / max(time.time() - t0, 1e-9), 1))
    out = out_dir(cfg, "evaluation", smoke)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    for v, (m, n, y) in per_video.items():
        np.savez_compressed(out / "scores" / f"{v}.npz", score=m, score_norm=n, labels=y)
    if "auc_raw" in res:
        y = np.concatenate([p[2] for p in per_video.values()])
        fpr, tpr, _ = roc_curve(y, np.concatenate([p[0] for p in per_video.values()]))
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 500)).astype(int))
        with open(out / "roc.csv", "w", newline="") as f:
            csv.writer(f).writerows([["fpr", "tpr"]] + [[round(float(fpr[i]), 5), round(float(tpr[i]), 5)] for i in keep])
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ====================================================================== all models side by side
def compare():
    cfg = load_config()
    root = Path(cfg["paths"]["outputs"])
    scores = {m: {f.stem: np.load(f) for f in sorted((root / "evaluation" / m / "scores").glob("*.npz"))}
              for m in MODELS if (root / "evaluation" / m / "scores").exists()}
    if not scores:
        raise SystemExit("No evaluated model yet: run python -m src.evaluate --config ... first.")
    ev = root / "behavior" / "shanghaitech" / "events"
    rules = {f.stem: np.load(f)["frame_flags"] for f in sorted(ev.glob("*.npz"))} if ev.exists() else {}
    common = set.intersection(*(set(s) for s in scores.values()))
    common = sorted(common & set(rules)) if rules else sorted(common)
    first = next(iter(scores.values()))
    y = np.concatenate([first[v]["labels"] for v in common])
    rows, curves = [], []
    for name, s in scores.items():
        raw, norm = np.concatenate([s[v]["score"] for v in common]), np.concatenate([s[v]["score_norm"] for v in common])
        run, ev_m = root / "training" / name / "run.json", root / "evaluation" / name / "metrics.json"
        p = json.loads(run.read_text())["parameters"].get("Total") if run.exists() else None
        rows.append({"key": name, "method": MODELS[name], "parameters_M": round(p / 1e6, 1) if p else None,
                     "auc_raw": float(roc_auc_score(y, raw)), "auc_norm": float(roc_auc_score(y, norm)),
                     "auc_macro": macro_auc([(s[v]["score"], s[v]["labels"]) for v in common]), "eer": eer_point(y, norm)[0],
                     "fps_scored": json.loads(ev_m.read_text()).get("fps_scored") if ev_m.exists() else None})
        curves.append((MODELS[name], *roc_curve(y, raw)[:2], rows[-1]["auc_raw"]))
    point = None
    if rules:
        b = np.concatenate([np.pad(rules[v], (0, max(0, len(first[v]["labels"]) - len(rules[v]))))[:len(first[v]["labels"])] > 0
                            for v in common])
        m = binary_metrics(y, b)
        rows.append({"key": "rules", "method": "Behavior rules (no training)", "parameters_M": 0, "auc_raw": m.get("auc"),
                     "auc_norm": m.get("auc"), "auc_macro": None, "eer": m["eer"], "fps_scored": None})
        point = (m["fpr"], m["tpr"], m.get("auc"))
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({"test_videos": len(common), "frames": int(len(y)),
                                                     **{r["key"]: {k: v for k, v in r.items() if k not in ("key", "method")} for r in rows}}, indent=2))
    pct = lambda v: "-" if v is None else f"{v * 100:.1f}%"
    md = [f"# Comparison on {len(common)} test videos, {len(y):,} frames ({int(y.sum()):,} abnormal)", "",
          "| Method | Parameters | AUC micro | AUC per-video norm | AUC macro | EER | Frames/s |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['method']} | {r['parameters_M']}M | {pct(r['auc_raw'])} | {pct(r['auc_norm'])} | {pct(r['auc_macro'])} | "
           f"{pct(r['eer'])} | {r['fps_scored'] or '-'} |" for r in rows]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=150)
    ax.plot([0, 1], [0, 1], color="#e4e3dd", lw=1, ls="--")
    for (name, fpr, tpr, auc), c in zip(curves, ["#2a78d6", "#eb6834", "#7a5cd6", "#d64f8c"]):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {auc * 100:.1f}%")
    if point:
        ax.plot(point[0], point[1], "o", ms=9, color="#1baf7a", label=f"Behavior rules  AUC {point[2] * 100:.1f}%")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Frame-level ROC, {len(common)} test videos", loc="left")
    ax.grid(color="#e4e3dd", lw=0.6); ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc.png")
    print("\n".join(md))


# ====================================================================== size and speed
def benchmark(models, n_frames):
    from src.data import load_split
    base = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vid = load_split(base)["test"][0]
    vdir = frames_dir(base, "shanghaitech") / vid
    sync = lambda: torch.cuda.synchronize() if device.type == "cuda" else None
    res = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "video": vid, "frames": n_frames, "models": {}}
    for name in models:
        cfg = load_config(f"configs/{name}.yaml")
        ckpt = out_dir(cfg, "training") / "model.pth"
        if not ckpt.exists():
            print(f"{name}: not trained yet, skipped")
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if cfg["model"].get("family") == "mulde":
            from src.models.mulde import Features, Scorer, levels, parameters
            sc = Scorer(cfg, ckpt, device)
            sc.features = Features(cfg, device)
            sc.features.video(vdir, vid, None, 1, 20)                    # warm-up
            sync(); t0 = time.time()
            _, X = sc.features.video(vdir, vid, None, 1, n_frames)
            sync(); t1 = time.time()
            sc.gmms["all"].score_samples(levels(sc.net, ((X - sc.mean) / sc.std).astype(np.float32), sc.sigmas, device))
            ms = {"features": round((t1 - t0) / len(X) * 1000, 2), "density": round((time.time() - t1) / len(X) * 1000, 2)}
            params = parameters(cfg)
        else:
            from src.data import video_dataset
            from src.models.astt import autocast, count_parameters, load_trained
            model = load_trained(cfg, ckpt, device)
            ds, _ = video_dataset(cfg, vdir)
            x = torch.as_tensor(ds[0]["standard"][:4]).unsqueeze(0).to(device)
            with torch.no_grad(), autocast(cfg, device):
                for _ in range(3):
                    model(x)
                sync(); t0 = time.time()
                for _ in range(min(n_frames, 50)):
                    model(x)
                sync()
            ms = {"network": round((time.time() - t0) / min(n_frames, 50) * 1000, 2)}
            params = count_parameters(model)
        total = sum(ms.values())
        res["models"][name] = {"parameters": params, "ms_per_frame": {**ms, "total": round(total, 2)},
                               "fps": round(1000 / max(total, 1e-9), 2),
                               "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
        print(name, res["models"][name])
    out = Path(base["paths"]["outputs"]) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))
    md = [f"# Size and speed ({res['device']}, video {vid}, {n_frames} frames)", "",
          "| Model | Parameters | ms / frame | Frames / s | Peak GPU memory |", "|---|---|---|---|---|"]
    md += [f"| {k} | {r['parameters']['Total'] / 1e6:.1f}M | {', '.join(f'{a} {b}' for a, b in r['ms_per_frame'].items())} | "
           f"{r['fps']} | {r['peak_gpu_memory_mb'] or '-'} MB |" for k, r in res["models"].items()]
    (out / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-hours", type=float, help="stop after this many hours (Kaggle)")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.benchmark:
        benchmark(a.models, a.frames)
    elif a.config:
        evaluate(a.config, a.smoke, a.limit, a.max_hours)
    else:
        ap.error("give --config, --compare or --benchmark")


if __name__ == "__main__":
    main()
