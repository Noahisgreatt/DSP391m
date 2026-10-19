"""Compare the thesis ASTT, CrowdFormer and the rule baseline on the SAME test frames.

    python -m src.compare                                   # latest run of each model
    python -m src.compare --runs runs/thesis_astt/<id> runs/crowdformer/<id>

Needs `models.thesis_astt.evaluate` / `models.crowdformer.evaluate` to have been run (they write per-frame scores).
Only videos scored by every model are used, so the numbers are like-for-like.
Writes results/comparison/comparison.md, comparison.csv and roc.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils import load_config, results_dir

# validated categorical palette (dataviz reference): blue, orange, aqua; text stays neutral
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
STYLES = ["-", "--", ":"]
INK, MUTED, GRID = "#28261f", "#6b6a63", "#e4e3dd"
LEVEL = {"thesis_astt": "frame (red heat-map shows where)", "crowdformer": "person (boxes the person)"}
LABEL = {"thesis_astt": "ASTT (thesis, unchanged)", "crowdformer": "CrowdFormer (own model)"}


def latest(name):
    runs = sorted(p for p in Path("runs", name).glob("*") if (results_dir(p) / "metrics_test.json").exists())
    return runs[-1] if runs else None


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs] if args.runs else [
        r for r in (latest("thesis_astt"), latest("crowdformer")) if r]
    if not runs:
        raise SystemExit("No evaluated runs found. Run models.thesis_astt.evaluate / models.crowdformer.evaluate first.")

    data = {}
    for r in runs:
        sdir = results_dir(r) / "scores"
        data[r] = {f.stem: np.load(f) for f in sorted(sdir.glob("*.npz"))}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    if not common:
        raise SystemExit("The runs share no scored videos.")

    cfg = load_config()
    ev_dir = Path(cfg["data"]["features_dir"]) / "events" / "shanghai"
    y = np.concatenate([data[runs[0]][v]["labels"] for v in common])
    rows, curves = [], []
    for k, r in enumerate(runs):
        raw = np.concatenate([data[r][v]["mse"] for v in common])
        norm = np.concatenate([data[r][v]["score_norm"] for v in common])
        params = json.loads((r / "params.json").read_text())["Total"] if (r / "params.json").exists() else None
        name = LABEL.get(r.parent.name, r.parent.name)
        rows.append({"model": name, "run": r.name, "params_M": round(params / 1e6, 2) if params else "",
                     "auc_norm": roc_auc_score(y, norm), "auc_raw": roc_auc_score(y, raw),
                     "eer": eer(y, norm), "level": LEVEL.get(r.parent.name, "")})
        fpr, tpr, _ = roc_curve(y, norm)
        curves.append((name, fpr, tpr, rows[-1]["auc_norm"], COLORS[k % 3], STYLES[k % 3]))

    point = None
    if all((ev_dir / f"{v}.npz").exists() for v in common):
        flags = []
        for v in common:
            ff = np.load(ev_dir / f"{v}.npz")["frame_flags"]
            n = len(data[runs[0]][v]["labels"])
            flags.append(np.pad(ff, (0, max(0, n - len(ff))))[:n] > 0)
        b = np.concatenate(flags)
        tpr_b = float((b & (y == 1)).sum() / max((y == 1).sum(), 1))
        fpr_b = float((b & (y == 0)).sum() / max((y == 0).sum(), 1))
        rows.append({"model": "rules (no training)", "run": "", "params_M": 0,
                     "auc_norm": roc_auc_score(y, b), "auc_raw": roc_auc_score(y, b),
                     "eer": (fpr_b + 1 - tpr_b) / 2, "level": "person (with a reason)"})
        point = (fpr_b, tpr_b, rows[-1]["auc_norm"])

    out = Path("results/comparison")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    md = [f"# Model comparison on {len(common)} test videos, {len(y):,} frames "
          f"({int(y.sum()):,} abnormal)", "",
          "| Model | Parameters | Frame AUC (norm) | Frame AUC (raw) | EER | Localises |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params_M']}M | {r['auc_norm']*100:.1f}% | "
                  f"{r['auc_raw']*100:.1f}% | {r['eer']*100:.1f}% | {r['level']} |")
    md += ["", "AUC (norm): scores min-max normalised per video, as in most papers. "
           "AUC (raw): unnormalised. EER: equal error rate (lower is better).",
           "The rules give yes/no answers, so their AUC is a single operating point."]
    (out / "comparison.md").write_text("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--", zorder=1)
    for name, fpr, tpr, auc, c, s in curves:
        ax.plot(fpr, tpr, color=c, lw=2, ls=s, label=f"{name}  AUC {auc*100:.1f}%", zorder=3)
    if point:
        ax.plot(point[0], point[1], "o", ms=9, color=COLORS[2], mec="#fcfcfb", mew=2, zorder=4,
                label=f"rules  AUC {point[2]*100:.1f}%")
    ax.set_xlabel("False positive rate (normal frames flagged)", color=MUTED)
    ax.set_ylabel("True positive rate (abnormal frames caught)", color=MUTED)
    ax.set_title(f"Frame-level ROC, {len(common)} test videos", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=MUTED)
    leg = ax.legend(loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out / "roc.png")
    print("\n".join(md))
    print(f"\nwrote {out}/comparison.md, comparison.csv, roc.png")


if __name__ == "__main__":
    main()
