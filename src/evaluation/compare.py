"""Compare the 4 models and the behavior rules on the SAME test frames.

    python -m src.evaluation.compare                       (DVC stage: compare)

In:   outputs/evaluation/<model>/scores/*.npz, outputs/behavior/shanghaitech/events/*.npz
Out:  outputs/comparison/comparison.md    table for the report
      outputs/comparison/comparison.json  same numbers (DVC metrics: `dvc metrics diff`)
      outputs/comparison/roc.png          ROC curves
Only videos scored by every method are used, so the numbers are like-for-like.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import load_config
from src.evaluation.metrics import binary_metrics, eer_point, macro_auc

MODELS = {"astt_baseline": "Model 1: ASTT (thesis, unchanged)",
          "astt_improved": "Model 2: ASTT improved (ours)",
          "aivad": "Model 3: AI-VAD (Reiss & Hoshen)",
          "aivad_lite": "Model 4: AI-VAD Lite (ours)"}
OUTPUT = {"astt_baseline": "frame score + error heat-map", "astt_improved": "frame score + error heat-map",
          "aivad": "frame score + unusual objects + reason", "aivad_lite": "frame score + unusual objects + reason"}
COLORS = ["#2a78d6", "#eb6834", "#7a5cd6", "#d64f8c", "#1baf7a"]
INK, MUTED, GRID = "#28261f", "#6b6a63", "#e4e3dd"


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["outputs"])
    scores = {}
    for name in MODELS:
        d = root / "evaluation" / name / "scores"
        if d.exists():
            scores[name] = {f.stem: np.load(f) for f in sorted(d.glob("*.npz"))}
    ev_dir = root / "behavior" / "shanghaitech" / "events"
    rules = {f.stem: np.load(f)["frame_flags"] for f in sorted(ev_dir.glob("*.npz"))} if ev_dir.exists() else {}
    if not scores:
        raise SystemExit("No evaluated model yet: run src.evaluation.evaluate first.")
    common = sorted(set.intersection(*(set(s) for s in scores.values()))
                    & (set(rules) if rules else set.union(*(set(s) for s in scores.values()))))
    if not common:
        raise SystemExit("The methods share no scored videos.")
    first = next(iter(scores.values()))
    y = np.concatenate([first[v]["labels"] for v in common])

    rows, curves = [], []
    for k, (name, s) in enumerate(scores.items()):
        norm = np.concatenate([s[v]["score_norm"] for v in common])
        raw = np.concatenate([s[v]["score"] for v in common])
        ev = root / "evaluation" / name / "metrics.json"
        tr = root / "training" / name / "run.json"
        rows.append({"method": MODELS[name], "key": name,
                     "parameters_M": round(json.loads(tr.read_text())["parameters"]["Total"] / 1e6, 1) if tr.exists() else None,
                     "auc_norm": float(roc_auc_score(y, norm)), "auc_raw": float(roc_auc_score(y, raw)),
                     "auc_macro": macro_auc([(s[v]["score"], s[v]["labels"]) for v in common]),
                     "eer": eer_point(y, norm)[0],
                     "fps_scored": json.loads(ev.read_text()).get("fps_scored") if ev.exists() else None,
                     "output": OUTPUT[name]})
        fpr, tpr, _ = roc_curve(y, raw)
        curves.append((MODELS[name], fpr, tpr, rows[-1]["auc_raw"], COLORS[k]))
    point = None
    if rules:
        b = np.concatenate([np.pad(rules[v], (0, max(0, len(first[v]["labels"]) - len(rules[v]))))
                            [: len(first[v]["labels"])] > 0 for v in common])
        m = binary_metrics(y, b)
        rows.append({"method": "Behavior rules (no training)", "key": "rules", "parameters_M": 0,
                     "auc_norm": m.get("auc"), "auc_raw": m.get("auc"), "auc_macro": None, "eer": m["eer"],
                     "fps_scored": None, "output": "person + reason (loitering / counter-flow)"})
        point = (m["fpr"], m["tpr"], m.get("auc"))

    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({
        "test_videos": len(common), "frames": int(len(y)), "abnormal_frames": int(y.sum()),
        **{r["key"]: {k: r[k] for k in ("auc_raw", "auc_norm", "auc_macro", "eer", "parameters_M", "fps_scored")}
           for r in rows}},
        indent=2))
    pct = lambda v: "" if v is None else f"{v * 100:.1f}%"
    md = [f"# Comparison on {len(common)} test videos, {len(y):,} frames ({int(y.sum()):,} abnormal)", "",
          "| Method | Parameters | AUC micro (raw) | AUC (per-video norm) | AUC macro | EER | Frames/s | Output |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['method']} | {r['parameters_M']}M | {pct(r['auc_raw'])} | {pct(r['auc_norm'])} | "
           f"{pct(r['auc_macro'])} | {pct(r['eer'])} | {r['fps_scored'] or '-'} | {r['output']} |" for r in rows]
    md += ["", "AUC micro: all test frames pooled, raw scores (thesis repo; AI-VAD's main number). "
           "AUC per-video norm: scores min-max normalised inside each video first. "
           "AUC macro: mean of per-video AUCs (AI-VAD's second number). EER: lower is better. "
           "Frames/s: whole scoring pipeline during evaluation (benchmark.md has clean per-network timings). "
           "The rules answer yes/no, so their AUC is one operating point."]
    (out / "comparison.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=150)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for name, fpr, tpr, auc, c in curves:
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {auc * 100:.1f}%")
    if point:
        ax.plot(point[0], point[1], "o", ms=9, color=COLORS[2], mec="white", mew=2,
                label=f"Behavior rules  AUC {point[2] * 100:.1f}%")
    ax.set_xlabel("False positive rate (normal frames flagged)", color=MUTED)
    ax.set_ylabel("True positive rate (abnormal frames caught)", color=MUTED)
    ax.set_title(f"Frame-level ROC (micro), {len(common)} test videos", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
