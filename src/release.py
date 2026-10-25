"""Write the metadata card of a project version, before tagging it in git.

    python -m src.release v1.0 --note "data + model 1 (thesis ASTT)"
    python -m src.release v2.0 --note "model 2 (improved ASTT)"
    python -m src.release v3.0 --note "model 3 (AI-VAD)"
    python -m src.release v4.0 --note "model 4 (AI-VAD Lite) + full comparison"

Out: releases/<version>.json   everything that defines this version (machine-readable)
     releases/<version>.md     the same as a short readable page
It collects: git commit, data versions (DVC hashes + data cards), every trained model (config,
parameters, weights hash, training time), evaluation metrics, behavior metrics, comparison, and
the software versions. Then tag it (the script prints the exact commands):
     git add releases dvc.lock && git commit -m "release v1.0" && git tag -a v1.0 -m "..." && dvc push
Later:  git checkout v1.0 && dvc pull     brings back exactly that code, data and models.
        dvc metrics diff v1.0 v2.0        what changed in the numbers.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config import git_commit, load_config

MODELS = ("astt_baseline", "astt_improved", "aivad", "aivad_lite")


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def dvc_hashes() -> dict:
    """md5 of every DVC-tracked input / output (data/raw/*.dvc + dvc.lock)."""
    out = {}
    for f in sorted(Path("data/raw").glob("*.dvc")):
        o = yaml.safe_load(f.read_text())["outs"][0]
        out[f"data/raw/{o['path']}"] = {"md5": o.get("md5"), "size": o.get("size"), "files": o.get("nfiles")}
    lock = Path("dvc.lock")
    if lock.exists():
        for stage in (yaml.safe_load(lock.read_text()) or {}).get("stages", {}).values():
            for o in stage.get("outs", []):
                if o["path"].endswith("model.pth"):
                    out[o["path"]] = {"md5": o.get("md5"), "size": o.get("size")}
    return out


def software() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("torch", "timm", "ultralytics", "dvc", "fastapi"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="e.g. v1.0")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = load_config()
    meta, out = Path(cfg["paths"]["metadata"]), Path(cfg["paths"]["outputs"])

    models = {}
    for m in MODELS:
        run = read_json(out / "training" / m / "run.json")
        if not run:
            continue
        train = read_json(out / "training" / m / "metrics.json") or {}
        ev = read_json(out / "evaluation" / m / "metrics.json") or {}
        models[m] = {
            "config_file": run.get("config_file", f"configs/{m}.yaml"), "config_hash": run.get("config_hash"),
            "trained_from_commit": run.get("git_commit"), "started_utc": run.get("started_utc"),
            "parameters": run["parameters"], "epochs_done": train.get("epochs_done"),
            "final_train_loss": train.get("final_train_loss"),
            "test": {k: ev.get(k) for k in ("videos", "frames", "auc_raw", "auc_norm", "auc_macro", "eer",
                                            "threshold_raw_at_eer", "fps_scored")} if ev else None}
    card = {
        "version": args.version, "note": args.note,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data": {"dvc": dvc_hashes(), "shanghaitech": read_json(meta / "shanghaitech.json"),
                 "manual": read_json(meta / "manual.json"),
                 "split": {k: len(v) for k, v in (read_json(meta / "split.json") or {}).items()}},
        "models": models,
        "behavior": {d: read_json(out / "behavior" / d / "metrics.json") for d in ("shanghaitech", "manual")},
        "comparison": read_json(out / "comparison" / "comparison.json"),
        "software": software(),
    }
    rel = Path("releases")
    rel.mkdir(exist_ok=True)
    (rel / f"{args.version}.json").write_text(json.dumps(card, indent=2))

    pct = lambda v: "-" if v is None else f"{v * 100:.1f}%"
    sh = card["data"]["shanghaitech"] or {}
    md = [f"# {args.version}" + (f": {args.note}" if args.note else ""), "",
          f"Created {card['created_utc']} from commit `{card['git_commit']}`.", "", "## Data",
          f"- ShanghaiTech Campus: {sh.get('videos', '-')} videos, {sh.get('frames', '-')} frames; "
          f"split {card['data']['split']}",
          f"- Manual videos: {(card['data']['manual'] or {}).get('videos', '-')}"]
    md += [f"- `{p}` md5 `{h['md5']}`" for p, h in card["data"]["dvc"].items()]
    md += ["", "## Models", "| Model | Parameters | AUC micro | AUC norm | AUC macro | EER |", "|---|---|---|---|---|---|"]
    for m, d in models.items():
        t = d["test"] or {}
        p = (d["parameters"] or {}).get("Total")
        md.append(f"| {m} | {p:,} | {pct(t.get('auc_raw'))} | {pct(t.get('auc_norm'))} | "
                  f"{pct(t.get('auc_macro'))} | {pct(t.get('eer'))} |" if p else
                  f"| {m} | - | {pct(t.get('auc_raw'))} | {pct(t.get('auc_norm'))} | {pct(t.get('auc_macro'))} | {pct(t.get('eer'))} |")
    b = (card["behavior"]["shanghaitech"] or {}).get("frame_level")
    if b:
        md += ["", f"Behavior rules (no training): frame AUC {pct(b.get('auc'))}, events "
               f"{card['behavior']['shanghaitech']['events']}"]
    (rel / f"{args.version}.md").write_text("\n".join(md) + "\n")
    missing = [m for m in MODELS if m not in models or not models[m]["test"]]
    print("\n".join(md))
    if missing:
        print(f"\nnote: not trained / evaluated yet: {missing}")
    print(f"\nnext:\n  git add releases dvc.lock data/metadata outputs && git commit -m \"release {args.version}\"\n"
          f"  git tag -a {args.version} -m \"{args.note or args.version}\"\n  git push --follow-tags && dvc push")


if __name__ == "__main__":
    main()
