"""Train a model. One script for both models; the config decides which.

    python -m src.training.train --config configs/astt_baseline.yaml           # model 1
    python -m src.training.train --config configs/astt_improved.yaml           # model 2
    python -m src.training.train --config configs/aivad.yaml                   # model 3 (fit)
    python -m src.training.train --config configs/aivad_lite.yaml              # model 4 (fit)
    python -m src.training.train --config configs/astt_baseline.yaml --smoke   # 2-minute test

Stopped (shutdown, crash, Ctrl+C)? Run the SAME command again: it continues from the last
checkpoint (saved every `checkpoint_minutes` and after every epoch). --fresh starts over.

Output: outputs/training/<model>/
    model.pth        trained weights (DVC-versioned; what evaluation / API load)
    metrics.json     final loss, epochs, time, parameters (DVC metrics)
    log.csv          loss / learning rate over time (DVC plots)
    run.json         everything needed to reproduce: config, git commit, data version, split
    checkpoints/     last.pth (end of epoch) and resume.pth (mid-epoch), for continuing

Model 1 keeps the thesis repo's training procedure exactly (train_val_UIT_ADrone_baseline_ca.py):
MSE(model(frames[:, :4]), frame[4]), SGD momentum 0.9 + OneCycleLR stepped every batch, fp32.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src.config import (config_hash, frames_dir, git_commit, load_config, training_dir)
from src.data.dataset import train_dataset
from src.data.split import load_split, video_labels
from src.evaluation.metrics import frame_metrics
from src.evaluation.scoring import autocast, gradient_loss, score_video
from src.models import backbone_parameters, build_model, count_parameters
from src.training.monitor import TrainingMonitor


def make_optimizer(cfg, model):
    t = cfg["train"]
    if t["backbone_lr"] == t["lr"]:
        groups = [{"params": list(model.parameters()), "lr": t["lr"]}]       # repo: one group
    else:
        bb = {id(p) for p in backbone_parameters(cfg, model)}
        groups = [{"params": [p for p in model.parameters() if id(p) in bb], "lr": t["backbone_lr"]},
                  {"params": [p for p in model.parameters() if id(p) not in bb], "lr": t["lr"]}]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=t["lr"], weight_decay=t["wd"], momentum=0.9)
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=t["lr"], weight_decay=t["wd"])
    raise ValueError(t["optimizer"])


def make_scheduler(cfg, optimizer, total_steps):
    t = cfg["train"]
    if t["scheduler"] == "onecycle":                              # repo
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
            pct_start=t["warmup_steps"] / total_steps, total_steps=total_steps)
    if t["scheduler"] == "cosine":
        w = t["warmup_steps"]
        f = lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(total_steps - w, 1)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    raise ValueError(t["scheduler"])


def save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)                                 # never leave a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the setup")
    ap.add_argument("--fresh", action="store_true", help="delete the old output and start over")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, "configs/api.yaml")
    if cfg["model"].get("family") == "aivad":                 # models 3 / 4: nothing to backprop
        from src.models.aivad.pipeline import fit
        return fit(cfg, args.smoke, args.fresh)
    t = cfg["train"]
    split = load_split(cfg)
    if args.smoke:
        split = {"train": split["train"][:2], "val": (split["val"] or split["test"])[:1], "test": []}
        t.update(epochs=1, batch_size=2, warmup_steps=2, checkpoint_minutes=1)
        cfg["eval"]["batch_size"] = 2
    out = training_dir(cfg, args.smoke)
    ck_dir = out / "checkpoints"
    chash = config_hash({k: cfg[k] for k in ("model", "input", "train", "shanghaitech")})

    # ---- same command again = continue; different settings = refuse (unless --fresh)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    resume_from = None
    if ck_dir.exists() and any(ck_dir.glob("*.pth")):
        old = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
        if old.get("config_hash") != chash:
            raise SystemExit(f"{out} holds a run with DIFFERENT settings. Use --fresh to start over.")
        resume_from = max(ck_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    ck_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    params = count_parameters(cfg, model)
    print(f"{cfg['model']['name']} on {device}: {params['Total']:,} parameters")

    vid_dirs = [frames_dir(cfg, "shanghaitech") / v for v in split["train"]]
    ds = train_dataset(cfg, vid_dirs)
    loader = data.DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                             num_workers=t["num_workers"], drop_last=True)
    steps_per_epoch = min(len(loader), 20) if args.smoke else len(loader)
    total = steps_per_epoch * t["epochs"] if t["train_steps"] in ("auto", None) else int(t["train_steps"])
    if total < steps_per_epoch * t["epochs"]:
        raise SystemExit("train_steps < epochs x batches (OneCycleLR would crash). Use train_steps: auto")
    print(f"train: {len(split['train'])} videos, {len(ds):,} windows, {steps_per_epoch:,} batches/epoch, "
          f"{total:,} steps")

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer, total)
    mse_loss = nn.MSELoss(reduction="mean")
    start_epoch, skip, best, history = 1, 0, -1.0, []
    if resume_from:
        ck = torch.load(resume_from, map_location=device)
        model.load_state_dict(ck["state_dict"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["lr_scheduler"])
        start_epoch, skip = (ck["epoch"], ck["batch"] + 1) if "batch" in ck else (ck["epoch"] + 1, 0)
        best, history = ck.get("best", -1.0), ck.get("history", [])
        print(f"continuing from {resume_from.name}: epoch {start_epoch}, batch {skip}")
    else:
        (out / "run.json").write_text(json.dumps({
            "model": cfg["model"]["name"], "config_file": args.config, "config_hash": chash,
            "git_commit": git_commit(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": str(device), "parameters": params,
            "data": json.loads(Path(cfg["paths"]["metadata"], "shanghaitech.json").read_text())
            if Path(cfg["paths"]["metadata"], "shanghaitech.json").exists() else None,
            "split": {k: len(v) for k, v in split.items()}, "config": cfg}, indent=2))
        with open(out / "log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr", "minutes"])

    monitor = TrainingMonitor(cfg["model"]["name"], 0 if args.smoke else
                              cfg["monitoring"]["training_metrics_port"])
    state = lambda e, **kw: {"epoch": e, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "lr_scheduler": scheduler.state_dict(), "best": best, "history": history, **kw}
    log_f = open(out / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    t0 = last_save = time.time()
    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        losses, te = [], time.time()
        first = skip if epoch == start_epoch else 0
        for k, batch in enumerate(loader):                     # --- repo train_epoch
            if k >= steps_per_epoch - first:
                break
            b_idx = first + k
            target = batch["256"][:, 4].to(device).float()
            frames = batch["standard"][:, :4].to(device)
            optimizer.zero_grad()
            with autocast(cfg, device):
                pred = model(frames)
            loss = mse_loss(target, pred.float())
            if t["loss"] == "mse+gradient":
                loss = loss + t["gradient_loss_weight"] * gradient_loss(pred.float(), target)
            loss.backward()
            if t.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step = (epoch - 1) * steps_per_epoch + b_idx + 1
            if b_idx % 100 == 0 or b_idx == steps_per_epoch - 1:
                speed = (k + 1) / (time.time() - te)
                lr = optimizer.param_groups[-1]["lr"]
                print(f"epoch {epoch}/{t['epochs']} batch {b_idx:05d}/{steps_per_epoch:05d} "
                      f"loss {np.mean(losses):.4f}  ({speed:.2f} it/s, eta epoch "
                      f"{(steps_per_epoch - b_idx) / max(speed, 1e-9) / 60:.0f} min)")
                log.writerow([step, epoch, round(float(np.mean(losses)), 6), lr,
                              round((time.time() - t0) / 60, 2)])
                log_f.flush()
                monitor.update(loss=float(np.mean(losses)), lr=lr, speed=speed, epoch=epoch,
                               step=step, total_steps=total)
            if time.time() - last_save > 60 * t["checkpoint_minutes"] and b_idx < steps_per_epoch - 1:
                save(ck_dir / "resume.pth", state(epoch, batch=b_idx))
                last_save = time.time()
                print(f"  checkpoint saved (epoch {epoch}, batch {b_idx})")

        # ---- end of epoch: validation (if any), checkpoints
        val_auc = float("nan")
        if split["val"]:
            scores = {v: score_video(model, cfg, frames_dir(cfg, "shanghaitech") / v, device)
                      for v in split["val"]}
            res, _ = frame_metrics(scores, {v: video_labels(cfg, v, len(s)) for v, s in scores.items()})
            val_auc = res.get("auc_raw", float("nan"))           # repo picks the epoch on raw AUC
            monitor.validation(val_auc)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                        "val_auc_raw": val_auc})
        improved = not split["val"] or (val_auc == val_auc and val_auc > best)
        if improved:
            best = val_auc if split["val"] else best
            save(out / "model.pth", {"epoch": epoch, "state_dict": model.state_dict()})
        save(ck_dir / "last.pth", state(epoch))
        (ck_dir / "resume.pth").unlink(missing_ok=True)
        last_save = time.time()
        print(f"epoch {epoch} done, loss {history[-1]['train_loss']:.4f}"
              + (f", val AUC {val_auc:.4f}{' (best)' if improved else ''}" if split["val"]
                 else " (no validation: model.pth = latest epoch)"))
        (out / "metrics.json").write_text(json.dumps({
            "epochs_done": epoch, "final_train_loss": history[-1]["train_loss"],
            "best_val_auc_raw": best if split["val"] else None,
            "train_minutes_this_session": round((time.time() - t0) / 60, 1),
            "parameters_total": params["Total"], "history": history}, indent=2))
    log_f.close()
    print(f"done: {out}/model.pth\nnext: python -m src.evaluation.evaluate --config {args.config}"
          + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
