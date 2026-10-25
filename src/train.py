"""Step 3, TRAINING. One command for every model; the config decides which.

    python -m src.train --config configs/astt_baseline.yaml           # model 1
    python -m src.train --config configs/astt_improved.yaml           # model 2
    python -m src.train --config configs/aivad.yaml                   # model 3 (attributes + density fit)
    python -m src.train --config configs/aivad_lite.yaml              # model 4
    add --smoke for a 2-minute test run, --fresh to throw away an old run and start over

Stopped (shutdown, crash, Ctrl+C)? Run the SAME command again: it continues from the last
checkpoint (saved every `checkpoint_minutes` and after every epoch).

Out: outputs/training/<model>/
    model.pth     trained weights (DVC-versioned; what evaluation / demo / API load)
    metrics.json  final loss, epochs, time, parameters        log.csv  loss over time
    run.json      config, git commit, data version, split     checkpoints/  to continue a run
While it runs, live numbers for Grafana: http://localhost:8001/metrics (configs/api.yaml).

Model 1 keeps the thesis repo's training exactly (train_val_UIT_ADrone_baseline_ca.py):
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
import torch.utils.data as tud

from src.data import load_split, train_dataset, video_labels
from src.utils import config_hash, frames_dir, git_commit, load_config, out_dir


class Monitor:
    """Live training metrics for Prometheus / Grafana (optional: needs prometheus-client)."""

    def __init__(self, model, port):
        self.on = False
        if not port:
            return
        try:
            from prometheus_client import Gauge, start_http_server
            start_http_server(port)
        except Exception as e:                                # not installed / port busy
            print(f"live metrics off ({e})")
            return
        g = lambda n, h: Gauge(n, h, ["model"]).labels(model)
        self.loss, self.lr, self.speed = g("vad_train_loss", "loss"), g("vad_train_learning_rate", "lr"), g("vad_train_iterations_per_second", "it/s")
        self.epoch, self.progress = g("vad_train_epoch", "epoch"), g("vad_train_progress_ratio", "done 0..1")
        self.eta, self.val, self.gpu = g("vad_train_eta_seconds", "s left"), g("vad_train_val_auc", "val AUC"), g("vad_train_gpu_memory_bytes", "GPU")
        self.on = True
        print(f"live metrics: http://localhost:{port}/metrics")

    def update(self, loss, lr, speed, epoch, step, total):
        if self.on:
            self.loss.set(loss); self.lr.set(lr); self.speed.set(speed); self.epoch.set(epoch)
            self.progress.set(step / total); self.eta.set((total - step) / max(speed, 1e-9))
            if torch.cuda.is_available():
                self.gpu.set(torch.cuda.memory_allocated())


def optimizer_and_schedule(cfg, model, total):
    t = cfg["train"]
    if t["backbone_lr"] == t["lr"]:                           # model 1 (repo): one group
        groups = [{"params": list(model.parameters()), "lr": t["lr"]}]
    else:                                                     # model 2: pretrained ViT 10x slower
        bb = {id(p) for p in model.spatial_transformer.parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) in bb], "lr": t["backbone_lr"]},
                  {"params": [p for p in model.parameters() if id(p) not in bb], "lr": t["lr"]}]
    opt = (torch.optim.SGD(groups, lr=t["lr"], weight_decay=t["wd"], momentum=0.9) if t["optimizer"] == "sgd"
           else torch.optim.AdamW(groups, lr=t["lr"], weight_decay=t["wd"]))
    w = t["warmup_steps"]
    if w >= total:                                            # only on tiny test data
        w = max(1, total // 10)
        print(f"warning: warm-up longer than the whole run; using {w} steps")
    if t["scheduler"] == "onecycle":                          # repo
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                  pct_start=w / total, total_steps=total)
    else:                                                     # warm-up + cosine
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(total - w, 1))))
    return opt, sch


def save(path, obj):
    torch.save(obj, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)                    # never a half-written file


def train_astt(cfg, args):
    from src.evaluate import frame_metrics
    from src.models.astt import autocast, build, count_parameters, gradient_loss, score_video
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, split = cfg["train"], load_split(cfg)
    if args.smoke:
        split = {"train": split["train"][:2], "val": (split["val"] or split["test"])[:1], "test": []}
        t.update(epochs=1, batch_size=2, warmup_steps=2, checkpoint_minutes=1)
        cfg["eval"]["batch_size"] = 2
    out = out_dir(cfg, "training", args.smoke)
    ck_dir, chash = out / "checkpoints", config_hash({k: cfg[k] for k in ("model", "input", "train", "shanghaitech")})
    if args.fresh and out.exists():
        shutil.rmtree(out)
    resume = None                       # same command again = continue; other settings = refuse
    if ck_dir.exists() and any(ck_dir.glob("*.pth")):
        old = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
        if old.get("config_hash") != chash:
            raise SystemExit(f"{out} holds a run with DIFFERENT settings. Use --fresh to start over.")
        resume = max(ck_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    ck_dir.mkdir(parents=True, exist_ok=True)

    model = build(cfg).to(device)
    params = count_parameters(model)
    print(f"{cfg['model']['name']} on {device}: {params['Total']:,} parameters")
    ds = train_dataset(cfg, [frames_dir(cfg, "shanghaitech") / v for v in split["train"]])
    loader = tud.DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    per_epoch = min(len(loader), 20) if args.smoke else len(loader)
    total = per_epoch * t["epochs"] if t["train_steps"] in ("auto", None) else int(t["train_steps"])
    print(f"train: {len(split['train'])} videos, {len(ds):,} windows, {per_epoch:,} batches/epoch, {total:,} steps")
    opt, sch = optimizer_and_schedule(cfg, model, total)
    mse = nn.MSELoss()
    start_epoch, skip, best, history = 1, 0, -1.0, []
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"]); sch.load_state_dict(ck["lr_scheduler"])
        start_epoch, skip = (ck["epoch"], ck["batch"] + 1) if "batch" in ck else (ck["epoch"] + 1, 0)
        best, history = ck.get("best", -1.0), ck.get("history", [])
        print(f"continuing from {resume.name}: epoch {start_epoch}, batch {skip}")
    else:
        card = Path(cfg["paths"]["metadata"]) / "shanghaitech.json"
        (out / "run.json").write_text(json.dumps({
            "model": cfg["model"]["name"], "config_file": args.config, "config_hash": chash, "git_commit": git_commit(),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": str(device),
            "parameters": params, "data": json.loads(card.read_text()) if card.exists() else None,
            "split": {k: len(v) for k, v in split.items()}, "config": cfg}, indent=2))
        with open(out / "log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr", "minutes"])

    monitor = Monitor(cfg["model"]["name"], 0 if args.smoke else cfg["monitoring"]["training_metrics_port"])
    state = lambda e, **kw: {"epoch": e, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                             "lr_scheduler": sch.state_dict(), "best": best, "history": history, **kw}
    log_f = open(out / "log.csv", "a", newline="")
    log, t0 = csv.writer(log_f), time.time()
    last_save = t0
    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        losses, te, first = [], time.time(), skip if epoch == start_epoch else 0
        for k, batch in enumerate(loader):                     # --- repo train_epoch
            if k >= per_epoch - first:
                break
            b = first + k
            target, frames = batch["256"][:, 4].to(device).float(), batch["standard"][:, :4].to(device)
            opt.zero_grad()
            with autocast(cfg, device):
                pred = model(frames)
            loss = mse(target, pred.float())
            if t["loss"] == "mse+gradient":
                loss = loss + t["gradient_loss_weight"] * gradient_loss(pred.float(), target)
            loss.backward()
            if t.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            sch.step()
            losses.append(loss.item())
            step = (epoch - 1) * per_epoch + b + 1
            if b % 100 == 0 or b == per_epoch - 1:
                speed, lr = (k + 1) / (time.time() - te), opt.param_groups[-1]["lr"]
                print(f"epoch {epoch}/{t['epochs']} batch {b:05d}/{per_epoch:05d} loss {np.mean(losses):.4f}  "
                      f"({speed:.2f} it/s, eta epoch {(per_epoch - b) / max(speed, 1e-9) / 60:.0f} min)")
                log.writerow([step, epoch, round(float(np.mean(losses)), 6), lr, round((time.time() - t0) / 60, 2)])
                log_f.flush()
                monitor.update(float(np.mean(losses)), lr, speed, epoch, step, total)
            if time.time() - last_save > 60 * t["checkpoint_minutes"] and b < per_epoch - 1:
                save(ck_dir / "resume.pth", state(epoch, batch=b))
                last_save = time.time()
                print(f"  checkpoint saved (epoch {epoch}, batch {b})")
        val_auc = float("nan")                                 # --- end of epoch
        if split["val"]:
            scores = {v: score_video(model, cfg, frames_dir(cfg, "shanghaitech") / v, device) for v in split["val"]}
            val_auc = frame_metrics(scores, {v: video_labels(cfg, v, len(s)) for v, s in scores.items()})[0].get("auc_raw", float("nan"))
            if monitor.on and val_auc == val_auc:
                monitor.val.set(val_auc)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "val_auc_raw": val_auc})
        improved = not split["val"] or (val_auc == val_auc and val_auc > best)
        if improved:
            best = val_auc if split["val"] else best
            save(out / "model.pth", {"epoch": epoch, "state_dict": model.state_dict()})
        save(ck_dir / "last.pth", state(epoch))
        (ck_dir / "resume.pth").unlink(missing_ok=True)
        last_save = time.time()
        print(f"epoch {epoch} done, loss {history[-1]['train_loss']:.4f}" +
              (f", val AUC {val_auc:.4f}{' (best)' if improved else ''}" if split["val"] else " (model.pth = this epoch)"))
        (out / "metrics.json").write_text(json.dumps({
            "epochs_done": epoch, "final_train_loss": history[-1]["train_loss"], "best_val_auc_raw": best if split["val"] else None,
            "train_minutes_this_session": round((time.time() - t0) / 60, 1), "parameters_total": params["Total"],
            "history": history}, indent=2))
    log_f.close()
    print(f"done: {out}/model.pth\nnext: python -m src.evaluate --config {args.config}" + (" --smoke" if args.smoke else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the setup")
    ap.add_argument("--fresh", action="store_true", help="delete the old output and start over")
    args = ap.parse_args()
    cfg = load_config(args.config, "configs/api.yaml")
    if cfg["model"].get("family") == "aivad":                 # models 3 / 4: no gradient training
        from src.models.aivad import fit
        return fit(cfg, args.smoke, args.fresh)
    train_astt(cfg, args)


if __name__ == "__main__":
    main()
