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
    metrics.json  epochs, final loss, parameters              log.csv         loss over time
    run.json      settings, git commit, parameters            checkpoint.pth  to continue a run
While it runs, live numbers for Grafana: http://localhost:8001/metrics (configs/api.yaml).

Model 1 keeps the thesis repo's training exactly (train_val_UIT_ADrone_baseline_ca.py):
MSE(model(frames[:, :4]), frame[4]), SGD momentum 0.9 + OneCycleLR stepped every batch, fp32.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from src.data import load_split, train_dataset
from src.utils import config_hash, frames_dir, git_commit, load_config, out_dir


class Monitor:
    """Live training numbers for Prometheus / Grafana (http://localhost:<port>/metrics)."""

    def __init__(self, model, port):
        self.gauges = {}
        try:
            from prometheus_client import Gauge, start_http_server
            if port:
                start_http_server(port)
                print(f"live metrics: http://localhost:{port}/metrics")
                self.gauges = {k: Gauge(f"vad_train_{k}", k, ["model"]).labels(model)
                               for k in ("loss", "learning_rate", "iterations_per_second", "epoch", "progress_ratio", "eta_seconds")}
        except Exception as e:                                # not installed / port busy: training goes on
            print(f"live metrics off ({e})")

    def update(self, **values):
        for k, v in values.items():
            if k in self.gauges:
                self.gauges[k].set(v)


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
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(obj, tmp)
    except (OSError, RuntimeError) as e:                      # e.g. disk full: keep the last good file
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"could not save {path} ({e}); free some disk space and run again: "
                         "training continues from the last checkpoint")
    tmp.replace(path)                                         # never a half-written file


def train_astt(cfg, args):
    from src.models.astt import autocast, build, count_parameters, gradient_loss, half_dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, train_vids = cfg["train"], load_split(cfg)["train"]
    if args.smoke:
        train_vids = train_vids[:2]
        t.update(epochs=1, batch_size=2, warmup_steps=2)
    out = out_dir(cfg, "training", args.smoke)
    ckpt, chash = out / "checkpoint.pth", config_hash({k: cfg[k] for k in ("model", "input", "train")})
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if ckpt.exists() and json.loads((out / "run.json").read_text())["config_hash"] != chash:
        raise SystemExit(f"{out} holds a run with DIFFERENT settings. Add --fresh to start over.")

    model = build(cfg).to(device)
    params = count_parameters(model)
    print(f"{cfg['model']['name']} on {device}: {params['Total']:,} parameters")
    ds = train_dataset(cfg, [frames_dir(cfg, "shanghaitech") / v for v in train_vids])
    loader = tud.DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    per_epoch = min(len(loader), 20) if args.smoke else len(loader)
    total = per_epoch * t["epochs"] if t["train_steps"] in ("auto", None) else int(t["train_steps"])
    print(f"train: {len(train_vids)} videos, {len(ds):,} windows, {per_epoch:,} batches/epoch, {total:,} steps")
    opt, sch = optimizer_and_schedule(cfg, model, total)
    fp16 = device.type == "cuda" and t.get("amp", False) and half_dtype(device) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)       # fp16 needs loss scaling; bf16 / fp32 don't
    epoch, done = 1, 0                                         # where to start (batches done in that epoch)
    if ckpt.exists():                                          # same command again = continue
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"]); sch.load_state_dict(ck["lr_scheduler"])
        if ck.get("scaler"):                                    # empty when saved by a bf16 / fp32 run
            scaler.load_state_dict(ck["scaler"])
        epoch, done = ck["epoch"], ck["done"]
        print(f"continuing: epoch {epoch}, batch {done}")
    else:
        (out / "run.json").write_text(json.dumps({
            "model": cfg["model"]["name"], "config_file": args.config, "config_hash": chash, "git_commit": git_commit(),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": str(device),
            "parameters": params, "train_videos": len(train_vids), "config": cfg}, indent=2))
        (out / "log.csv").write_text("step,epoch,loss,lr,minutes\n")

    monitor = Monitor(cfg["model"]["name"], 0 if args.smoke else cfg["monitoring"]["training_metrics_port"])
    checkpoint = lambda e, d: save(ckpt, {"epoch": e, "done": d, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                                          "lr_scheduler": sch.state_dict(), "scaler": scaler.state_dict()})
    t0 = last_save = time.time()
    while epoch <= t["epochs"]:
        model.train()
        losses, te = [], time.time()
        for k, batch in zip(range(done, per_epoch), loader):    # the thesis repo's train_epoch
            target, frames = batch["256"][:, 4].to(device).float(), batch["standard"][:, :4].to(device)
            opt.zero_grad()
            with autocast(cfg, device):
                pred = model(frames).float()
            loss = nn.functional.mse_loss(pred, target)
            if t["loss"] == "mse+gradient":
                loss = loss + t["gradient_loss_weight"] * gradient_loss(pred, target)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt); scaler.update(); sch.step()
            losses.append(loss.item())
            if k % 100 == 0 or k == per_epoch - 1:
                speed, lr, step = (len(losses)) / (time.time() - te), opt.param_groups[-1]["lr"], (epoch - 1) * per_epoch + k + 1
                print(f"epoch {epoch}/{t['epochs']} batch {k:05d}/{per_epoch:05d} loss {np.mean(losses):.4f}  "
                      f"({speed:.2f} it/s, eta epoch {(per_epoch - k) / speed / 60:.0f} min)")
                with open(out / "log.csv", "a") as f:
                    f.write(f"{step},{epoch},{np.mean(losses):.6f},{lr},{(time.time() - t0) / 60:.2f}\n")
                monitor.update(loss=np.mean(losses), learning_rate=lr, iterations_per_second=speed, epoch=epoch,
                               progress_ratio=step / total, eta_seconds=(total - step) / speed)
            if time.time() - last_save > 60 * t["checkpoint_minutes"]:
                checkpoint(epoch, k + 1)
                last_save = time.time()
            if args.max_hours and time.time() - t0 > 3600 * args.max_hours:   # e.g. Kaggle's 12-hour limit
                checkpoint(epoch, k + 1)
                return print(f"stopped after {args.max_hours} h at epoch {epoch}, batch {k + 1}: run again to continue")
        save(out / "model.pth", {"epoch": epoch, "state_dict": model.state_dict()})   # model.pth = last epoch
        epoch, done = epoch + 1, 0
        checkpoint(epoch, 0)
        print(f"epoch {epoch - 1} done, loss {np.mean(losses):.4f}")
        (out / "metrics.json").write_text(json.dumps({"epochs_done": epoch - 1, "final_train_loss": float(np.mean(losses)),
                                                      "parameters_total": params["Total"]}, indent=2))
    print(f"done: {out}/model.pth")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the setup")
    ap.add_argument("--fresh", action="store_true", help="delete the old output and start over")
    ap.add_argument("--max-hours", type=float, help="save and stop after this many hours (continue later)")
    args = ap.parse_args()
    cfg = load_config(args.config, "configs/api.yaml")
    if cfg["model"].get("family") == "aivad":                 # models 3 / 4: no gradient training
        from src.models.aivad import fit
        return fit(cfg, args.smoke, args.fresh)
    train_astt(cfg, args)


if __name__ == "__main__":
    main()
