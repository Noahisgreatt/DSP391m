"""Train any model from its YAML config.

    python -m src.train --config configs/models/vit_flow.yaml
    python -m src.train --config configs/models/vit_flow.yaml --set train.lr=5e-5 train.epochs=10

Writes models/<model_name>/<run_id>/{config.yaml, best.pt, last.pt, metrics.json}
and models/<model_name>/latest.txt (run id of the newest run, used by evaluate/visualize).
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FlowSequenceDataset
from src.models import build_model, count_params
from src.utils import apply_overrides, load_params, load_yaml, resolve, save_json, save_yaml, set_seed, split_of


def make_run_id(cfg: dict) -> str:
    lr = f"{cfg['train']['lr']:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{dt.date.today().isoformat()}_lr{lr}"


def split_videos(params: dict, cfg: dict) -> tuple[list[str], list[str]]:
    d = cfg["data"]
    feat_dir = resolve(params["paths"]["features"]) / d["feature_method"] / d["dataset"]
    ids = [f.stem for f in sorted(feat_dir.glob("*.npz")) if split_of(f.stem, params, d["dataset"]) == "train"]
    rng = np.random.default_rng(params.get("seed", 42))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * d["val_fraction"]))) if len(ids) > 1 else 0
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def losses(out: dict, batch: dict, cfg: dict, class_w: torch.Tensor) -> dict:
    w = cfg["train"]["loss_weights"]
    l_flow = F.mse_loss(out["flow"], batch["y_flow"])
    l_cls = F.cross_entropy(out["logits"], batch["y_cls"], weight=class_w)
    return {"loss": w["flow"] * l_flow + w["cls"] * l_cls, "flow": l_flow, "cls": l_cls}


@torch.no_grad()
def validate(model, loader, cfg, class_w, device, amp) -> dict:
    model.eval()
    tot = {"loss": 0.0, "flow": 0.0, "cls": 0.0}
    correct = n = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device.type, enabled=amp):
            out = model(batch["x"], out_size=batch["y_flow"].shape[-2:])
        ls = losses({k: v.float() for k, v in out.items()}, batch, cfg, class_w)
        bs = len(batch["x"])
        for k in tot:
            tot[k] += ls[k].item() * bs
        correct += (out["logits"].argmax(1) == batch["y_cls"]).sum().item()
        n += bs
    return {**{f"val_{k}": v / max(n, 1) for k, v in tot.items()}, "val_acc": correct / max(n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="Overrides like train.lr=1e-4")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="Debug: stop each epoch after N steps.")
    args = ap.parse_args()

    params = load_params()
    cfg = apply_overrides(load_yaml(resolve(args.config)), args.set)
    set_seed(params.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"].get("amp")) and device.type == "cuda"

    model = build_model(cfg).to(device)
    total, trainable = count_params(model)
    print(f"{cfg['model']['name']}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    if total < cfg["model"].get("min_params", 0):
        raise SystemExit(f"Model has {total:,} params < required {cfg['model']['min_params']:,}")

    train_ids, val_ids = split_videos(params, cfg)
    train_ds = FlowSequenceDataset(params, cfg, video_ids=train_ids)
    val_ds = FlowSequenceDataset(params, cfg, video_ids=val_ids) if val_ids else None
    print(f"train windows: {len(train_ds)} ({len(train_ids)} videos) | val: {len(val_ds) if val_ds else 0}")

    nc = cfg["model"]["num_classes"]
    counts = train_ds.class_counts(nc)
    print("pseudo-label counts (normal, loitering, counter-flow):", counts.tolist())
    # inverse-frequency weights, capped so a handful of positives cannot dominate
    class_w = torch.tensor(np.clip(counts.sum() / np.maximum(counts, 1) / nc, 0.1, 20.0),
                           dtype=torch.float32, device=device)

    tc = cfg["train"]
    kw = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"], pin_memory=device.type == "cuda",
              persistent_workers=tc["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > tc["batch_size"], **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw) if val_ds else None

    opt = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": tc["backbone_lr"]},
        {"params": model.head_parameters(), "lr": tc["lr"]},
    ], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train_dl) if not args.max_steps else min(len(train_dl), args.max_steps)
    total_steps = max(1, steps_per_epoch * tc["epochs"])
    warm = steps_per_epoch * tc.get("warmup_epochs", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total_steps - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=amp)

    run_id = args.run_id or make_run_id(cfg)
    run_dir = resolve(params["paths"]["models"]) / cfg["model"]["name"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    (run_dir.parent / "latest.txt").write_text(run_id)

    history, best, bad = [], float("inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, run = time.time(), {"loss": 0.0, "flow": 0.0, "cls": 0.0}
        pbar = tqdm(train_dl, total=steps_per_epoch, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            if args.max_steps and step >= args.max_steps:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device.type, enabled=amp):
                out = model(batch["x"], out_size=batch["y_flow"].shape[-2:])
            ls = losses({k: v.float() for k, v in out.items()}, batch, cfg, class_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(ls["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k in run:
                run[k] += ls[k].item()
            pbar.set_postfix(loss=f"{ls['loss'].item():.4f}")
        rec = {"epoch": epoch, **{f"train_{k}": v / max(steps_per_epoch, 1) for k, v in run.items()},
               "time_s": round(time.time() - t0, 1)}
        if val_dl:
            rec.update(validate(model, val_dl, cfg, class_w, device, amp))
        history.append(rec)
        print({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()})

        score = rec.get("val_loss", rec["train_loss"])
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "class_weights": class_w.cpu()}
        torch.save(ckpt, run_dir / "last.pt")
        if score < best:
            best, bad = score, 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad += 1
        save_json({"params_total": total, "params_trainable": trainable, "best_val_loss": best,
                   "train_videos": train_ids, "val_videos": val_ids, "history": history},
                  run_dir / "metrics.json")
        if bad >= tc.get("early_stop_patience", 10**9):
            print(f"Early stop at epoch {epoch}")
            break
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
