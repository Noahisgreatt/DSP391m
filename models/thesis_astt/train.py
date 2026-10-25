"""Train the thesis model exactly like the repo's train_val_UIT_ADrone_*.py, on ShanghaiTech.

    python -m models.thesis_astt.train --config models/thesis_astt/config.yaml
    python -m models.thesis_astt.train --config models/thesis_astt/config.yaml --smoke
    python -m models.thesis_astt.train --resume runs/thesis_astt/<run_id>   # after a shutdown

Kept from the repo (models/thesis_astt/original_repo/train_val_UIT_ADrone_baseline_ca.py):
  * data: models.thesis_astt.original_repo.data_utils.DataLoader on a folder of training videos
    (all frames concatenated, 4 input frames at 384 px + target frame at 256 px, BGR, [-1, 1])
  * loss: nn.MSELoss(batch_data_256[:, 4], model(batch_data[:, :4]))
  * SGD(lr 1e-4, momentum 0.9, wd 1e-4) + OneCycleLR(warm-up 500), stepped every batch
  * best checkpoint = best validation frame AUC, checked after every epoch
Only adapted to this project:
  * paths: the training folder is a folder of links to the normal training videos of our split
    (same split as CrowdFormer); validation uses our val videos instead of the repo's single
    hard-coded test video (DJI_0073), so the test videos are never used to pick the epoch
  * run folder runs/thesis_astt/<run_id>/ like every other model in this project
  * `train_steps: auto` = epochs x batches (the repo's fixed 38000 crashes on bigger data)
  * checkpoints every `checkpoint_minutes` (resume.pth) + `--resume`, so a shutdown doesn't
    cost a whole run. After a mid-epoch resume, the rest of that epoch uses a new random order.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import yaml

from src.data.split import make_split, video_labels
from src.metrics import frame_metrics
from models.thesis_astt.model import build_thesis_model, count_parameters
from src.utils import frames_root, list_frames, load_config
from models.thesis_astt.original_repo.data_utils import DataLoader as RepoDataLoader

LINKS = Path("data/interim/thesis_links")


def jpg_view(src_dir: Path, dst_dir: Path) -> Path:
    """A folder of links to only the .jpg frames (the repo loader expects nothing else in it)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in list_frames(src_dir):
        link = dst_dir / f.name
        if not link.exists():
            os.symlink(f.resolve(), link)
    return dst_dir


def repo_dataset(folder: Path, image_size: int):
    return RepoDataLoader(str(folder), transforms.Compose([transforms.ToTensor()]),
                          resize_height=image_size, resize_width=image_size)


def autocast(cfg, device):
    if device.type == "cuda" and cfg["train"].get("amp", False):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def score_video(model, cfg, dataset: str, vid: str, device, return_maps=False):
    """Repo test_all_scenes for one video -> per-frame MSE (NaN for the first 4 frames)."""
    folder = jpg_view(frames_root(cfg, dataset) / vid, LINKS / dataset / vid)
    ds = repo_dataset(folder, cfg["data"]["image_size"])
    dl = data.DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                         num_workers=cfg["train"]["num_workers"], drop_last=False)
    n = len(ds.video_frames)
    mse = np.full(n, np.nan, np.float32)
    maps = np.zeros((n, 256, 256), np.float16) if return_maps else None
    loss_func_mse = nn.MSELoss(reduction="mean")
    model.eval()
    t = cfg["data"]["num_frames"]
    for b in dl:
        batch_data_256, batch_data = b["256"].to(device), b["standard"].to(device)
        with autocast(cfg, device):
            batch_pred = model(batch_data[:, :4])
        for j in range(len(batch_pred)):
            tgt, pr = batch_data_256[j, 4].float(), batch_pred[j].float()
            mse[t] = loss_func_mse(tgt, pr).item()
            if return_maps:
                maps[t] = ((pr - tgt) ** 2).mean(0).cpu().numpy()
            t += 1
    return (mse, maps) if return_maps else mse


def save_state(path, model, optimizer, lr_scheduler, epoch, batch=None, history=None,
               best_acc=None):
    """Repo checkpoint format, plus `batch` (mid-epoch) and history so a run can resume."""
    state = {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()}
    if batch is not None:
        state.update(batch=batch, history=history, best_acc=best_acc)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                       # never leave a half-written checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/thesis_astt/config.yaml")
    ap.add_argument("--run-name")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", help="run folder to continue, e.g. runs/thesis_astt/full")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        run = Path(args.resume)
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        split = json.loads((run / "split.json").read_text())
        t = cfg["train"]
    else:
        cfg = load_config(args.config)
        t = cfg["train"]
        split = make_split(cfg)
        if args.smoke:
            split = {**split, "train": split["train"][:2], "val": split["val"][:1]}
            t.update(epochs=1, batch_size=2, warmup_steps=2)
        run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run = Path("runs") / cfg["experiment"]["name"] / (run_id + ("_smoke" if args.smoke else ""))
        if run.exists():
            raise SystemExit(f"{run} already exists. Continue it with --resume {run}, "
                             "or pick another --run-name.")
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (run / "split.json").write_text(json.dumps(split, indent=1))
    print(f"run: {run}   device: {device}")

    train_dir = LINKS / "train_split" / run.name
    for v in split["train"]:
        jpg_view(frames_root(cfg, "shanghai") / v, train_dir / v)
    train_dataset = repo_dataset(train_dir, cfg["data"]["image_size"])
    train_batch = data.DataLoader(train_dataset, batch_size=t["batch_size"], shuffle=True,
                                  num_workers=t["num_workers"], drop_last=True)
    smoke = run.name.endswith("_smoke")
    steps_per_epoch = min(len(train_batch), 20) if smoke else len(train_batch)
    train_steps = steps_per_epoch * t["epochs"] if t["train_steps"] in ("auto", None) else int(t["train_steps"])
    if train_steps < steps_per_epoch * t["epochs"]:
        raise SystemExit(f"train_steps {train_steps} < epochs x batches = {steps_per_epoch * t['epochs']}; "
                         "the repo's OneCycleLR would crash. Set train_steps: auto")
    print(f"train size: {len(train_dataset)}  ({steps_per_epoch} batches/epoch, {train_steps} steps)")

    model = build_thesis_model(cfg).to(device)
    params = count_parameters(model)
    (run / "params.json").write_text(json.dumps(params, indent=2))
    print("parameters:", {k: f"{v / 1e6:.2f}M" for k, v in params.items()})

    loss_func_mse = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(params=model.parameters(), lr=t["lr"], weight_decay=t["wd"],
                                momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=t["lr"], pct_start=t["warmup_steps"] / train_steps,
        total_steps=train_steps)

    # ---- resume: newest of resume.pth (mid-epoch) / current.pth (end of epoch)
    start_epoch, skip, best_acc, history = 1, 0, 0.0, []
    if args.resume:
        cands = [p for p in (run / "resume.pth", run / "current.pth") if p.exists()]
        if not cands:
            raise SystemExit(f"No checkpoint in {run} yet (it stopped during epoch 1 before the "
                             "first save). Start again: delete the folder or use a new --run-name.")
        ck_path = max(cands, key=lambda p: p.stat().st_mtime)
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        lr_scheduler.load_state_dict(ck["lr_scheduler"])
        if "batch" in ck:                                    # stopped in the middle of an epoch
            start_epoch, skip = ck["epoch"], ck["batch"] + 1
            best_acc, history = ck["best_acc"], ck["history"]
        else:                                                # stopped after a whole epoch
            start_epoch = ck["epoch"] + 1
            if (run / "metrics.json").exists():
                m = json.loads((run / "metrics.json").read_text())
                best_acc, history = m["best_val_auc_raw"], m["history"]
        print(f"resumed from {ck_path.name}: epoch {start_epoch}, batch {skip}")

    log_f = open(run / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if start_epoch == 1 and skip == 0:
        log.writerow(["epoch", "batch", "loss", "sec"])
    save_every = 60 * float(t.get("checkpoint_minutes", 20))
    t0 = last_save = time.time()
    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        average_loss, te = [], time.time()
        first = skip if epoch == start_epoch else 0
        todo = steps_per_epoch - first            # after a resume: the rest of this epoch
        for k, batch in enumerate(train_batch):              # --- repo train_epoch
            if k >= todo:
                break
            batch_idx = first + k
            batch_data_256, batch_data = batch["256"].to(device), batch["standard"].to(device)
            optimizer.zero_grad()
            with autocast(cfg, device):
                batch_pred = model(batch_data[:, :4])
            loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred.float())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            average_loss.append(loss.item())
            if batch_idx % 100 == 0 or batch_idx == steps_per_epoch - 1:
                rate = (k + 1) / (time.time() - te)
                print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Reconstruction Loss: {:.4f}  "
                      "({:.2f} it/s, eta {:.0f} min)".format(
                          epoch, batch_idx, steps_per_epoch, np.mean(average_loss), rate,
                          (steps_per_epoch - batch_idx) / max(rate, 1e-9) / 60))
                log.writerow([epoch, batch_idx, np.mean(average_loss), round(time.time() - t0)])
                log_f.flush()
            if time.time() - last_save > save_every and batch_idx < steps_per_epoch - 1:
                save_state(run / "resume.pth", model, optimizer, lr_scheduler, epoch, batch_idx,
                           history, best_acc)
                last_save = time.time()
                print(f"  checkpoint saved (epoch {epoch}, batch {batch_idx})")

        scores = {v: score_video(model, cfg, "shanghai", v, device) for v in split["val"]}
        res, _ = frame_metrics(cfg, scores) if scores else ({}, None)
        frame_auc = res.get("auc_raw", float("nan"))           # repo: AUC of raw MSE
        history.append({"epoch": epoch, **res})
        best = not np.isnan(frame_auc) and frame_auc > best_acc
        best_acc = max(best_acc, frame_auc) if not np.isnan(frame_auc) else best_acc
        save_state(run / "current.pth", model, optimizer, lr_scheduler, epoch)   # repo names
        if best or not split["val"]:
            save_state(run / "best.pth", model, optimizer, lr_scheduler, epoch)
        (run / "resume.pth").unlink(missing_ok=True)          # current.pth is newer now
        last_save = time.time()
        if split["val"]:
            print("Test Epoch: {:03d}), AUC@1: {:.4f} {}".format(epoch, frame_auc, "(best)" if best else ""))
        else:
            print(f"Epoch {epoch:03d} done (no validation videos: best.pth = latest epoch)")
        (run / "metrics.json").write_text(json.dumps(
            {"best_val_auc_raw": best_acc, "history": history, "parameters": params,
             "train_minutes": round((time.time() - t0) / 60, 1)}, indent=2))
    log_f.close()
    print(f"done. Evaluate with:  python -m models.thesis_astt.evaluate --run {run}")


if __name__ == "__main__":
    main()
