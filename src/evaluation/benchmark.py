"""How big and how fast is each model? Same video, same machine, one model at a time.

    python -m src.evaluation.benchmark                       (DVC stage: benchmark)
    python -m src.evaluation.benchmark --models aivad aivad_lite --frames 200

For every trained model: parameters (per network), time per frame of each step, frames per
second of the whole scoring pipeline, and peak GPU memory, on the first `--frames` frames of the
first test video. Out: outputs/comparison/benchmark.json and benchmark.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import frames_dir, load_config, training_dir
from src.data.split import load_split

MODELS = ("astt_baseline", "astt_improved", "aivad", "aivad_lite")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_astt(cfg, device, video_dir, n_frames):
    from src.data.dataset import video_dataset
    from src.evaluation.scoring import ASTTScorer, autocast
    from src.models import count_parameters
    sc = ASTTScorer(cfg, training_dir(cfg) / "model.pth", device)
    ds, _ = video_dataset(cfg, video_dir)
    T = cfg["model"]["num_frames"]
    x = torch.as_tensor(ds[0]["standard"][:T]).unsqueeze(0).to(device)
    with torch.no_grad(), autocast(cfg, device):
        for _ in range(3):
            sc.model(x)
        sync(device)
        t0, n = time.time(), min(n_frames, 50)
        for _ in range(n):
            sc.model(x)
        sync(device)
    net_ms = (time.time() - t0) / n * 1000
    return {"parameters": count_parameters(cfg, sc.model), "ms_per_frame": {"network": round(net_ms, 2)}}


def bench_aivad(cfg, device, video_dir, n_frames):
    from src.models.aivad.features import Extractor
    from src.models.aivad.pipeline import VideoScorer, count_parameters
    ex = Extractor(cfg, device)
    vs = VideoScorer(cfg, training_dir(cfg) / "model.pth", device, extractor=ex)
    ex.video(video_dir, None, max_frames=5)                  # warm-up
    ex.timing = {}
    t0 = time.time()
    feat = ex.video(video_dir, None, max_frames=n_frames)
    t1 = time.time()
    vs.scorer.score(video_dir.name, feat)
    t2 = time.time()
    n = int(feat["n_frames"])
    ms = {k: round(v / n * 1000, 2) for k, v in ex.timing.items()}
    ms["attributes (cpu)"] = round(((t1 - t0) - sum(ex.timing.values())) / n * 1000, 2)
    ms["density scoring"] = round((t2 - t1) / n * 1000, 2)
    return {"parameters": count_parameters(ex), "ms_per_frame": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_config()
    vid = load_split(base)["test"][0]
    video_dir = frames_dir(base, "shanghaitech") / vid
    res = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "video": vid, "frames": args.frames, "models": {}}
    for name in args.models:
        cfg = load_config(f"configs/{name}.yaml")
        if not (training_dir(cfg) / "model.pth").exists():
            print(f"{name}: not trained yet, skipped")
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        r = (bench_aivad if cfg["model"].get("family") == "aivad" else bench_astt)(cfg, device, video_dir, args.frames)
        total = sum(r["ms_per_frame"].values())
        r["ms_per_frame"]["total"] = round(total, 2)
        r["fps"] = round(1000 / max(total, 1e-9), 2)
        r["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None
        res["models"][name] = r
        print(name, json.dumps(r))
    out = Path(base["paths"]["outputs"]) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))
    md = [f"# Size and speed ({res['device']}, video {vid}, {args.frames} frames)", "",
          "| Model | Parameters | ms / frame (steps) | Frames / s | Peak GPU memory |", "|---|---|---|---|---|"]
    for name, r in res["models"].items():
        steps = ", ".join(f"{k} {v}" for k, v in r["ms_per_frame"].items() if k != "total")
        md.append(f"| {name} | {r['parameters']['Total'] / 1e6:.1f}M | {steps} | {r['fps']} | "
                  f"{r['peak_gpu_memory_mb'] or '-'} MB |")
    (out / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
