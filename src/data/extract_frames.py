"""Cut the manual videos into frames.

ShanghaiTech is already stored as frames, so this script is only needed for
data/raw/manual. Output: data/interim/frames/manual/<video_name>/000000.jpg ...
plus a meta.json per video (original fps, size, frame count).

    python -m src.data.extract_frames
    python -m src.data.extract_frames --overwrite
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from src.utils import load_config


def extract_video(video_path: Path, out_dir: Path, resize, step: int, quality: int) -> dict:
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if resize:
                frame = cv2.resize(frame, tuple(resize), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        idx += 1
    cap.release()

    meta = {
        "source": str(video_path),
        "orig_fps": fps,
        "fps": fps / step if fps else None,
        "step": step,
        "orig_size": [orig_w, orig_h],
        "size": list(resize) if resize else [orig_w, orig_h],
        "orig_frames": idx,
        "frames": saved,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML that overrides params.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fcfg = cfg["frames"]
    src = Path(cfg["data"]["manual_videos"])
    dst = Path(cfg["data"]["frames_dir"]) / "manual"
    dst.mkdir(parents=True, exist_ok=True)   # DVC expects this output even with no videos
    exts = set(fcfg["video_exts"])
    videos = sorted(p for p in src.rglob("*") if p.suffix in exts)
    if not videos:
        print(f"No videos with extensions {sorted(exts)} found under {src}")
        return

    overwrite = args.overwrite or cfg["run"]["overwrite"]
    jobs = {}
    with ProcessPoolExecutor(max_workers=cfg["run"]["workers"]) as pool:
        for v in videos:
            out = dst / v.stem
            if (out / "meta.json").exists() and not overwrite:
                print(f"skip  {v.name} (already extracted)")
                continue
            jobs[pool.submit(extract_video, v, out, fcfg["resize"], fcfg["step"],
                             fcfg["jpg_quality"])] = v
        for fut in as_completed(jobs):
            meta = fut.result()
            print(f"done  {jobs[fut].name}: {meta['frames']} frames @ {meta['fps']} fps")


if __name__ == "__main__":
    main()
