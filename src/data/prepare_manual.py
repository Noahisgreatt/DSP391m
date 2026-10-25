"""Cut our own (manual) videos into frames + write the data card.

    python -m src.data.prepare_manual                   (DVC stage: prepare_manual)

In:   data/raw/manual/**/*.mp4|mov|...
Out:  data/processed/manual/frames/<video>/000000.jpg ... + meta.json (fps, size, frames)
      data/metadata/manual.json                          data card
`extract_video` is also used by the demo / API for uploaded videos.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from src.config import frames_dir, load_config


def extract_video(video_path: Path, out_dir: Path, size=None, step: int = 1,
                  quality: int = 95) -> dict:
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    orig = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if size:
                frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        idx += 1
    cap.release()
    meta = {"source": Path(video_path).name, "orig_fps": fps, "fps": fps / step if fps else None,
            "step": step, "orig_size": orig, "size": list(size) if size else orig,
            "orig_frames": idx, "frames": saved}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    cfg = load_config()
    f = cfg["frames"]
    src, dst = Path(cfg["paths"]["raw_manual"]), frames_dir(cfg, "manual")
    dst.mkdir(parents=True, exist_ok=True)
    exts = set(cfg["manual"]["video_exts"])
    videos = sorted(p for p in src.rglob("*") if p.suffix in exts)
    metas = {}
    with ProcessPoolExecutor(max_workers=cfg["run"]["workers"]) as pool:
        jobs = {pool.submit(extract_video, v, dst / v.stem, f["resize"], f["step"],
                            f["jpg_quality"]): v for v in videos}
        for fut in as_completed(jobs):
            m = fut.result()
            metas[jobs[fut].stem] = m
            print(f"{jobs[fut].name}: {m['frames']} frames @ {m['fps']} fps")
    card = {"dataset": "manual (own phone videos)",
            "videos": len(metas), "frames": sum(m["frames"] for m in metas.values()),
            "frame_size": f["resize"], "step": f["step"],
            "per_video": {k: {"frames": m["frames"], "fps": m["fps"]} for k, m in sorted(metas.items())}}
    meta = Path(cfg["paths"]["metadata"])
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "manual.json").write_text(json.dumps(card, indent=2))
    print(f"{card['videos']} videos, {card['frames']} frames")


if __name__ == "__main__":
    main()
