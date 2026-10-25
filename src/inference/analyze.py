"""Analyze ONE video end to end: people, behavior events, model anomaly score, result video.
Used by the demo command below and by the API (api/main.py). Works with all 4 models.

    python -m src.inference.analyze --video path/to/clip.mp4                 # model from configs/api.yaml
    python -m src.inference.analyze --video clip.mp4 --model astt_baseline   # | astt_improved | aivad | aivad_lite
    python -m src.inference.analyze --video data/processed/shanghaitech/frames/01_0014   # a frame folder
    python -m src.inference.analyze --video clip.mp4 --model none            # behavior rules only

Out: outputs/demo/<video>/result.mp4 and result.json
Steps: 1 cut into frames  2 track people  3 behavior rules (loitering / counter-flow / too fast)
       4 model score per frame vs. the EER threshold from evaluation (+ for models 3/4: which
       object is unusual and why)  5 draw.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.behavior.detect import EVENTS, detect_events, summarize
from src.behavior.tracking import track_and_save
from src.config import evaluation_dir, load_config, training_dir
from src.data.frames import list_frames
from src.data.prepare_manual import extract_video
from src.evaluation.scoring import build_scorer
from src.inference.render import render_video

REASONS = {"velocity": "unusual motion", "relative_velocity": "moves unlike the crowd",
           "pose": "unusual pose", "appearance": "unusual look"}


class Analyzer:
    """Loads the model once; `run()` analyzes one video."""

    def __init__(self, model: str | None = None, checkpoint=None, threshold=None, device=None):
        api = load_config("configs/api.yaml")["api"]
        self.name = model or api["model"]
        if self.name in (None, "none"):
            self.name = None
        self.behavior = load_config("configs/behavior.yaml")
        self.device = torch.device(device or api["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scorer = self.cfg = self.threshold = self.checkpoint = None
        if self.name:
            self.cfg = load_config(f"configs/{self.name}.yaml")
            ckpt = Path(checkpoint or api["checkpoint"] or training_dir(self.cfg) / "model.pth")
            self.scorer = build_scorer(self.cfg, ckpt, self.device)
            self.threshold = threshold or api["threshold"]
            if self.threshold is None:
                m = evaluation_dir(self.cfg) / "metrics.json"
                if not m.exists():
                    raise SystemExit(f"No threshold: evaluate {self.name} first, or pass --threshold")
                self.threshold = json.loads(m.read_text())["threshold_raw_at_eer"]
            self.checkpoint = str(ckpt)

    def info(self) -> dict:
        return {"model": self.name or "none", "device": str(self.device),
                "score_type": None if self.scorer is None else self.scorer.kind,
                "threshold": self.threshold, "checkpoint": self.checkpoint}

    @staticmethod
    def unusual_objects(extra: dict) -> list[dict]:
        """Models 3/4: objects with an attribute above the training 99.9th percentile (> 1.0)."""
        f, obj = extra["features"], extra["objects"]
        best = np.zeros(len(f["frame"]))
        why = np.full(len(f["frame"]), "", object)
        for att, s in obj.items():
            rows = f["pose_object"] if att == "pose" else np.arange(len(f["frame"]))
            for r, v in zip(rows, s):
                if r >= 0 and v > best[r]:
                    best[r], why[r] = v, att
        return [{"frame": int(f["frame"][r]), "box": [round(float(x), 1) for x in f["boxes"][r]],
                 "score": round(float(best[r]), 3), "attribute": why[r], "reason": REASONS[why[r]]}
                for r in np.flatnonzero(best > 1.0) if "boxes" in f and not f["whole"][r]]

    def run(self, video: Path, work: Path, render: bool = True) -> dict:
        video, work = Path(video), Path(work)
        work.mkdir(parents=True, exist_ok=True)
        timings, t0 = {}, time.time()
        # 1. frames
        if video.is_dir():
            fdir, fps = video, 25.0
            if (video / "meta.json").exists():
                fps = json.loads((video / "meta.json").read_text()).get("fps") or fps
        else:
            fdir = work / "frames"
            meta = (json.loads((fdir / "meta.json").read_text()) if (fdir / "meta.json").exists()
                    else extract_video(video, fdir))
            fps = meta["fps"] or 25.0
        frames = list_frames(fdir)
        timings["frames_s"] = round(time.time() - t0, 2)
        # 2. people
        t1 = time.time()
        tf = work / "tracking.npz"
        if not tf.exists():
            track_and_save(self.behavior, frames, tf)
        tr = np.load(tf)
        tracks, vel, moving = tr["tracks"], tr["velocity"], tr["moving"]
        timings["tracking_s"] = round(time.time() - t1, 2)
        # 3. behavior rules (no camera map for a new video: compare with the crowd only)
        R, C = self.behavior["optical_flow"]["grid"]
        flags = detect_events(self.behavior, tracks, vel, moving, np.zeros((R, C, 2), np.float32),
                              np.zeros((R, C), np.float32), fps)
        events = summarize(tracks, flags, fps)
        # 4. model
        scores = maps = None
        unusual = []
        t2 = time.time()
        if self.scorer is not None:
            scores, extra = self.scorer(fdir, video.stem, details=True, cache=False, num_workers=0)
            maps = extra.get("maps")
            if "objects" in extra:
                unusual = self.unusual_objects(extra)
                fs = self.behavior["frame_size"]
                h, w = cv2.imread(str(frames[0])).shape[:2]
                for u in unusual:                                     # boxes in frame_size coords
                    u["box_draw"] = [u["box"][0] * fs[0] / w, u["box"][1] * fs[1] / h,
                                     u["box"][2] * fs[0] / w, u["box"][3] * fs[1] / h]
        timings["model_s"] = round(time.time() - t2, 2)
        # 5. result video
        result_video = None
        if render:
            t3 = time.time()
            result_video = str(render_video(work / "result.mp4", frames, self.behavior["frame_size"], fps,
                                            tracks, vel, moving, flags, scores, maps, self.threshold,
                                            self.name or "", unusual))
            timings["render_s"] = round(time.time() - t3, 2)
        timings["total_s"] = round(time.time() - t0, 2)

        abnormal = [] if scores is None else [int(i) for i in np.flatnonzero(np.nan_to_num(scores) > self.threshold)]
        for u in unusual:
            u.pop("box_draw", None)
        result = {
            "video": video.name, "frames": len(frames), "fps": fps, **self.info(),
            "abnormal_frames": abnormal,
            "abnormal_ratio": round(len(abnormal) / max(len(frames), 1), 4),
            "frame_scores": None if scores is None else [None if np.isnan(s) else round(float(s), 6) for s in scores],
            "unusual_objects": unusual,
            "people_tracked": int(len(np.unique(tracks[:, 1]))) if len(tracks) else 0,
            "events": events,
            "event_counts": {n: sum(e["event"] == n for e in events) for n, _ in EVENTS.values()},
            "result_video": result_video, "timings": timings}
        (work / "result.json").write_text(json.dumps(result, indent=2))
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file or folder of frames")
    ap.add_argument("--model", help="astt_baseline | astt_improved | aivad | aivad_lite | none")
    ap.add_argument("--threshold", type=float, help="frame threshold (default: EER from evaluation)")
    args = ap.parse_args()
    a = Analyzer(args.model, threshold=args.threshold)
    v = Path(args.video)
    r = a.run(v, Path(load_config()["paths"]["outputs"]) / "demo" / (v.stem if v.is_file() else v.name))
    print(f"{r['video']}: {len(r['abnormal_frames'])}/{r['frames']} abnormal frames, "
          f"{r['people_tracked']} people, {len(r['unusual_objects'])} unusual objects, events {r['event_counts']}")
    print(f"wrote {r['result_video']}")


if __name__ == "__main__":
    main()
