"""Step 6, DEMO: analyze ONE video end to end and draw the result. The API (api/main.py) uses it.

    python -m src.demo --video path/to/clip.mp4 --model aivad_lite
    python -m src.demo --video data/processed/shanghaitech/frames/01_0014 --model astt_baseline
    python -m src.demo --video clip.mp4 --model none          # behavior rules only

Steps: 1 cut into frames  2 find + track people  3 behavior rules (loitering / counter-flow /
too fast)  4 model score per frame vs. the EER threshold from evaluation (models 3/4 also: which
object is unusual and why)  5 draw.   Out: outputs/demo/<video>/result.mp4 + result.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.behavior import COUNTER, DIRECTIONS, EVENTS, FAST, LOITER, PeopleTracker, angle_to_bin, detect_events, summarize
from src.data import extract_video
from src.utils import list_frames, load_config, out_dir, read_frame

REASONS = {"velocity": "unusual motion", "relative_velocity": "moves unlike the crowd", "pose": "unusual pose",
           "appearance": "unusual look"}
RED, YELLOW, WHITE, PURPLE = (0, 0, 255), (0, 200, 255), (255, 255, 255), (160, 0, 160)
TAGS = ((COUNTER, "COUNTER-FLOW", RED), (FAST, "TOO FAST", RED), (LOITER, "LOITERING", YELLOW))


class Analyzer:
    """Loads the networks once; run() analyzes one video."""

    def __init__(self, model=None, threshold=None, device=None):
        api = load_config("configs/api.yaml")["api"]
        self.name = None if (model or api["model"]) in (None, "none") else (model or api["model"])
        self.behavior = load_config("configs/behavior.yaml")
        self.device = torch.device(device or api["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.people = PeopleTracker(self.behavior, self.device)
        self.scorer = self.cfg = self.threshold = None
        if self.name:
            from src.models import scorer
            self.cfg = load_config(f"configs/{self.name}.yaml")
            self.scorer = scorer(self.cfg, out_dir(self.cfg, "training") / "model.pth", self.device)
            self.threshold = threshold or api["threshold"]
            if self.threshold is None:
                m = out_dir(self.cfg, "evaluation") / "metrics.json"
                if not m.exists():
                    raise SystemExit(f"No threshold: evaluate {self.name} first, or pass --threshold")
                self.threshold = json.loads(m.read_text())["threshold_raw_at_eer"]

    def info(self):
        return {"model": self.name or "none", "device": str(self.device), "threshold": self.threshold,
                "score_type": None if self.scorer is None else self.scorer.kind}

    def run(self, video, work, draw=True) -> dict:
        video, work = Path(video), Path(work)
        work.mkdir(parents=True, exist_ok=True)
        tm, t0 = {}, time.time()
        if video.is_dir():                                            # 1. frames
            fdir, meta = video, video / "meta.json"
            fps = (json.loads(meta.read_text()).get("fps") if meta.exists() else None) or 25.0
        else:
            fdir = work / "frames"
            meta = json.loads((fdir / "meta.json").read_text()) if (fdir / "meta.json").exists() else extract_video(video, fdir)
            fps = meta["fps"] or 25.0
        frames = list_frames(fdir)
        tm["frames_s"] = round(time.time() - t0, 2)
        t1 = time.time()                                              # 2. people
        tr = self.people.video(frames)
        tracks, vel, moving = tr["tracks"], tr["velocity"], tr["moving"]
        tm["tracking_s"] = round(time.time() - t1, 2)
        R, C = self.behavior["optical_flow"]["grid"]                  # 3. rules (crowd only: no camera map)
        flags = detect_events(self.behavior, tracks, vel, moving, np.zeros((R, C, 2), np.float32), np.zeros((R, C), np.float32), fps)
        events = summarize(tracks, flags, fps)
        scores = maps = None                                          # 4. model
        unusual, t2 = [], time.time()
        if self.scorer is not None:
            scores, extra = self.scorer(fdir, video.stem, details=True, cache=False, num_workers=0)
            maps = extra.get("maps")
            if "objects" in extra:
                unusual = unusual_objects(extra)
        tm["model_s"] = round(time.time() - t2, 2)
        result_video = None
        if draw:                                                      # 5. draw
            t3 = time.time()
            result_video = str(render(work / "result.mp4", frames, self.behavior["frame_size"], fps, tracks, vel, moving,
                                      flags, scores, maps, self.threshold, self.name or "", unusual))
            tm["render_s"] = round(time.time() - t3, 2)
        tm["total_s"] = round(time.time() - t0, 2)
        abnormal = [] if scores is None else [int(i) for i in np.flatnonzero(np.nan_to_num(scores) > self.threshold)]
        result = {"video": video.name, "frames": len(frames), "fps": fps, **self.info(), "abnormal_frames": abnormal,
                  "abnormal_ratio": round(len(abnormal) / max(len(frames), 1), 4),
                  "frame_scores": None if scores is None else [None if np.isnan(s) else round(float(s), 6) for s in scores],
                  "unusual_objects": unusual, "people_tracked": int(len(np.unique(tracks[:, 1]))) if len(tracks) else 0,
                  "events": events, "event_counts": {n: sum(e["event"] == n for e in events) for n, _ in EVENTS.values()},
                  "result_video": result_video, "timings": tm}
        (work / "result.json").write_text(json.dumps(result, indent=2))
        return result


def unusual_objects(extra) -> list[dict]:
    """Models 3/4: objects with an attribute above the training 99.9th percentile (score > 1)."""
    f, obj = extra["features"], extra["objects"]
    best, why = np.zeros(len(f["frame"])), np.full(len(f["frame"]), "", object)
    for att, s in obj.items():
        rows = f["pose_object"] if att == "pose" else np.arange(len(f["frame"]))
        for r, v in zip(rows, s):
            if r >= 0 and v > best[r]:
                best[r], why[r] = v, att
    return [{"frame": int(f["frame"][r]), "box": [round(float(x), 1) for x in f["boxes"][r]], "score": round(float(best[r]), 3),
             "attribute": why[r], "reason": REASONS[why[r]]}
            for r in np.flatnonzero(best > 1.0) if "boxes" in f and not f["whole"][r]]


# ====================================================================== drawing
def _color(b):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(180 * b / 8), 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = int(min(max(x, 0), img.shape[1] - tw - 4)), int(max(y, th + 4))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def _people(img, rows, vel, moving, flags):
    """One box + one direction arrow per person; flagged people get a thick box and tags."""
    for (_, tid, x1, y1, x2, y2, _), v, mv, fl in zip(rows, vel, moving, flags):
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        col = (200, 200, 200)
        if mv:
            col = _color(int(angle_to_bin(np.arctan2(v[1], v[0]) % (2 * np.pi), 8)))
            s = float(np.hypot(*v))
            end = (int(cx + v[0] / s * max(s * 12, 18)), int(cy + v[1] / s * max(s * 12, 18)))
            cv2.arrowedLine(img, (cx, cy), end, (0, 0, 0), 4, tipLength=0.3)
            cv2.arrowedLine(img, (cx, cy), end, col, 2, tipLength=0.3)
        else:
            cv2.circle(img, (cx, cy), 4, col, -1)
        tags = [(t, c) for bit, t, c in TAGS if int(fl) & bit]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), tags[0][1] if tags else col, 2 if tags else 1)
        for j, (text, c) in enumerate(tags):
            _label(img, f"#{int(tid)} {text}" if j == 0 else text, x1, y1 - 4 - 20 * (len(tags) - 1 - j), c)


def _curve(img, scores, t, thr, h=56):
    H, W = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, H - h), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
    valid = scores[~np.isnan(scores)]
    hi = max(float(valid.max()) if len(valid) else 1.0, thr * 1.5)
    xy = lambda i, s: (int(i * (W - 1) / max(len(scores) - 1, 1)), int(H - 4 - (h - 8) * min(s / hi, 1)))
    cv2.line(img, (0, xy(0, thr)[1]), (W, xy(0, thr)[1]), YELLOW, 1)
    for i in range(1, t + 1):
        if not (np.isnan(scores[i - 1]) or np.isnan(scores[i])):
            cv2.line(img, xy(i - 1, scores[i - 1]), xy(i, scores[i]), RED if scores[i] > thr else WHITE, 1)


def render(out_path, frame_paths, size, fps, tracks, vel, moving, flags, scores=None, maps=None, thr=None,
           model_name="", unusual=()):
    fidx = tracks[:, 0].astype(int)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
    for t, p in enumerate(frame_paths):
        img = read_frame(p, size)
        if t == 0:
            h0, w0 = cv2.imread(str(p)).shape[:2]
            sx, sy = size[0] / w0, size[1] / h0
        bad = scores is not None and not np.isnan(scores[t]) and scores[t] > thr
        if bad and maps is not None:                              # models 1/2: where it failed to predict
            m = cv2.resize(maps[t].astype(np.float32), tuple(size)) > 4 * thr
            red = img.copy()
            red[m] = RED
            cv2.addWeighted(red, 0.45, img, 0.55, 0, img)
        if scores is not None:
            _curve(img, scores, t, thr)
        if bad:
            for o in (o for o in unusual if o["frame"] == t):      # models 3/4: which object and why
                x1, y1, x2, y2 = o["box"][0] * sx, o["box"][1] * sy, o["box"][2] * sx, o["box"][3] * sy
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                _label(img, f"{o['reason']} {o['score']:.1f}", x1, y2 + 18, PURPLE)
            _label(img, f"ABNORMAL FRAME ({model_name} {scores[t]:.3f})", img.shape[1] // 2 - 120, 24, RED)
        sel = fidx == t
        _people(img, tracks[sel], vel[sel], moving[sel], flags[sel])
        for i, name in enumerate(DIRECTIONS):                     # arrow colour legend
            cv2.rectangle(img, (6, 6 + i * 16), (18, 18 + i * 16), _color(i), -1)
            cv2.putText(img, name, (22, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        writer.write(img)
    writer.release()
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file or folder of frames")
    ap.add_argument("--model", help="astt_baseline | astt_improved | aivad | aivad_lite | none")
    ap.add_argument("--threshold", type=float, help="frame threshold (default: EER from evaluation)")
    a = ap.parse_args()
    v = Path(a.video)
    r = Analyzer(a.model, a.threshold).run(v, Path(load_config()["paths"]["outputs"]) / "demo" / (v.stem if v.is_file() else v.name))
    print(f"{r['video']}: {len(r['abnormal_frames'])}/{r['frames']} abnormal frames, {r['people_tracked']} people, "
          f"{len(r['unusual_objects'])} unusual objects, events {r['event_counts']}\nwrote {r['result_video']}")


if __name__ == "__main__":
    main()
