"""Step 6, INFERENCE: analyze ONE video end to end and draw the result. The API (api/main.py) uses it.

    python -m src.inference --video path/to/clip.mp4 --model mulde_improved
    python -m src.inference --video data/processed/shanghaitech/frames/01_0014 --model astt_baseline
    python -m src.inference --video clip.mp4 --model none          # behavior rules only

Steps: 1 cut into frames  2 find + track people  3 behavior rules (loitering / counter-flow /
too fast)  4 model score per frame vs. the EER threshold from evaluation  5 draw.
Out: outputs/inference/<video>/result.mp4 (H.264, plays in a browser) + result.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.behavior import (COUNTER, DIRECTIONS, EVENTS, FAST, LOITER, PeopleTracker, angle_to_bin, detect_events,
                          per_person, summarize)
from src.dataset import extract_video
from src.utils import list_frames, load_config, out_dir, read_frame

RED, YELLOW, WHITE = (0, 0, 255), (0, 200, 255), (255, 255, 255)
TAGS = ((COUNTER, "COUNTER-FLOW", RED), (FAST, "TOO FAST", RED), (LOITER, "LOITERING", YELLOW))
_TRACKERS: dict = {}                                     # one people tracker per device, shared by all models


def tracker(cfg, device):
    key = (PeopleTracker, str(device))
    if key not in _TRACKERS:
        _TRACKERS[key] = PeopleTracker(cfg, device)
    return _TRACKERS[key]


class Analyzer:
    """Loads the networks once; run() analyzes one video."""

    def __init__(self, model=None, threshold=None, device=None, allow_missing=False):
        """allow_missing (the API): a model that isn't trained / evaluated yet is skipped with a
        warning (behavior rules only) instead of stopping the service."""
        api = load_config("configs/api.yaml")["api"]
        self.name = None if (model or api["model"]) in (None, "none") else (model or api["model"])
        self.behavior = load_config("configs/behavior.yaml")
        self.device = torch.device(device or api["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.people = tracker(self.behavior, self.device)
        self.scorer = self.cfg = self.threshold = self.warning = None
        if self.name:
            cfg = load_config(f"configs/{self.name}.yaml")
            need = [out_dir(cfg, "training") / "model.pth"]
            if (threshold or api["threshold"]) is None:
                need.append(out_dir(cfg, "evaluation") / "metrics.json")
            missing = [str(p) for p in need if not p.exists()]
            if missing:
                msg = (f"model {self.name} is not ready (missing {', '.join(missing)}): train and evaluate it "
                       f"first (bash run_all.sh); serving the behavior rules only")
                if not allow_missing:
                    raise SystemExit(msg)
                print("WARNING:", msg)
                self.warning, self.name = msg, None
        if self.name:                                     # trained + evaluated (checked above)
            from src.models import scorer
            self.cfg = cfg
            self.scorer = scorer(cfg, out_dir(cfg, "training") / "model.pth", self.device)
            self.threshold = threshold or api["threshold"] or json.loads(
                (out_dir(cfg, "evaluation") / "metrics.json").read_text())["threshold_raw_at_eer"]

    def info(self):
        return {"model": self.name or "none", "device": str(self.device), "threshold": self.threshold,
                "score_type": None if self.scorer is None else self.scorer.kind, "warning": self.warning}

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
        t2 = time.time()
        if self.scorer is not None:
            scores, extra = self.scorer(fdir, video.stem, details=True, cache=False, num_workers=0)
            maps = extra.get("maps")
        tm["model_s"] = round(time.time() - t2, 2)
        warning = self.warning
        valid = np.array([]) if scores is None else scores[~np.isnan(scores)]
        if len(valid) and np.mean(valid > self.threshold) > 0.8:
            warning = ("The model finds almost every frame abnormal: this camera probably looks unlike the 13 "
                       "ShanghaiTech training cameras, so its scores are not calibrated here. Trust the people "
                       "boxes (behavior rules); for the model, train it on normal video from this camera.")
        result_video = None
        if draw:                                                      # 5. draw
            t3 = time.time()
            result_video = str(render(work / "result.mp4", frames, self.behavior["frame_size"], fps, tracks, vel, moving,
                                      flags, scores, maps, self.threshold, self.name or ""))
            tm["render_s"] = round(time.time() - t3, 2)
        tm["total_s"] = round(time.time() - t0, 2)
        abnormal = [] if scores is None else [int(i) for i in np.flatnonzero(np.nan_to_num(scores) > self.threshold)]
        result = {"video": video.name, "frames": len(frames), "fps": fps, **self.info(), "warning": warning,
                  "abnormal_frames": abnormal,
                  "abnormal_ratio": round(len(abnormal) / max(len(frames), 1), 4),
                  "frame_scores": None if scores is None else [None if np.isnan(s) else round(float(s), 6) for s in scores],
                  "people_tracked": int(len(np.unique(tracks[:, 1]))) if len(tracks) else 0,
                  "people": people_summary(tracks, vel, moving, flags, fps),
                  "events": events, "event_counts": {n: sum(e["event"] == n for e in events) for n, _ in EVENTS.values()},
                  "result_video": result_video, "timings": tm}
        (work / "result.json").write_text(json.dumps(result, indent=2))
        return result


def people_summary(tracks, vel, moving, flags, fps):
    """One line per tracked person: when seen, main walking direction, behaviors found."""
    out = []
    for tid, idx in per_person(tracks):
        mv = moving[idx].astype(bool)
        if mv.any():
            v = vel[idx][mv]
            bins = angle_to_bin(np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi), 8)
            direction = DIRECTIONS[int(np.bincount(bins, minlength=8).argmax())]
        else:
            direction = "standing"
        fr = tracks[idx, 0]
        out.append({"track_id": int(tid), "from_s": round(float(fr.min()) / fps, 2), "to_s": round(float(fr.max()) / fps, 2),
                    "direction": direction, "moving_share": round(float(mv.mean()), 2),
                    "events": [name for bit, (name, _) in EVENTS.items() if (flags[idx].astype(int) & bit).any()]})
    return out


def browser_mp4(path):
    """OpenCV writes MPEG-4 part 2, which browsers do not play: re-encode as H.264 when ffmpeg exists."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return path
    tmp = Path(path).with_suffix(".h264.mp4")
    r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(path), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                       capture_output=True)
    if r.returncode == 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)
    return path


# ====================================================================== drawing
def _color(b):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(180 * b / 8), 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _label(img, text, x, y, color, f=1.0):
    fs, th = 0.5 * f, max(1, round(f))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x, y = int(min(max(x, 0), img.shape[1] - tw - 4)), int(max(y, tht + 4))
    cv2.rectangle(img, (x, y - tht - 4), (x + tw + 4, y + 2 * th), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, th, cv2.LINE_AA)


def _people(img, rows, vel, moving, flags, sx=1.0, sy=1.0, f=1.0):
    """One box + one direction arrow per person; flagged people get a thick box and tags.
    Boxes are in frame_size coordinates; (sx, sy) scales them to the drawn video."""
    th = max(1, round(f))
    for (_, tid, x1, y1, x2, y2, _), v, mv, fl in zip(rows, vel, moving, flags):
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        col = (200, 200, 200)
        if mv:
            col = _color(int(angle_to_bin(np.arctan2(v[1], v[0]) % (2 * np.pi), 8)))
            s = float(np.hypot(*v))
            n = max(s * 12, 18) * f
            end = (int(cx + v[0] / s * n), int(cy + v[1] / s * n))
            cv2.arrowedLine(img, (cx, cy), end, (0, 0, 0), 2 * th + 2, tipLength=0.3)
            cv2.arrowedLine(img, (cx, cy), end, col, 2 * th, tipLength=0.3)
        else:
            cv2.circle(img, (cx, cy), 4 * th, col, -1)
        tags = [(t, c) for bit, t, c in TAGS if int(fl) & bit]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), tags[0][1] if tags else col, 2 * th if tags else th)
        for j, (text, c) in enumerate(tags):
            _label(img, f"#{int(tid)} {text}" if j == 0 else text, x1, y1 - 4 * f - 20 * f * (len(tags) - 1 - j), c, f)


def _curve(img, scores, t, thr, f=1.0):
    H, W = img.shape[:2]
    h, th = int(56 * f), max(1, round(f))
    ov = img.copy()
    cv2.rectangle(ov, (0, H - h), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
    valid = scores[~np.isnan(scores)]
    hi = max(float(valid.max()) if len(valid) else 1.0, thr * 1.5)
    xy = lambda i, s: (int(i * (W - 1) / max(len(scores) - 1, 1)), int(H - 4 - (h - 8) * min(s / hi, 1)))
    cv2.line(img, (0, xy(0, thr)[1]), (W, xy(0, thr)[1]), YELLOW, th)
    for i in range(1, t + 1):
        if not (np.isnan(scores[i - 1]) or np.isnan(scores[i])):
            cv2.line(img, xy(i - 1, scores[i - 1]), xy(i, scores[i]), RED if scores[i] > thr else WHITE, th)


def region_masks(maps, thr):
    """Models 1/2: WHERE a frame is abnormal = pixels whose prediction error is far above the usual
    error AT THAT PLACE in this video. Edges and textures are always hard to predict; without
    subtracting each pixel's median error the whole scene's outline would be painted red."""
    step = max(1, len(maps) // 200)                              # median of <= 200 frames: little memory
    usual = np.median(maps[::step].astype(np.float32), axis=0)
    kernel = np.ones((5, 5), np.uint8)
    return lambda t: cv2.morphologyEx(((maps[t].astype(np.float32) - usual) > 4 * thr).astype(np.uint8),
                                      cv2.MORPH_OPEN, kernel)    # drops single-pixel speckles


def render(out_path, frame_paths, size, fps, tracks, vel, moving, flags, scores=None, maps=None, thr=None,
           model_name="", max_width=1280):
    """Draw at the video's own resolution (at most max_width wide), not at the 640x360 the
    behavior steps work in, so the result is as sharp as the input."""
    fidx = tracks[:, 0].astype(int)
    h0, w0 = read_frame(frame_paths[0]).shape[:2]
    k = min(1.0, max_width / w0)
    out = (int(w0 * k) // 2 * 2, int(h0 * k) // 2 * 2)
    sx, sy, f = out[0] / size[0], out[1] / size[1], max(out[0] / 640, 1.0)
    region = region_masks(maps, thr) if maps is not None else None
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out)
    for t, p in enumerate(frame_paths):
        img = read_frame(p, out)
        bad = scores is not None and not np.isnan(scores[t]) and scores[t] > thr
        if bad and region is not None:                           # models 1/2: the unusual region
            m = cv2.resize(region(t), out, interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                red = img.copy()
                red[m] = RED
                cv2.addWeighted(red, 0.4, img, 0.6, 0, img)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, RED, max(1, round(f)))
        if scores is not None:
            _curve(img, scores, t, thr, f)
        if bad:
            _label(img, f"ABNORMAL FRAME ({model_name} {scores[t]:.3f})", out[0] // 2 - 120 * f, 24 * f, RED, f)
        sel = fidx == t
        _people(img, tracks[sel], vel[sel], moving[sel], flags[sel], sx, sy, f)
        for i, name in enumerate(DIRECTIONS):                     # arrow colour legend
            y = int((6 + i * 16) * f)
            cv2.rectangle(img, (int(6 * f), y), (int(18 * f), y + int(12 * f)), _color(i), -1)
            cv2.putText(img, name, (int(22 * f), y + int(10 * f)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * f, WHITE, max(1, round(f)))
        writer.write(img)
    writer.release()
    return browser_mp4(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file or folder of frames")
    ap.add_argument("--model", help="any name in src/models/__init__.py MODELS (e.g. astt_lite_smooth), or none")
    ap.add_argument("--threshold", type=float, help="frame threshold (default: EER from evaluation)")
    a = ap.parse_args()
    v = Path(a.video)
    r = Analyzer(a.model, a.threshold).run(v, Path(load_config()["paths"]["outputs"]) / "inference" / (v.stem if v.is_file() else v.name))
    print(f"{r['video']}: {len(r['abnormal_frames'])}/{r['frames']} abnormal frames, {r['people_tracked']} people, "
          f"events {r['event_counts']}\nwrote {r['result_video']}")


if __name__ == "__main__":
    main()
