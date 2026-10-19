"""Draw the per-cell direction arrows on a video so you can check them by eye.

    python -m src.visualize --dataset shanghai --video 01_0016
    python -m src.visualize --dataset manual --video my_clip --max-frames 300

Output: results/visualize/<method>/<dataset>/<video>.mp4
Arrow colour encodes direction (same colours as the legend in the top-left corner),
arrow length encodes speed, the grid shows the cells.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from src.features.direction import DIRECTION_NAMES_8, angle_to_bin
from src.utils import (features_path, frames_root, list_frames, load_config, read_frame,
                       video_fps)


def bin_color(b: int, n_bins: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[int(180 * b / n_bins), 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw(frame: np.ndarray, mean: np.ndarray, dominant: np.ndarray, n_bins: int,
         scale: float) -> np.ndarray:
    H, W = frame.shape[:2]
    R, C = dominant.shape
    out = frame.copy()
    for r in range(1, R):
        cv2.line(out, (0, r * H // R), (W, r * H // R), (90, 90, 90), 1)
    for c in range(1, C):
        cv2.line(out, (c * W // C, 0), (c * W // C, H), (90, 90, 90), 1)
    for r in range(R):
        for c in range(C):
            b = int(dominant[r, c])
            if b < 0:
                continue
            cx, cy = int((c + 0.5) * W / C), int((r + 0.5) * H / R)
            dx, dy = mean[r, c] * scale
            end = (int(cx + dx), int(cy + dy))
            cv2.arrowedLine(out, (cx, cy), end, (0, 0, 0), 4, tipLength=0.35)
            cv2.arrowedLine(out, (cx, cy), end, bin_color(b, n_bins), 2, tipLength=0.35)
    if n_bins == 8:  # legend
        for i, name in enumerate(DIRECTION_NAMES_8):
            y = 16 + i * 16
            cv2.rectangle(out, (6, y - 10), (18, y + 2), bin_color(i, 8), -1)
            cv2.putText(out, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return out


def draw_legend(out: np.ndarray) -> None:
    for i, name in enumerate(DIRECTION_NAMES_8):
        y = 16 + i * 16
        cv2.rectangle(out, (6, y - 10), (18, y + 2), bin_color(i, 8), -1)
        cv2.putText(out, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


RED, YELLOW = (0, 0, 255), (0, 200, 255)


def label(img, text, x, y, color):
    """Text on a filled box, kept inside the frame."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = int(min(max(x, 0), img.shape[1] - tw - 4))
    y = int(max(y, th + 4))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)


RULE_TAGS = ((2, "ABNORMAL: against crowd", RED), (4, "ABNORMAL: too fast", RED),
             (1, "LOITERING", YELLOW))


def person_tags(flag: int, model_score=None, model_thr=None, model_name="model"):
    """Tags (text, colour) for one person: rule reasons + the learned model's verdict."""
    tags = [(t, c) for b, t, c in RULE_TAGS if flag & b]
    if model_score is not None and model_thr is not None and model_score > model_thr:
        tags.insert(0, (f"ABNORMAL ({model_name} {model_score:.2f})", RED))
    return tags


def draw_events(img, rows, flags, scores=None, thr=None, model_name="model"):
    """Thick box + one tag line per reason for every flagged person, plus a frame banner.

    flags: rule bits per row (or None). scores: learned per-person anomaly score per row.
    """
    counts = {}
    for k, (_, tid, x1, y1, x2, y2, _) in enumerate(rows):
        tags = person_tags(int(flags[k]) if flags is not None else 0,
                           None if scores is None or np.isnan(scores[k]) else float(scores[k]),
                           thr, model_name)
        if not tags:
            continue
        color = RED if any(c == RED for _, c in tags) else YELLOW
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        above = y1 > 22 * len(tags) + 18
        for j, (text, c) in enumerate(tags):         # stacked above the box (or below at the top)
            ty = (y1 - 4 - 20 * (len(tags) - 1 - j)) if above else (y2 + 20 + 20 * j)
            label(img, f"#{int(tid)} {text}" if j == 0 else text, x1, ty, c)
            key = re.sub(r"\s[\d.]+\)$", ")", text)      # "ABNORMAL (crowdformer)"
            counts[key] = counts.get(key, 0) + 1
    H, W = img.shape[:2]
    for i, (text, n) in enumerate(sorted(counts.items())):   # summary, bottom-right
        t = f"{text}: {n} person" + ("s" if n > 1 else "")
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label(img, t, W - tw - 10, H - 10 - i * 22, YELLOW if text == "LOITERING" else RED)


def draw_tracks(frame: np.ndarray, rows: np.ndarray, vel: np.ndarray, moving: np.ndarray,
                scale: float, min_len: float, flags=None, scores=None, thr=None,
                model_name="model") -> np.ndarray:
    """One box and ONE arrow per tracked person, plus ABNORMAL / LOITERING tags."""
    out = frame.copy()
    for (_, tid, x1, y1, x2, y2, _), v, mv in zip(rows, vel, moving):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if mv:
            b = int(angle_to_bin(np.arctan2(v[1], v[0]) % (2 * np.pi), 8))
            col = bin_color(b, 8)
            speed = float(np.hypot(*v))
            length = max(speed * scale, min_len)
            end = (int(cx + v[0] / speed * length), int(cy + v[1] / speed * length))
            cv2.rectangle(out, p1, p2, col, 1)
            cv2.arrowedLine(out, (cx, cy), end, (0, 0, 0), 4, tipLength=0.3)
            cv2.arrowedLine(out, (cx, cy), end, col, 2, tipLength=0.3)
        else:
            col = (200, 200, 200)
            cv2.rectangle(out, p1, p2, col, 1)
            cv2.circle(out, (cx, cy), 4, col, -1)        # standing still
        cv2.putText(out, str(int(tid)), (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, col, 1)
    if flags is not None or scores is not None:
        draw_events(out, rows, flags, scores, thr, model_name)
    draw_legend(out)
    return out


def write_video(frames_iter, out_path: Path, fps: float) -> int:
    writer, n = None, 0
    for img in frames_iter:
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
        n += 1
    if writer:
        writer.release()
    return n


def main_tracks(args, cfg):
    feat_file = features_path(cfg, "tracking", args.dataset, args.video)
    if not feat_file.exists():
        raise FileNotFoundError(f"{feat_file} missing; run src.features.tracking first")
    f = np.load(feat_file)
    tracks, vel, moving = f["tracks"], f["velocity"], f["moving"]
    by_name = {p.name: p for p in list_frames(frames_root(cfg, args.dataset) / args.video)}
    names = list(f["frame_names"])
    n = len(names) if args.max_frames is None else min(args.max_frames, len(names))
    frame_idx = tracks[:, 0].astype(int)
    v = cfg["visualize"]

    ev_file = features_path(cfg, "events", args.dataset, args.video)
    flags = None
    if ev_file.exists():
        flags = np.load(ev_file)["flags"]
        if len(flags) != len(tracks):
            print("events file is older than the tracks; re-run src.baseline_rules. Showing tracks only.")
            flags = None
    else:
        print(f"no events for {args.video}; run `python -m src.baseline_rules --dataset {args.dataset}` "
              "to show LOITERING / ABNORMAL labels")

    def gen():
        for t in range(n):
            sel = frame_idx == t
            frame = read_frame(by_name[names[t]], cfg["frames"]["resize"])
            yield draw_tracks(frame, tracks[sel], vel[sel], moving[sel],
                              v["track_arrow_scale"], v["track_arrow_min"],
                              None if flags is None else flags[sel])

    out_path = Path(args.out or f"results/visualize/tracking/{args.dataset}/{args.video}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = video_fps(cfg, args.dataset, args.video)       # play back at real speed
    print(f"wrote {out_path} ({write_video(gen(), out_path, fps)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghai", "manual"])
    ap.add_argument("--video", required=True, help="video id, e.g. 01_0016")
    ap.add_argument("--mode", default="tracks", choices=["tracks", "grid"],
                    help="tracks = one arrow per person; grid = one arrow per flow cell")
    ap.add_argument("--config", help="YAML that overrides params.yaml")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", help="output .mp4 path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "tracks":
        return main_tracks(args, cfg)
    method = cfg["flow"]["method"]
    feat_file = features_path(cfg, method, args.dataset, args.video)
    if not feat_file.exists():
        raise FileNotFoundError(f"{feat_file} missing; run src.features.optical_flow first")
    f = np.load(feat_file)
    mean, dominant, n_bins = f["mean"].astype(np.float32), f["dominant"], int(f["n_bins"])

    frame_dir = frames_root(cfg, args.dataset) / args.video
    by_name = {p.name: p for p in list_frames(frame_dir)}
    names = list(f["frame_names"])

    out_path = Path(args.out or f"results/visualize/{method}/{args.dataset}/{args.video}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resize = cfg["frames"]["resize"]
    n = len(dominant) if args.max_frames is None else min(args.max_frames, len(dominant))

    writer = None
    for t in range(n):
        frame = read_frame(by_name[names[t + 1]], resize)  # row t = motion t -> t+1
        img = draw(frame, mean[t], dominant[t], n_bins, cfg["visualize"]["arrow_scale"])
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     cfg["visualize"]["fps"], (w, h))
        writer.write(img)
    if writer:
        writer.release()
    print(f"wrote {out_path} ({n} frames)")


if __name__ == "__main__":
    main()
