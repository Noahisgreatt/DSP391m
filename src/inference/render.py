"""Drawing the result video: a box + direction arrow per person, event tags, the model's
error heat-map on abnormal frames, and the frame-score curve along the bottom."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.behavior.detect import COUNTER, FAST, LOITER
from src.behavior.optical_flow import DIRECTION_NAMES_8, angle_to_bin

RED, YELLOW, WHITE = (0, 0, 255), (0, 200, 255), (255, 255, 255)
TAGS = ((COUNTER, "COUNTER-FLOW", RED), (FAST, "TOO FAST", RED), (LOITER, "LOITERING", YELLOW))


def bin_color(b: int, n_bins: int = 8):
    hsv = np.uint8([[[int(180 * b / n_bins), 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def label(img, text, x, y, color):
    """Text on a filled box, kept inside the frame."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = int(min(max(x, 0), img.shape[1] - tw - 4)), int(max(y, th + 4))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_legend(img):
    for i, name in enumerate(DIRECTION_NAMES_8):
        y = 16 + i * 16
        cv2.rectangle(img, (6, y - 10), (18, y + 2), bin_color(i), -1)
        cv2.putText(img, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)


def draw_people(img, rows, vel, moving, flags, arrow_scale=12.0, arrow_min=18):
    """One box + one arrow per person; flagged people get a thick box and their tags."""
    counts = {}
    for (_, tid, x1, y1, x2, y2, _), v, mv, fl in zip(rows, vel, moving, flags):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if mv:
            col = bin_color(int(angle_to_bin(np.arctan2(v[1], v[0]) % (2 * np.pi), 8)))
            speed = float(np.hypot(*v))
            length = max(speed * arrow_scale, arrow_min)
            end = (int(cx + v[0] / speed * length), int(cy + v[1] / speed * length))
            cv2.arrowedLine(img, (cx, cy), end, (0, 0, 0), 4, tipLength=0.3)
            cv2.arrowedLine(img, (cx, cy), end, col, 2, tipLength=0.3)
        else:
            col = (200, 200, 200)
            cv2.circle(img, (cx, cy), 4, col, -1)                  # standing still
        tags = [(t, c) for bit, t, c in TAGS if int(fl) & bit]
        cv2.rectangle(img, p1, p2, tags[0][1] if tags else col, 2 if tags else 1)
        if not tags:
            cv2.putText(img, str(int(tid)), (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        above = y1 > 22 * len(tags) + 18
        for j, (text, c) in enumerate(tags):
            ty = (y1 - 4 - 20 * (len(tags) - 1 - j)) if above else (y2 + 20 + 20 * j)
            label(img, f"#{int(tid)} {text}" if j == 0 else text, x1, ty, c)
            counts[(text, c)] = counts.get((text, c), 0) + 1
    H, W = img.shape[:2]
    for i, ((text, c), n) in enumerate(sorted(counts.items())):     # summary, bottom-right
        t = f"{text}: {n} " + ("people" if n > 1 else "person")
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label(img, t, W - tw - 10, H - 66 - i * 22, c)


def draw_curve(img, scores, t, thr, height=56):
    H, W = img.shape[:2]
    y0 = H - height
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
    valid = scores[~np.isnan(scores)]
    hi = max(float(valid.max()) if len(valid) else 1.0, thr * 1.5)
    n = len(scores)
    xy = lambda i, s: (int(i * (W - 1) / max(n - 1, 1)), int(H - 4 - (height - 8) * min(s / hi, 1)))
    cv2.line(img, (0, xy(0, thr)[1]), (W, xy(0, thr)[1]), YELLOW, 1)
    for i in range(1, t + 1):
        if not (np.isnan(scores[i - 1]) or np.isnan(scores[i])):
            cv2.line(img, xy(i - 1, scores[i - 1]), xy(i, scores[i]), RED if scores[i] > thr else WHITE, 1)


def paint_error(img, emap, thr, k=4.0):
    """Red overlay where the model's pixel error is k x the frame threshold."""
    m = cv2.resize(emap.astype(np.float32), (img.shape[1], img.shape[0])) > k * thr
    red = img.copy()
    red[m] = RED
    cv2.addWeighted(red, 0.45, img, 0.55, 0, img)


def draw_unusual(img, objs):
    """Models 3/4: box every object AI-VAD finds unusual, with the attribute that makes it so."""
    for o in objs:
        x1, y1, x2, y2 = (int(v) for v in o["box_draw"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label(img, f"{o['reason']} {o['score']:.1f}", x1, y2 + 18, (160, 0, 160))


def render_video(out_path: Path, frame_paths, size, fps, tracks, vel, moving, flags,
                 scores=None, maps=None, thr=None, model_name="model", unusual=()) -> Path:
    from src.data.frames import read_frame
    fidx = tracks[:, 0].astype(int)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
    for t, p in enumerate(frame_paths):
        img = read_frame(p, size)
        abnormal = scores is not None and not np.isnan(scores[t]) and scores[t] > thr
        if abnormal and maps is not None:
            paint_error(img, maps[t], thr)
        if scores is not None:
            draw_curve(img, scores, t, thr)
        if abnormal and unusual:
            draw_unusual(img, [o for o in unusual if o["frame"] == t])
        sel = fidx == t
        draw_people(img, tracks[sel], vel[sel], moving[sel], flags[sel])
        draw_legend(img)
        if abnormal:
            label(img, f"ABNORMAL FRAME ({model_name} {scores[t]:.4f})", img.shape[1] // 2 - 120, 24, RED)
        writer.write(img)
    writer.release()
    return out_path
