"""Behavior events per person: LOITERING, COUNTER_FLOW (against the crowd), TOO_FAST.

    python -m src.behavior.detect --dataset shanghaitech    (DVC stage: behavior@shanghaitech)
    python -m src.behavior.detect --dataset manual          (DVC stage: behavior@manual)

Uses tracking.py (people + velocities) and optical_flow.py (each camera's normal direction).
No training. Rules: counter_flow.py and loitering.py; thresholds: configs/behavior.yaml.
Out: outputs/behavior/<dataset>/events/<video>.json   readable list of events with times
     outputs/behavior/<dataset>/events/<video>.npz    flags per tracking row + per frame
     outputs/behavior/<dataset>/metrics.json          event counts (+ frame AUC on ShanghaiTech)
Flag bits: 1 = LOITERING, 2 = COUNTER_FLOW, 4 = TOO_FAST.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.behavior.counter_flow import detect_counter_flow
from src.behavior.loitering import detect_loitering
from src.behavior.optical_flow import scene_of
from src.config import behavior_dir, features_path, load_config, video_fps
from src.data.split import video_labels
from src.evaluation.metrics import binary_metrics

LOITER, COUNTER, FAST = 1, 2, 4
EVENTS = {LOITER: ("LOITERING", "stays in one place"),
          COUNTER: ("COUNTER_FLOW", "moving against the crowd"),
          FAST: ("TOO_FAST", "much faster than the crowd")}


def normal_map(cfg, dataset, vid):
    f = features_path(cfg, "normal_maps", dataset, scene_of(dataset, vid))
    if f.exists():
        d = np.load(f)
        return d["normal"], d["consistency"]
    R, C = cfg["optical_flow"]["grid"]                 # no map (e.g. an uploaded video):
    return np.zeros((R, C, 2), np.float32), np.zeros((R, C), np.float32)   # crowd rule only


def detect_events(cfg, tracks, vel, moving, normal, consistency, fps) -> np.ndarray:
    """flags (N,) uint8 per tracking row."""
    flags = np.zeros(len(tracks), np.uint8)
    if len(tracks) == 0:
        return flags
    against, fast = detect_counter_flow(tracks, vel, moving, normal, consistency,
                                        tuple(cfg["frame_size"]), fps, cfg["counter_flow"])
    flags[against] |= COUNTER
    flags[fast] |= FAST
    flags[detect_loitering(tracks, fps, cfg["loitering"])] |= LOITER
    return flags


def summarize(tracks, flags, fps: float) -> list[dict]:
    """Per-row flags -> event segments (one per person and type) with start / end times."""
    events, max_gap = [], int(fps * 0.5)
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        for bit, (name, reason) in EVENTS.items():
            frames = tracks[idx[(flags[idx] & bit) > 0], 0].astype(int)
            if len(frames) == 0:
                continue
            start = prev = frames[0]
            for fr in list(frames[1:]) + [None]:
                if fr is None or fr - prev > max_gap:
                    events.append({"track_id": int(tid), "event": name, "reason": reason,
                                   "start_frame": int(start), "end_frame": int(prev),
                                   "start_s": round(start / fps, 2), "end_s": round(prev / fps, 2)})
                    start = fr
                if fr is not None:
                    prev = fr
    return sorted(events, key=lambda e: e["start_frame"])


def analyze(cfg, dataset, vid, tracking_file):
    """-> flags per row, frame_flags per frame, events list, fps."""
    tr = np.load(tracking_file)
    fps = video_fps(cfg, dataset, vid)
    normal, cons = normal_map(cfg, dataset, vid)
    flags = detect_events(cfg, tr["tracks"], tr["velocity"], tr["moving"], normal, cons, fps)
    frame_flags = np.zeros(len(tr["frame_names"]), np.uint8)
    np.bitwise_or.at(frame_flags, tr["tracks"][:, 0].astype(int), flags)
    return flags, frame_flags, summarize(tr["tracks"], flags, fps), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["shanghaitech", "manual"])
    args = ap.parse_args()
    cfg = load_config("configs/behavior.yaml")
    out = behavior_dir(cfg, args.dataset)
    (out / "events").mkdir(parents=True, exist_ok=True)
    files = sorted(features_path(cfg, "tracking", args.dataset, "x").parent.glob("*.npz"))
    counts = {name: 0 for name, _ in EVENTS.values()}
    ys, ff = [], []
    for f in files:
        vid = f.stem
        flags, frame_flags, events, fps = analyze(cfg, args.dataset, vid, f)
        np.savez_compressed(out / "events" / f"{vid}.npz", flags=flags, frame_flags=frame_flags)
        (out / "events" / f"{vid}.json").write_text(json.dumps(
            {"video": vid, "fps": fps, "events": events}, indent=2))
        for e in events:
            counts[e["event"]] += 1
        if args.dataset == "shanghaitech":
            ys.append(video_labels(cfg, vid, len(frame_flags)))
            ff.append(frame_flags > 0)
        print(f"{vid}: " + ", ".join(f"{sum(e['event'] == n for e in events)} {n.lower()}"
                                     for n in counts))
    metrics = {"videos": len(files), "events": counts}
    if ys:
        metrics["frame_level"] = binary_metrics(np.concatenate(ys), np.concatenate(ff))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
