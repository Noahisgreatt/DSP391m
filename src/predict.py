"""Run a trained model on one video and write a result video that marks WHO is abnormal.

    python -m src.predict --run runs/crowdformer/<run_id> --video path/to/clip.MOV
    python -m src.predict --run runs/thesis_astt/<run_id> --video path/to/clip.MOV
    python -m src.predict --run runs/crowdformer/<run_id> --dataset shanghai --id 01_0014

Every tracked person gets a box and a direction arrow. Tags on a person:
  ABNORMAL (crowdformer 1.23)   learned model: this person moves very differently from what the
                                model expected given the crowd around them (CrowdFormer runs only)
  ABNORMAL: against crowd       rule: walking against the nearby crowd
  ABNORMAL: too fast            rule: much faster than the people around (running, cycling)
  LOITERING                     rule: stayed in one small area for `loiter_seconds`
ASTT runs: ASTT scores whole frames, so it can't name a person. Instead the regions it failed to
predict are painted red, and the frame gets an "ABNORMAL (ASTT)" banner above the threshold.
A score curve runs along the bottom.

A video file is cut into frames (data/interim/frames/other/<name>) and tracked
(data/features/tracking/other/<name>.npz) first. "other" is kept apart from the DVC-tracked
manual dataset, so trying a video never changes the pipeline's outputs.
Outputs: results/<model>/<run_id>/videos/<video>.mp4 and <video>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils import (features_path, frames_root, list_frames, load_config, read_frame,
                       results_dir, video_fps)

RED = (0, 0, 255)


# ---------------------------------------------------------------- inputs
def prepare_video(args, params):
    """Returns (dataset, video_id), extracting + tracking a raw video file if needed."""
    if args.id:
        return args.dataset, args.id
    src = Path(args.video)
    vid = src.stem if src.is_file() else src.name
    fdir = frames_root(params, "other") / vid
    if src.is_file() and not (fdir / "meta.json").exists():
        from src.data.extract_frames import extract_video
        print(f"extracting frames of {src.name} ...")
        extract_video(src, fdir, params["frames"]["resize"], params["frames"]["step"], 95)
    elif src.is_dir():                                   # a folder of frames: link it in
        fdir.mkdir(parents=True, exist_ok=True)
        for f in list_frames(src):
            if not (fdir / f.name).exists():
                (fdir / f.name).symlink_to(f.resolve())
    tf = features_path(params, "tracking", "other", vid)
    if not tf.exists():
        print("tracking people (YOLO + ByteTrack) ...")
        from src.features.direction import track_velocities
        from src.features.tracking import track_video
        t = params["tracking"]
        frames = list_frames(fdir)
        tracks = track_video(t["model"], frames, t, params["frames"]["resize"])
        vel, moving = track_velocities(tracks, t["window"], t["smooth"], t["min_speed"])
        tf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tf, tracks=tracks, velocity=vel, moving=moving,
                            frame_names=np.array([f.name for f in frames]))
    return "other", vid


def rule_flags(params, dataset, vid, tracks, vel, moving):
    """LOITERING / ABNORMAL rule bits per tracking row (uses src.baseline_rules results if present)."""
    ef = features_path(params, "events", dataset, vid)
    if ef.exists():
        f = np.load(ef)["flags"]
        if len(f) == len(tracks):
            return f
    from src.baseline_rules import detect_events
    d = params["detection"]
    R, C = params["direction"]["grid"]
    mp = Path(params["data"]["features_dir"]) / "normal_maps" / dataset / f"{vid}.npz"
    normal, cons = ((np.load(mp)["normal"], np.load(mp)["consistency"]) if mp.exists()
                    else (np.zeros((R, C, 2), np.float32), np.zeros((R, C), np.float32)))
    return detect_events(tracks, vel, moving, normal, cons, tuple(params["frames"]["resize"]),
                         video_fps(params, dataset, vid), d)


def threshold_from(run: Path, given, key="threshold_raw_at_eer"):
    if given is not None:
        return given
    for f in ("metrics_test.json", "metrics_val.json"):
        if (results_dir(run) / f).exists():
            thr = json.loads((results_dir(run) / f).read_text()).get(key)
            if thr:
                print(f"{key} {thr:.4f} from {f}")
                return thr
    raise SystemExit("No threshold yet: run the evaluate script first, or pass --threshold")


def sustained_person_scores(tracks, row_scores, thr, min_frames):
    """Keep a person's score only where it stays above thr for >= min_frames (no flicker)."""
    out = np.full_like(row_scores, np.nan)
    from src.baseline_rules import _sustain
    above = np.nan_to_num(row_scores) > thr
    for tid in np.unique(tracks[:, 1]):
        idx = np.where(tracks[:, 1] == tid)[0]
        idx = idx[np.argsort(tracks[idx, 0])]
        keep = idx[_sustain(above, idx, min_frames)]
        out[keep] = row_scores[keep]
    return out


# ---------------------------------------------------------------- model scores
@torch.no_grad()
def crowd_scores(run, dataset, vid, device):
    from models.crowdformer.evaluate import load_crowd_run, score_persons
    cfg, _, model, _ = load_crowd_run(run, device)
    params = load_config()
    cfg["data"]["frames_dir"] = params["data"]["frames_dir"]
    fs, rs = score_persons(model, cfg, dataset, [vid], device, cfg["eval"]["batch_size"],
                           num_workers=2)[vid]
    return fs, rs


# ---------------------------------------------------------------- drawing
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
    cv2.line(img, (0, xy(0, thr)[1]), (W, xy(0, thr)[1]), (0, 200, 255), 1)
    for i in range(1, t + 1):
        if not (np.isnan(scores[i - 1]) or np.isnan(scores[i])):
            cv2.line(img, xy(i - 1, scores[i - 1]), xy(i, scores[i]),
                     RED if scores[i] > thr else (255, 255, 255), 1)
    if not np.isnan(scores[t]):
        cv2.circle(img, xy(t, scores[t]), 3, (255, 255, 255), -1)


def paint_error(img, emap, thr, k=4.0):
    """Red overlay where ASTT's pixel error is k x the frame threshold."""
    m = cv2.resize(emap.astype(np.float32), (img.shape[1], img.shape[0])) > k * thr
    red = img.copy()
    red[m] = (0, 0, 255)
    cv2.addWeighted(red, 0.45, img, 0.55, 0, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--video", help="video file or frame folder")
    ap.add_argument("--dataset", choices=["shanghai", "manual", "other"], default="manual")
    ap.add_argument("--id", help="video id inside --dataset")
    ap.add_argument("--threshold", type=float, help="frame-level threshold (default: EER)")
    ap.add_argument("--person-threshold", type=float,
                    help="CrowdFormer person threshold (default: 99th pct of normal people)")
    ap.add_argument("--min-seconds", type=float, default=0.5,
                    help="a person must stay above the threshold this long to be tagged")
    args = ap.parse_args()
    if not (args.video or args.id):
        ap.error("give --video or --id")

    run = Path(args.run)
    import yaml
    kind = yaml.safe_load((run / "config.yaml").read_text())["experiment"].get("type", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = load_config()
    dataset, vid = prepare_video(args, params)
    frame_dir = frames_root(params, dataset) / vid
    thr = threshold_from(run, args.threshold)          # frame level (banner + curve)

    tr = np.load(features_path(params, "tracking", dataset, vid))
    tracks, vel, moving = tr["tracks"], tr["velocity"], tr["moving"]
    flags = rule_flags(params, dataset, vid, tracks, vel, moving)

    row_scores, emaps = None, None
    person_thr = thr
    if kind == "crowdformer":
        frame_score, row_scores = crowd_scores(run, dataset, vid, device)
        person_thr = threshold_from(run, args.person_threshold, "person_threshold_p99")
        min_frames = max(1, int(round(args.min_seconds * video_fps(params, dataset, vid))))
        row_scores = sustained_person_scores(tracks, row_scores, person_thr, min_frames)
    elif kind == "thesis_astt":
        from models.thesis_astt.evaluate import load_thesis_run
        from models.thesis_astt.train import score_video
        tcfg, _, tmodel, _ = load_thesis_run(run, device)
        tcfg["data"]["frames_dir"] = params["data"]["frames_dir"]
        frame_score, emaps = score_video(tmodel, tcfg, dataset, vid, device, return_maps=True)
    else:
        raise SystemExit(f"unknown model type {kind!r}")
    abnormal = np.nan_to_num(frame_score, nan=0) > thr

    from src.visualize import draw_tracks, label
    out_dir = results_dir(run) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(tr["frame_names"])
    fidx = tracks[:, 0].astype(int)
    v = params["visualize"]
    writer = None
    n_people_flagged = set()
    for t, name in enumerate(names):
        img = read_frame(frame_dir / name, params["frames"]["resize"])
        if emaps is not None and abnormal[t]:
            paint_error(img, emaps[t], thr)
        draw_curve(img, frame_score, t, thr)
        sel = fidx == t
        sc = None if row_scores is None else row_scores[sel]
        img = draw_tracks(img, tracks[sel], vel[sel], moving[sel], v["track_arrow_scale"],
                          v["track_arrow_min"], flags[sel], sc, person_thr, kind)
        if sc is not None:
            n_people_flagged |= set(tracks[sel][np.nan_to_num(sc) > person_thr, 1].astype(int).tolist())
        if emaps is not None and abnormal[t]:
            name = "thesis ASTT"
            label(img, f"ABNORMAL ({name} {frame_score[t]:.4f})", img.shape[1] // 2 - 100, 24, RED)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / f"{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     video_fps(params, dataset, vid), (w, h))
        writer.write(img)
    writer.release()

    with open(out_dir / f"{vid}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "model_score", "model_abnormal",
                    "rule_against_crowd", "rule_too_fast", "rule_loitering"])
        for k, row in enumerate(tracks):
            s = np.nan if row_scores is None else row_scores[k]
            w.writerow([int(row[0]), int(row[1]), "" if np.isnan(s) else f"{s:.4f}",
                        int(not np.isnan(s) and s > person_thr), int(flags[k] & 2 > 0),
                        int(flags[k] & 4 > 0), int(flags[k] & 1 > 0)])
    print(f"{vid}: {int(abnormal.sum())}/{len(abnormal)} frames above the {kind} threshold"
          + (f"; people flagged by the model: {sorted(n_people_flagged)}" if row_scores is not None else ""))
    print(f"rules: against crowd {sorted(set(tracks[flags & 2 > 0, 1].astype(int).tolist()))}, "
          f"too fast {sorted(set(tracks[flags & 4 > 0, 1].astype(int).tolist()))}, "
          f"loitering {sorted(set(tracks[flags & 1 > 0, 1].astype(int).tolist()))}")
    print(f"wrote {out_dir / (vid + '.mp4')}")


if __name__ == "__main__":
    main()
