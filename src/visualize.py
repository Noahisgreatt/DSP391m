"""Render videos with flow arrows, tracks, rule flags and the anomaly score drawn on them.

    python -m src.visualize --dataset shanghaitech --model vit_flow --run latest --videos test__01_0014
    python -m src.visualize --dataset manual --model vit_flow --run latest --all
    python -m src.visualize --dataset manual --all          # no model: rules only (outputs/rules/tracking/)

Output: outputs/<model_name>/<run_id>/<video_id>.mp4
  * thin arrows on a grid  = Farneback flow (red = against the scene's dominant direction)
  * boxes                  = tracked people (green normal, orange loitering, red counter-flow)
  * strip at the bottom    = anomaly score over time (red line = GT anomaly, white cursor = now)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.evaluate import gt_for, resolve_run
from src.features.direction import counterflow_scores
from src.features.tracking import apply_rules
from src.utils import list_frame_dirs, list_frames, load_params, load_yaml, resolve, scene_of, split_of

GREEN, ORANGE, RED, WHITE, YELLOW = (0, 200, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255), (0, 255, 255)


def draw_flow(img, flow, cos_map, compute_size, step=24):
    """flow: (h, w, 2) in compute_size px units. cos_map: (gr, gc) or None."""
    H, W = img.shape[:2]
    fh, fw = flow.shape[:2]
    sx = W / compute_size[0]  # flow units -> output px
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            dx, dy = flow[min(int(y * fh / H), fh - 1), min(int(x * fw / W), fw - 1)] * sx * 3
            if dx * dx + dy * dy < 4:
                continue
            color = YELLOW
            if cos_map is not None:
                gr, gc = cos_map.shape
                if cos_map[min(y * gr // H, gr - 1), min(x * gc // W, gc - 1)] < -0.5:
                    color = RED
            cv2.arrowedLine(img, (x, y), (int(x + dx), int(y + dy)), color, 1, tipLength=0.35)


def draw_strip(img, score, gt, t, height=40):
    H, W = img.shape[:2]
    strip = np.zeros((height, W, 3), np.uint8)
    n = len(score)
    xs = (np.arange(n) * (W - 1) / max(n - 1, 1)).astype(int)
    if gt is not None:
        for i in np.flatnonzero(gt):
            cv2.line(strip, (xs[i], 0), (xs[i], 4), RED, 1)
    pts = np.stack([xs, (height - 2 - score * (height - 8)).astype(int)], 1).reshape(-1, 1, 2)
    cv2.polylines(strip, [pts.astype(np.int32)], False, ORANGE, 1)
    cv2.line(strip, (xs[t], 0), (xs[t], height - 1), WHITE, 1)
    return np.vstack([img, strip])


def render(dataset, vid, params, out_path, score=None, fcfg=None, tcfg=None):
    frames = list_frames(resolve(params["paths"]["frames"]) / dataset / vid)
    n = len(frames)
    first = cv2.imread(str(frames[0]))
    H, W = first.shape[:2]
    fps = float(params["datasets"][dataset].get("fps") or 25)

    feat_root = resolve(params["paths"]["features"])
    flow_npz = feat_root / "farneback" / dataset / f"{vid}.npz"
    flow = cos = None
    if flow_npz.exists():
        d = np.load(flow_npz)
        flow = {int(f): k for k, f in enumerate(d["frame_idx"])}
        flow_arr = d["flow"]
        prior_p = feat_root / "farneback" / dataset / "_priors" / f"{scene_of(vid)}.npz"
        if prior_p.exists() and fcfg:
            _, cos = counterflow_scores(d["hist"], dict(np.load(prior_p)), fcfg["direction"])

    track_npz = feat_root / "tracking" / dataset / f"{vid}.npz"
    tracks = rules = None
    if track_npz.exists() and tcfg:
        tracks = dict(np.load(track_npz))
        rules = apply_rules(tracks, n, fps, tcfg["rules"])
        by_frame = {}
        for k, f in enumerate(tracks["frame"]):
            by_frame.setdefault(int(f), []).append(k)
        if score is None:
            score = rules["frame_score"]

    gt = gt_for(params, dataset, vid, n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + 40))
    for t, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if flow is not None and t in flow:
            k = flow[t]
            draw_flow(img, flow_arr[k].astype(np.float32), None if cos is None else cos[k],
                      fcfg["compute_size"] if fcfg else (W, H))
        if tracks is not None:
            for k in by_frame.get(t, []):
                x, y, w, h = (tracks[c][k] for c in ("cx", "cy", "w", "h"))
                lo, co = rules["det_loiter"][k], rules["det_counter"][k]
                color, tag = (RED, "nguoc dong") if co >= 0.5 else (ORANGE, "lang vang") if lo >= 0.5 else (GREEN, "")
                p1, p2 = (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2))
                cv2.rectangle(img, p1, p2, color, 2 if tag else 1)
                cv2.putText(img, f"{tracks['id'][k]} {tag}", (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                vx, vy = rules["velocity"][k] * h * 10
                cv2.arrowedLine(img, (int(x), int(y)), (int(x + vx), int(y + vy)), color, 2, tipLength=0.3)
        s = float(score[t]) if score is not None else 0.0
        cv2.putText(img, f"frame {t}  score {s:.2f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        if gt is not None:
            cv2.putText(img, "GT: BAT THUONG" if gt[t] else "GT: binh thuong", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED if gt[t] else GREEN, 2)
        writer.write(draw_strip(img, score if score is not None else np.zeros(n), gt, t))
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run", default="latest")
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="All test videos of the dataset.")
    ap.add_argument("--flow-config", default="configs/features/farneback.yaml")
    ap.add_argument("--tracking-config", default="configs/features/tracking.yaml")
    args = ap.parse_args()

    params = load_params()
    fcfg = load_yaml(resolve(args.flow_config))
    tcfg = load_yaml(resolve(args.tracking_config))
    vids = args.videos or [d.name for d in list_frame_dirs(resolve(params["paths"]["frames"]) / args.dataset)
                           if not d.name.startswith("_") and (not args.all or split_of(d.name, params, args.dataset) == "test")]

    if args.model:
        run_dir = resolve_run(params, args.model, args.run)
        out_dir = resolve(params["paths"]["outputs"]) / args.model / run_dir.name
    else:
        run_dir, out_dir = None, resolve(params["paths"]["outputs"]) / "rules" / "tracking"

    for vid in tqdm(vids, desc="render"):
        score = None
        if run_dir is not None:
            sp = run_dir / "scores" / args.dataset / f"{vid}.npz"
            if not sp.exists():
                print(f"skip {vid}: run src.evaluate first ({sp} missing)")
                continue
            score = np.load(sp)["score"]
        render(args.dataset, vid, params, out_dir / f"{vid}.mp4", score, fcfg, tcfg)
    print("->", out_dir)


if __name__ == "__main__":
    main()
