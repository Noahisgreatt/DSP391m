"""Models 3 and 4: AI-VAD (Reiss & Hoshen, TMLR 2025) and our lighter AI-VAD Lite.

AI-VAD does not train a network. It describes every object in every frame by interpretable
ATTRIBUTES, and measures how unusual each one is compared with all objects of the normal
training videos:
    objects     detector boxes + "moving blobs" (frame-difference contours)
    velocity    optical flow inside the box -> 8 direction bins (mean flow size)      -> GMM
    pose        17 body keypoints relative to the box (34 numbers)                     -> 1-NN
    appearance  CLIP embedding of the box (512 numbers)                                 -> 1-NN
Frame score = sum over attributes of the most unusual object (calibrated on the training
videos), smoothed over time. Each object's score says WHY it is unusual.

                   model 3 aivad (331.0M)                 model 4 aivad_lite (98.3M)
    objects        Faster R-CNN, all 80 COCO classes      Keypoint R-CNN: people + keypoints
    pose           AlphaPose FastPose on each person      from the same Keypoint R-CNN pass
    flow           FlowNet2 (5 networks) at 1024x640      FlowNetC (its 1st network) at 512x320
    appearance     CLIP ViT-B/16                          not used (AI-VAD doesn't score it on ShanghaiTech)
    extra          -                                      motion relative to the crowd, one model of
                                                          "normal" per camera, smoothing along tracks
All networks: src/models/nets.py (our code + pretrained weights). AI-VAD's own code
(third_party/AIVAD) is used unchanged for the box filters, moving blobs, crops and velocity.

    python -m src.train    --config configs/aivad.yaml       ("training" = attributes of the
    python -m src.evaluate --config configs/aivad.yaml        normal videos + density fit)
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

from src.models import nets
from src.utils import config_hash, frames_dir, git_commit, list_frames, out_dir

AIVAD = nets.ROOT / "third_party" / "AIVAD"
POSE_W, POSE_H = 48.0, 112.0          # AI-VAD's pose box size (estimated from the authors' pose features)


# ====================================================================== AI-VAD's own code
def _aivad(module):
    """Import a file of third_party/AIVAD (its scripts import each other as top-level modules).
    `object_detector` needs detectron2, which we don't use: our detector is passed in instead."""
    if not AIVAD.exists():
        raise SystemExit("third_party/AIVAD missing: run  python third_party/fetch.py")
    for p in (AIVAD, AIVAD / "pre_processing"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if "object_detector" not in sys.modules:
        sys.modules["object_detector"] = types.SimpleNamespace(Predictor=None)
    return importlib.import_module(module)


def velocity_histogram(flow, magnitude, orientation, orientations=8):
    """= AI-VAD feature_extraction.extract_velocity. Theirs fails on numpy >= 2 (it assigns a
    1x1 array to one element); its uniform_filter over the whole crop read at the centre IS the
    mean over the crop, so: per direction bin, the mean of the magnitudes in that bin."""
    deg = orientation * (180 / np.pi)
    return np.array([np.where((deg >= 360 / orientations * i) & (deg < 360 / orientations * (i + 1)),
                              magnitude, 0).mean() for i in range(orientations)])


def normalise_pose(kps, boxes):
    """AI-VAD pose: keypoints - box top-left, scaled to a fixed POSE_W x POSE_H box, flattened."""
    if len(kps) == 0:
        return np.zeros((0, 34), np.float32)
    xy = kps[:, :, :2] - boxes[:, None, :2]
    xy[:, :, 0] *= POSE_W / np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)[:, None]
    xy[:, :, 1] *= POSE_H / np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)[:, None]
    return xy.reshape(len(kps), 34).astype(np.float32)


# ====================================================================== 1. attributes of every object
class Extractor:
    def __init__(self, cfg, device, weights=True):
        m = cfg["model"]
        self.cfg, self.device, self.timing = cfg, device, {}
        self.bb = _aivad("bboxes")                        # AI-VAD box filters + moving blobs
        self.c = self.bb.DATASET_CFGS["shanghaitech"]
        self.get_foreground = _aivad("video_dataset").get_foreground
        self.lite = m["objects"] == "keypointrcnn"
        if self.lite:
            self.detector = nets.RCNN(2, 17, score_thresh=m["detector_conf"])
            if weights:
                self.detector.load_pretrained("keypointrcnn")
            from src.behavior import IoUTracker
            self.tracker = IoUTracker()
        else:
            self.detector = nets.RCNN(91, score_thresh=self.c["confidence_threshold"])   # AI-VAD: 0.8
            if weights:
                self.detector.load_pretrained("maskrcnn")
        self.detector = self.detector.to(device).eval()
        self.flow_size = tuple(m["flow_size"])
        self.flow = (nets.FlowNetC(weights) if m["flow"] == "flownetc" else nets.flownet2(weights)).to(device).eval()
        self.pose = nets.FastPose().to(device).eval() if m["pose"] == "fastpose" else None
        if self.pose is not None and weights:
            self.pose.load_pretrained()
        self.clip = None
        if m["appearance"] == "clip":
            self.clip = nets.CLIPVisual()
            if weights:
                self.clip.load_pretrained()
            self.clip = self.clip.to(device).eval()
            if device.type == "cuda":
                self.clip = self.clip.half()               # as clip.load() does on a GPU

    def networks(self) -> dict:
        d = {"objects": self.detector, "flow": self.flow}
        if self.pose is not None:
            d["pose"] = self.pose
        if self.clip is not None:
            d["appearance"] = self.clip
        return d

    def _tick(self, key, t0):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timing[key] = self.timing.get(key, 0.0) + time.time() - t0

    # ---- objects: detector -> AI-VAD's size + overlap filters -> + moving blobs (class 0)
    def objects(self, prev, cur):
        det = self.detector([nets.to_rgb_tensor(cur, self.device)])[0]
        kps = None
        if self.lite:                                      # people only
            boxes, cls = det["boxes"].cpu().numpy(), np.zeros(len(det["boxes"]), np.int64)
            kps, scores = det["keypoints"].cpu().numpy(), det["scores"].cpu().numpy()
            area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            ok = area >= self.c["min_area"]
            boxes, cls, kps, scores, area = boxes[ok], cls[ok], kps[ok], scores[ok], area[ok]
            kept, idx = self.bb.delete_overlapped_bboxes(boxes, area, self.c["cover_threshold"], np.arange(len(boxes)))
            boxes, cls, kps, scores = kept, cls[idx], kps[idx], scores[idx]
            ids = self.tracker.update(boxes, scores)
        else:                                              # AI-VAD bboxes.extract, our detector
            labels = det["labels"].cpu().numpy()
            classes = torch.tensor([nets.COCO_TO_CONTIGUOUS[int(c)] for c in labels], dtype=torch.int64)
            boxes, area, cls = self.bb.get_objects_bboxes(cur, lambda img: (det["boxes"].cpu(), classes),
                                                          self.c["min_area"], "shanghaitech")
            boxes, cls = self.bb.delete_overlapped_bboxes(boxes, area, self.c["cover_threshold"], cls)
            ids = -np.ones(len(boxes), np.int64)
        n_det = len(boxes)
        fg = self.bb.get_foreground_bboxes(np.stack([prev, cur]), boxes, self.c["contour_min_area"],
                                           self.c["binary_threshold"], self.c["gauss_mask_size"])
        if len(fg):
            boxes = np.concatenate([boxes, fg], 0)
            cls = np.concatenate([cls, np.zeros(len(fg), np.int64)])
            ids = np.concatenate([ids, -np.ones(len(fg), np.int64)])
        return boxes.astype(np.float32), cls.astype(np.int64), kps, ids, n_det

    # ---- optical flow (AI-VAD pre_processing/flows.py): BGR 0-255 at flow_size, resized back
    @torch.no_grad()
    def optical_flow(self, prev, cur):
        h, w = cur.shape[:2]
        ims = np.array([cv2.resize(prev, self.flow_size), cv2.resize(cur, self.flow_size)]).astype(np.float32)
        x = torch.from_numpy(ims).unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous().to(self.device)
        return cv2.resize(self.flow(x)[0].float().cpu().numpy().transpose(1, 2, 0), (w, h))

    def frame(self, prev, cur) -> dict:
        t0 = time.time()
        boxes, cls, kps, ids, n_det = self.objects(prev, cur)
        self._tick("objects", t0)
        t0 = time.time()
        flow = self.optical_flow(prev, cur)
        self._tick("flow", t0)
        H, W = cur.shape[:2]
        boxes[:, [0, 2]], boxes[:, [1, 3]] = boxes[:, [0, 2]].clip(0, W), boxes[:, [1, 3]].clip(0, H)
        ok = (np.ceil(boxes[:, 2]) - np.ceil(boxes[:, 0]) >= 1) & (np.ceil(boxes[:, 3]) - np.ceil(boxes[:, 1]) >= 1)
        if not ok.all():                                   # empty crops would crash AI-VAD's code
            n_det = int(ok[:n_det].sum())
            kps = kps[ok[:len(kps)]] if kps is not None else None
            boxes, cls, ids = boxes[ok], cls[ok], ids[ok]
        whole = len(boxes) == 0                            # AI-VAD: then the frame itself is the object
        if whole:
            boxes, cls, ids = np.array([[0, 0, W, H]], np.float32), np.array([-1]), np.array([-1])
            fcrops = cv2.resize(flow, (224, 224))[None].transpose(0, 3, 1, 2)
        else:
            fcrops = self.get_foreground(flow.transpose(2, 0, 1), boxes, 224)
        heights = np.where(cls == 0, boxes[:, 3] - boxes[:, 1], 1.0)        # ShanghaiTech rule
        vel, mean_flow = [], []
        for i in range(len(boxes)):
            f = np.ascontiguousarray(fcrops[i].transpose(1, 2, 0))
            _, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
            vel.append(velocity_histogram(f, (np.abs(f[..., 0]) + np.abs(f[..., 1])) / heights[i], ang))
            mean_flow.append(f.reshape(-1, 2).mean(0) / max(boxes[i, 3] - boxes[i, 1], 1.0))
        out = {"boxes": boxes, "classes": cls, "ids": ids, "velocity": np.array(vel, np.float32),
               "mean_flow": np.array(mean_flow, np.float32), "whole": whole}
        t0 = time.time()
        people = np.zeros(0, np.int64) if whole else np.flatnonzero(cls[:n_det] == 0)
        if self.pose is not None:
            kp = self.pose.keypoints(cur, boxes[people])
        else:
            kp = kps[people] if kps is not None and len(people) else np.zeros((0, 17, 3), np.float32)
        out["pose"], out["pose_rows"] = normalise_pose(kp, boxes[people]), people
        self._tick("pose", t0)
        if self.clip is not None:
            t0 = time.time()
            crops = (cv2.resize(cur, (224, 224))[None] if whole else
                     self.get_foreground(cur.transpose(2, 0, 1), boxes, 224).transpose(0, 2, 3, 1))
            x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float().div(255)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
            with torch.no_grad():                          # AI-VAD normalises the BGR crop this way
                out["deep"] = self.clip(((x - mean) / std).to(self.device)).float().cpu().numpy().astype(np.float16)
            self._tick("appearance", t0)
        return out

    def video(self, frame_dir, cache=None, max_frames=None) -> dict:
        """All frames of one video -> object rows (frame index, box, attributes...), cached."""
        if cache is not None and Path(cache).exists():
            return dict(np.load(cache))
        if self.lite:
            self.tracker.reset()
        rows = {k: [] for k in ("frame", "boxes", "classes", "ids", "velocity", "mean_flow", "whole", "deep")}
        prow = {"pose_frame": [], "pose": [], "pose_object": []}
        prev, n_obj, t0 = None, 0, time.time()
        frames = list_frames(frame_dir)[:max_frames]
        for t, p in enumerate(frames):
            cur = cv2.imread(str(p))
            f = self.frame(cur if prev is None else prev, cur)
            n = len(f["boxes"])
            rows["frame"].append(np.full(n, t, np.int32))
            rows["whole"].append(np.full(n, f["whole"]))
            for k in ("boxes", "classes", "ids", "velocity", "mean_flow") + (("deep",) if "deep" in f else ()):
                rows[k].append(f[k])
            prow["pose_frame"].append(np.full(len(f["pose"]), t, np.int32))
            prow["pose"].append(f["pose"])
            prow["pose_object"].append(n_obj + f["pose_rows"])
            n_obj, prev = n_obj + n, cur
        cat = lambda l, shape, dt: np.concatenate(l, 0).astype(dt) if l else np.zeros(shape, dt)
        out = {"n_frames": np.array(len(frames)), "frame": cat(rows["frame"], (0,), np.int32),
               "boxes": cat(rows["boxes"], (0, 4), np.float32), "classes": cat(rows["classes"], (0,), np.int64),
               "ids": cat(rows["ids"], (0,), np.int64), "velocity": cat(rows["velocity"], (0, 8), np.float32),
               "mean_flow": cat(rows["mean_flow"], (0, 2), np.float32), "whole": cat(rows["whole"], (0,), bool),
               "pose_frame": cat(prow["pose_frame"], (0,), np.int32), "pose": cat(prow["pose"], (0, 34), np.float32),
               "pose_object": cat(prow["pose_object"], (0,), np.int64), "seconds": np.array(time.time() - t0)}
        if rows["deep"]:
            out["deep"] = cat(rows["deep"], (0, 512), np.float16)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(cache).with_name(Path(cache).stem + ".tmp.npz")
            np.savez_compressed(tmp, **out)
            tmp.replace(cache)
        return out


# ====================================================================== 2. how unusual is each object
@torch.no_grad()
def nn_distance(query, bank, q_group=None, b_group=None, device="cpu", q_chunk=2048, b_chunk=65536):
    """Squared distance to the nearest bank row (exact; = AI-VAD's faiss IndexFlatL2, k=1).
    With groups, rows of the query's own group are skipped (leave-one-video-out)."""
    if len(query) == 0:
        return np.zeros(0, np.float32)
    b = torch.as_tensor(bank, dtype=torch.float32, device=device)
    b2 = (b * b).sum(1)
    bg = None if b_group is None else torch.as_tensor(b_group, device=device)
    out = np.empty(len(query), np.float32)
    for s in range(0, len(query), q_chunk):
        q = torch.as_tensor(query[s:s + q_chunk], dtype=torch.float32, device=device)
        q2, best = (q * q).sum(1, keepdim=True), torch.full((len(q),), float("inf"), device=device)
        qg = None if bg is None else torch.as_tensor(q_group[s:s + q_chunk], device=device)
        for t in range(0, len(b), b_chunk):
            d = q2 + b2[None, t:t + b_chunk] - 2 * q @ b[t:t + b_chunk].T
            if qg is not None:
                d[qg[:, None] == bg[None, t:t + b_chunk]] = float("inf")
            best = torch.minimum(best, d.min(1).values)
        out[s:s + q_chunk] = best.clamp_min(0).cpu().numpy()
    return out


def relative_velocity(feat):
    """Model 4: each object's motion minus the median motion of the OTHER objects in the frame
    (body heights per frame). Walking against / much faster than the crowd -> large."""
    mf, fr, out = feat["mean_flow"], feat["frame"], feat["mean_flow"].copy()
    for t in np.unique(fr):
        rows = np.flatnonzero(fr == t)
        for r in rows if len(rows) > 1 else []:
            out[r] = mf[r] - np.median(mf[rows[rows != r]], 0)
    return out


class Density:
    """AI-VAD evaluate.py + score_calibration.py.
    fit:  velocity -> GMM (5 components), unusualness = -log p; pose / appearance -> bank of
          training samples, unusualness = squared distance to the nearest one. Calibration:
          (s - min) / (99.9th percentile - min) of the training scores (pose / appearance:
          leave-one-video-out; appearance uses the max, as in AI-VAD's code).
    score: per frame and attribute the most unusual object, calibrated; sum of `attributes`;
          Gaussian smoothing over time (sigma 7 on ShanghaiTech)."""

    def __init__(self, cfg):
        s = cfg["scoring"]
        self.attributes, self.extra = list(s["attributes"]), [a for a in s.get("also_calibrate", []) if a not in s["attributes"]]
        self.k, self.pct, self.sigma = s["gmm_components"], s["calibration_percentile"], s["smoothing_sigma"]
        self.per_scene, self.track_smooth = s.get("per_scene", False), s.get("track_smoothing", 0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state = {}

    def _fit(self, feats):
        vids, st = sorted(feats), {}
        for att in ("velocity", "relative_velocity"):
            if att in self.attributes + self.extra:
                X = np.concatenate([feats[v]["velocity"] if att == "velocity" else relative_velocity(feats[v])
                                    for v in vids]).astype(np.float64)
                g = GaussianMixture(n_components=self.k, random_state=0).fit(X)
                tr = -g.score_samples(X)
                st[att] = {"gmm": {k: torch.as_tensor(getattr(g, k)) for k in ("weights_", "means_", "covariances_", "precisions_cholesky_")},
                           "lo": float(tr.min()), "hi": float(np.percentile(tr, self.pct))}
        for att, key in (("pose", "pose"), ("appearance", "deep")):
            if att not in self.attributes + self.extra:
                continue
            have = [v for v in vids if key in feats[v] and len(feats[v][key])]
            if len(have) < 2:
                print(f"warning: {att} found in fewer than 2 training videos; it scores 0")
                st[att] = None
                continue
            bank = np.concatenate([feats[v][key] for v in have]).astype(np.float32)
            grp = np.concatenate([np.full(len(feats[v][key]), i) for i, v in enumerate(have)])
            d = nn_distance(bank, bank, grp, grp, self.device)
            d = d[np.isfinite(d)]
            st[att] = {"bank": torch.as_tensor(bank, dtype=torch.float16 if att == "appearance" else torch.float32),
                       "lo": float(d.min()), "hi": float(d.max() if att == "appearance" else np.percentile(d, self.pct))}
        return st

    def fit(self, feats):
        self.state = {"global": self._fit(feats)}
        if self.per_scene:                                      # one model of "normal" per camera
            for sc in sorted({v[:2] for v in feats}):
                sub = {v: f for v, f in feats.items() if v[:2] == sc}
                if len(sub) >= 2:
                    self.state[sc] = self._fit(sub)
        return {g: {a: None if s is None else {"lo": s["lo"], "hi": s["hi"]} for a, s in st.items()} for g, st in self.state.items()}

    def _objects(self, st, feat):
        out = {}
        for att, s in st.items():
            n = len(feat["pose"]) if att == "pose" else len(feat["frame"])
            if s is None:
                out[att] = np.zeros(n)
                continue
            if att in ("velocity", "relative_velocity"):
                g = GaussianMixture(n_components=len(s["gmm"]["weights_"]), covariance_type="full")
                for k, v in s["gmm"].items():
                    setattr(g, k, v.numpy())
                x = feat["velocity"] if att == "velocity" else relative_velocity(feat)
                raw = -g.score_samples(x.astype(np.float64))
            else:
                q = feat.get("pose" if att == "pose" else "deep", np.zeros((0, 1)))
                raw = nn_distance(np.asarray(q, np.float32), s["bank"].float().numpy(), device=self.device)
            out[att] = (raw - s["lo"]) / max(s["hi"] - s["lo"], 1e-12)
        return out

    def _smooth_tracks(self, ids, score, frame):
        out = score.copy()
        for tid in np.unique(ids[ids >= 0]):
            r = np.flatnonzero(ids == tid)
            r = r[np.argsort(frame[r])]
            c, k = np.cumsum(np.insert(score[r], 0, 0)), np.arange(1, len(r) + 1)
            lo = np.maximum(0, k - self.track_smooth)
            out[r] = (c[k] - c[lo]) / (k - lo)
        return out

    def score(self, video, feat) -> dict:
        st = self.state.get(video[:2], self.state["global"]) if self.per_scene else self.state["global"]
        n, obj, per_att = int(feat["n_frames"]), self._objects(st, feat), {}
        for att, s in obj.items():
            fr, rows = (feat["pose_frame"], feat["pose_object"]) if att == "pose" else (feat["frame"], np.arange(len(feat["frame"])))
            if self.track_smooth and len(s):
                ids = np.where(rows >= 0, feat["ids"][np.maximum(rows, 0)], -1)
                obj[att] = s = self._smooth_tracks(ids, s, fr)
            f = np.full(n, -np.inf)
            np.maximum.at(f, fr, s)
            per_att[att] = np.where(np.isfinite(f), f, 0.0)          # no pose in a frame -> 0
        total = sum(per_att[a] for a in self.attributes)
        return {"score": (gaussian_filter1d(total, self.sigma) if self.sigma else total).astype(np.float32),
                "per_attribute": per_att, "objects": obj}


# ====================================================================== 3. "training" and scoring
def cache_dir(cfg, split, smoke=False):
    return Path(cfg["paths"]["features"]) / (cfg["model"]["name"] + ("_smoke" if smoke else "")) / split


def count_parameters(ex) -> dict:
    out = {f"{k} ({type(v).__name__})": nets.n_params(v) for k, v in ex.networks().items()}
    out["Total"] = sum(out.values())
    return out


def fit(cfg, smoke=False, fresh=False):
    """Attributes of the normal training videos -> density estimators -> outputs/training/<model>/."""
    from src.data import load_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = out_dir(cfg, "training", smoke)
    if fresh and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    vids = load_split(cfg)["train"][: 2 if smoke else None]
    t0, ex = time.time(), Extractor(cfg, device)
    params = count_parameters(ex)
    print(f"{cfg['model']['name']}: {params['Total']:,} parameters in its networks, on {device}")
    feats, log = {}, ["step,video,frames,objects,fps,minutes"]
    for k, v in enumerate(vids, 1):                         # cached per video: a stopped run continues
        f = feats[v] = ex.video(frames_dir(cfg, "shanghaitech") / v, cache_dir(cfg, "train", smoke) / f"{v}.npz",
                                30 if smoke else None)
        fps = float(f["n_frames"]) / max(float(f["seconds"]), 1e-9)
        log.append(f"{k},{v},{int(f['n_frames'])},{len(f['frame'])},{fps:.2f},{(time.time() - t0) / 60:.2f}")
        print(f"[{k}/{len(vids)}] {v}: {int(f['n_frames'])} frames, {len(f['frame'])} objects, {len(f['pose'])} poses ({fps:.1f} fps)")
    (out / "log.csv").write_text("\n".join(log) + "\n")
    print("fitting the density estimators (leave-one-video-out calibration) ...")
    density = Density(cfg)
    calib = density.fit(feats)
    torch.save({"state": density.state, "config": cfg}, out / "model.pth")
    (out / "run.json").write_text(json.dumps({
        "model": cfg["model"]["name"], "config_hash": config_hash({k: cfg[k] for k in ("model", "scoring")}),
        "git_commit": git_commit(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device), "parameters": params, "train_videos": len(vids), "config": cfg},
        indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps({
        "train_videos": len(feats), "train_frames": int(sum(int(f["n_frames"]) for f in feats.values())),
        "train_objects": int(sum(len(f["frame"]) for f in feats.values())),
        "train_poses": int(sum(len(f["pose"]) for f in feats.values())), "calibration": calib,
        "minutes": round((time.time() - t0) / 60, 1), "parameters_total": params.get("Total")}, indent=2))
    print(f"done: {out}/model.pth")


class Scorer:
    """Score one video with a fitted model 3 / 4 (used by evaluate, demo, API)."""
    kind = "score"

    def __init__(self, cfg, checkpoint, device, smoke=False, extractor=None):
        self.cfg, self.device, self.smoke, self.extractor = cfg, device, smoke, extractor
        self.density = Density(cfg)
        self.density.state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state"]

    def features(self, video_dir, vid, cache=True, max_frames=None):
        from src.data import load_split
        test = load_split(self.cfg)["test"] if (Path(self.cfg["paths"]["metadata"]) / "split.json").exists() else []
        if self.extractor is None:
            self.extractor = Extractor(self.cfg, self.device)
        cache_file = cache_dir(self.cfg, "test", self.smoke) / f"{vid}.npz" if cache and vid in test else None
        return self.extractor.video(video_dir, cache_file, 30 if self.smoke else max_frames)

    def __call__(self, video_dir, vid, details=False, cache=True, max_frames=None, **_):
        f = self.features(Path(video_dir), vid, cache, max_frames)
        r = self.density.score(vid, f)
        return (r["score"], {**r, "features": f}) if details else r["score"]
