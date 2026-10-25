"""Frames -> per-object attributes (AI-VAD feature_extraction.py), one cache file per video.

For every frame t (with frame t-1 for flow and moving blobs; frame 0 is paired with itself,
like AI-VAD's padding):
  objects   boxes + classes (+ keypoints, track ids for model 4)
  velocity  flow inside the box, resized to 224x224, AI-VAD's extract_velocity (8 bins of mean
            L1 magnitude per direction); magnitude / box height for class 0 (ShanghaiTech rule)
  mean_flow mean (dx, dy) / box height inside the box (model 4's crowd-relative attribute)
  pose      17 keypoints - box top-left, scaled to a 48x112 box, flattened (34-d), people only
  deep      CLIP embedding of the 224x224 crop (model 3 only)
A frame without objects is described by the WHOLE frame as one object (AI-VAD does the same).
Cache: data/features/<model>/<video>.npz
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.frames import list_frames
from src.models.aivad import aivad_code, networks

POSE_W, POSE_H = 48.0, 112.0     # pose box size; estimated from the authors' released pose features


class Extractor:
    def __init__(self, cfg: dict, device):
        m, w = cfg["model"], cfg["weights"]
        self.cfg, self.device = cfg, device
        self.velocity_fn = aivad_code.extract_velocity()
        self.get_foreground = aivad_code.video_dataset().get_foreground
        if m["objects"] == "maskrcnn":
            self.objects = networks.MaskRCNNObjects(w["maskrcnn"], device)
        else:
            self.objects = networks.YoloPoseObjects(w["yolo_pose"], device, m["yolo_conf"],
                                                    m["yolo_imgsz"], m.get("track", True))
        if m["flow"] == "flownet2":
            self.flow = networks.FlowNet2Flow(w.get("flownet2"), device)
        else:
            self.flow = networks.RaftFlow(m["raft_size"], m["raft_iters"], device, m.get("raft_small", False))
        self.pose = networks.FastPoseKeypoints(w.get("fastpose"), device) if m["pose"] == "fastpose" else None
        self.appearance = networks.ClipAppearance(device) if m["appearance"] == "clip" else None
        self.timing = {}

    def networks(self) -> dict:
        nets = {"objects": self.objects.net, "flow": self.flow.net}
        if self.pose:
            nets["pose"] = self.pose.net
        if self.appearance:
            nets["appearance"] = self.appearance.net
        return nets

    def _t(self, key, t0):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timing[key] = self.timing.get(key, 0.0) + time.time() - t0

    def frame(self, prev, cur) -> dict:
        t0 = time.time()
        boxes, classes, kps, ids, n_det = self.objects(prev, cur)
        self._t("objects", t0)
        t0 = time.time()
        flow = self.flow(prev, cur)
        self._t("flow", t0)
        H, W = cur.shape[:2]
        boxes, classes, kps, ids, n_det = _valid(boxes, classes, kps, ids, n_det, W, H)
        whole = len(boxes) == 0
        if whole:                                            # AI-VAD: the frame itself is the object
            boxes = np.array([[0, 0, W, H]], np.float32)
            classes, ids = np.array([-1]), np.array([-1])
            flow_crops = cv2.resize(flow, (224, 224))[None].transpose(0, 3, 1, 2)
        else:
            flow_crops = self.get_foreground(flow.transpose(2, 0, 1), boxes, 224)    # (n, 2, 224, 224)
        heights = np.where(classes == 0, boxes[:, 3] - boxes[:, 1], 1.0)
        vel, mean_flow = [], []
        for i in range(len(boxes)):
            f = np.ascontiguousarray(flow_crops[i].transpose(1, 2, 0))
            _, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
            mag = (np.sqrt(f[..., 0] ** 2) + np.sqrt(f[..., 1] ** 2)) / heights[i]
            vel.append(self.velocity_fn(f, mag, ang, orientations=8))
            mean_flow.append(f.reshape(-1, 2).mean(0) / max(boxes[i, 3] - boxes[i, 1], 1.0))
        out = {"boxes": boxes, "classes": classes, "ids": ids, "velocity": np.array(vel, np.float32),
               "mean_flow": np.array(mean_flow, np.float32), "whole": whole}

        t0 = time.time()                                     # pose of the detected people
        det = np.zeros(0, np.int64) if whole else np.flatnonzero(classes[:n_det] == 0)
        if self.pose is not None:                            # model 3: AlphaPose on person boxes
            kp = self.pose(cur, boxes[det])
        else:                                                # model 4: keypoints from YOLO
            kp = kps[det] if kps is not None and len(det) else np.zeros((0, 17, 3), np.float32)
        pose_boxes = boxes[det]
        self._t("pose", t0)
        out["pose"] = normalise_pose(kp, pose_boxes)
        out["pose_rows"] = det                               # pose i belongs to object row det[i]

        if self.appearance is not None:
            t0 = time.time()
            crops = (cv2.resize(cur, (224, 224))[None] if whole else
                     self.get_foreground(cur.transpose(2, 0, 1), boxes, 224).transpose(0, 2, 3, 1))
            out["deep"] = self.appearance(np.ascontiguousarray(crops)).astype(np.float16)
            self._t("appearance", t0)
        return out

    def video(self, frame_dir: Path, out_file: Path | None = None, max_frames=None) -> dict:
        """All frames of one video -> arrays (rows = objects), saved to out_file if given."""
        if out_file is not None and out_file.exists():
            return dict(np.load(out_file))
        if hasattr(self.objects, "reset"):
            self.objects.reset()
        frames = list_frames(frame_dir)[:max_frames]
        rows = {k: [] for k in ("frame", "boxes", "classes", "ids", "velocity", "mean_flow", "whole", "deep")}
        prow = {"pose_frame": [], "pose": [], "pose_object": []}
        prev = None
        t_start = time.time()
        n_obj = 0
        for t, p in enumerate(frames):
            cur = cv2.imread(str(p))
            f = self.frame(cur if prev is None else prev, cur)
            n = len(f["boxes"])
            rows["frame"].append(np.full(n, t, np.int32))
            for k in ("boxes", "classes", "ids", "velocity", "mean_flow"):
                rows[k].append(f[k])
            rows["whole"].append(np.full(n, f["whole"]))
            if "deep" in f:
                rows["deep"].append(f["deep"])
            prow["pose_frame"].append(np.full(len(f["pose"]), t, np.int32))
            prow["pose"].append(f["pose"])
            prow["pose_object"].append(n_obj + f["pose_rows"])
            n_obj += n
            prev = cur
        cat = lambda l, shape, dt: np.concatenate(l, 0).astype(dt) if l else np.zeros(shape, dt)
        out = {"n_frames": np.array(len(frames)), "frame": cat(rows["frame"], (0,), np.int32),
               "boxes": cat(rows["boxes"], (0, 4), np.float32), "classes": cat(rows["classes"], (0,), np.int64),
               "ids": cat(rows["ids"], (0,), np.int64), "velocity": cat(rows["velocity"], (0, 8), np.float32),
               "mean_flow": cat(rows["mean_flow"], (0, 2), np.float32), "whole": cat(rows["whole"], (0,), bool),
               "pose_frame": cat(prow["pose_frame"], (0,), np.int32), "pose": cat(prow["pose"], (0, 34), np.float32),
               "pose_object": cat(prow["pose_object"], (0,), np.int64),
               "seconds": np.array(time.time() - t_start)}
        if rows["deep"]:
            out["deep"] = cat(rows["deep"], (0, 512), np.float16)
        if out_file is not None:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_file.with_name(out_file.stem + ".tmp.npz")
            np.savez_compressed(tmp, **out)
            tmp.replace(out_file)
        return out


def _valid(boxes, classes, kps, ids, n_det, W, H):
    """Clip boxes to the frame; drop boxes whose crop (AI-VAD crops with ceil()) would be empty
    (the original code would crash on them)."""
    if len(boxes) == 0:
        return boxes, classes, kps, ids, n_det
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    ok = (np.ceil(boxes[:, 2]) - np.ceil(boxes[:, 0]) >= 1) & (np.ceil(boxes[:, 3]) - np.ceil(boxes[:, 1]) >= 1)
    if ok.all():
        return boxes, classes, kps, ids, n_det
    keep_kps = kps[ok[:len(kps)]] if kps is not None else None
    return boxes[ok], classes[ok], keep_kps, ids[ok], int(ok[:n_det].sum())


def normalise_pose(kps: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """AI-VAD pose representation: keypoints - box top-left, scaled to a fixed box, flattened."""
    if len(kps) == 0:
        return np.zeros((0, 34), np.float32)
    xy = kps[:, :, :2] - boxes[:, None, :2]
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)[:, None]
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)[:, None]
    xy[:, :, 0] *= POSE_W / w
    xy[:, :, 1] *= POSE_H / h
    return xy.reshape(len(kps), 34).astype(np.float32)
