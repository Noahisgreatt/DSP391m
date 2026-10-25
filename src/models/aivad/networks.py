"""The pretrained networks of models 3 and 4. One class per job; all work on BGR uint8 frames.

Model 3 (AI-VAD, original networks)         Model 4 (AI-VAD Lite)
  MaskRCNNObjects  Mask R-CNN R50-FPN          YoloPoseObjects  YOLO11x-pose: boxes + keypoints
  FlowNet2Flow     FlowNet2                    RaftFlow         RAFT (torchvision)
  FastPoseKeypoints AlphaPose FastPose R50     (keypoints come from YOLO11x-pose)
  ClipAppearance   CLIP ViT-B/16               (not used: AI-VAD's ShanghaiTech score skips it)
Both add AI-VAD's "moving blobs" (frame-difference contours) as extra objects.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.models.aivad import aivad_code

ALPHAPOSE = Path(__file__).resolve().parents[3] / "third_party" / "AlphaPose"


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ============================================================== objects
class MaskRCNNObjects:
    """AI-VAD pre_processing/bboxes.py: Mask R-CNN (detectron2, COCO, R50-FPN 3x) with its
    ShanghaiTech settings, overlap filter, and moving blobs (labelled class 0, like AI-VAD)."""

    def __init__(self, weights: str, device):
        self.b = aivad_code.bbox_functions(need_detector=True)
        self.c = self.b.DATASET_CFGS["shanghaitech"]
        # = AI-VAD object_detector.Predictor, with our weights path instead of a relative one
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.c["confidence_threshold"]
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = str(device)
        self.predictor = DefaultPredictor(cfg)
        self.net = self.predictor.model

    def predict(self, img):
        with torch.no_grad():
            inst = self.predictor(img)["instances"]
        return inst.pred_boxes.tensor, inst.pred_classes

    def __call__(self, prev, cur):
        """-> boxes (n, 4) x1 y1 x2 y2, classes (n,), keypoints None, track ids (n,) -1, n_det
        (the first n_det rows are detections, the rest moving blobs)."""
        boxes, areas, classes = self.b.get_objects_bboxes(cur, self.predict, self.c["min_area"], "shanghaitech")
        boxes, classes = self.b.delete_overlapped_bboxes(boxes, areas, self.c["cover_threshold"], classes)
        n_det = len(boxes)
        fg = self.b.get_foreground_bboxes(np.stack([prev, cur]), boxes, self.c["contour_min_area"],
                                          self.c["binary_threshold"], self.c["gauss_mask_size"])
        if len(fg):
            boxes = np.concatenate([boxes, fg], 0)
            classes = np.concatenate([classes, np.zeros(len(fg))], 0)
        return boxes.astype(np.float32), classes.astype(np.int64), None, -np.ones(len(boxes), np.int64), n_det


class YoloPoseObjects:
    """Model 4: ONE network finds people AND their 17 body keypoints (replaces Mask R-CNN +
    AlphaPose), with ByteTrack ids. Same overlap filter and moving blobs as AI-VAD."""

    def __init__(self, weights: str, device, conf=0.5, imgsz=960, track=True):
        from ultralytics import YOLO
        self.model, self.device, self.conf, self.imgsz, self.track = YOLO(weights), device, conf, imgsz, track
        self.b = aivad_code.bbox_functions(need_detector=False)
        self.c = self.b.DATASET_CFGS["shanghaitech"]
        self.net = self.model.model

    def reset(self):                                     # new video = new track ids
        if self.track and getattr(self.model, "predictor", None) is not None:
            for t in getattr(self.model.predictor, "trackers", []) or []:
                t.reset()

    def __call__(self, prev, cur):
        kw = dict(conf=self.conf, imgsz=self.imgsz, classes=[0], device=self.device, verbose=False)
        r = (self.model.track(cur, persist=True, tracker="bytetrack.yaml", **kw) if self.track
             else self.model.predict(cur, **kw))[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None and len(boxes) else np.zeros((0, 17, 3))
        ids = (r.boxes.id.cpu().numpy().astype(np.int64) if r.boxes is not None and r.boxes.id is not None
               else -np.ones(len(boxes), np.int64))
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        keep = area >= self.c["min_area"]
        boxes, kps, ids, area = boxes[keep], kps[keep], ids[keep], area[keep]
        cls = np.zeros(len(boxes), np.int64)
        kept, _ = self.b.delete_overlapped_bboxes(boxes, area, self.c["cover_threshold"], np.arange(len(boxes)))
        idx = _rows(boxes, kept)
        boxes, kps, ids, cls = boxes[idx], kps[idx], ids[idx], cls[idx]
        fg = self.b.get_foreground_bboxes(np.stack([prev, cur]), boxes, self.c["contour_min_area"],
                                          self.c["binary_threshold"], self.c["gauss_mask_size"])
        n_det, n_fg = len(boxes), len(fg)
        if n_fg:
            boxes = np.concatenate([boxes, fg], 0)
            cls = np.concatenate([cls, np.zeros(n_fg, np.int64)])
            ids = np.concatenate([ids, -np.ones(n_fg, np.int64)])
        return boxes.astype(np.float32), cls, kps.astype(np.float32), ids, n_det


def _rows(all_boxes, kept):
    """Indices of the kept boxes (delete_overlapped_bboxes returns boxes, not indices)."""
    if len(kept) == 0:
        return np.zeros(0, np.int64)
    return np.array([int(np.argmin(np.abs(all_boxes - k).sum(1))) for k in kept], np.int64)


# ============================================================== optical flow
class FlowNet2Flow:
    """AI-VAD pre_processing/flows.py: FlowNet2 at 1024x640, BGR 0-255, flow resized back to the
    frame size (values stay in 1024x640 units, as in AI-VAD)."""
    SIZE = (1024, 640)

    def __init__(self, weights: str | None, device):
        self.net = aivad_code.flownet2_class()().to(device).eval()
        if weights:
            pre = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
            own = self.net.state_dict()
            own.update({k: v for k, v in pre.items() if k in own})
            self.net.load_state_dict(own)
        self.device = device

    @torch.no_grad()
    def __call__(self, prev, cur):
        h, w = cur.shape[:2]
        ims = np.array([cv2.resize(prev, self.SIZE), cv2.resize(cur, self.SIZE)]).astype(np.float32)
        ims = torch.from_numpy(ims).unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous().to(self.device)
        flow = self.net(ims)[0].float().cpu().numpy().transpose(1, 2, 0)
        return cv2.resize(flow, (w, h))


class RaftFlow:
    """Model 4: RAFT (torchvision, trained on FlyingChairs+Things+Sintel+KITTI+HD1K), 5.3M params
    vs. FlowNet2's 162.5M. Smaller input + fewer refinement steps = much faster."""

    def __init__(self, size=(512, 288), iters=12, device="cpu", small=False):
        from torchvision.models.optical_flow import (Raft_Large_Weights, Raft_Small_Weights,
                                                     raft_large, raft_small)
        if small:
            self.net = raft_small(weights=Raft_Small_Weights.C_T_V2)
        else:
            self.net = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        self.net = self.net.to(device).eval()
        self.size, self.iters, self.device = tuple(size), iters, device

    def _prep(self, img):
        x = cv2.cvtColor(cv2.resize(img, self.size), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(x).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(self, prev, cur):
        h, w = cur.shape[:2]
        flow = self.net(self._prep(prev), self._prep(cur), num_flow_updates=self.iters)[-1]
        return cv2.resize(flow[0].float().cpu().numpy().transpose(1, 2, 0), (w, h))


# ============================================================== pose (model 3)
def _alphapose_layers():
    """AlphaPose's layer files, loaded as a package without running AlphaPose's __init__
    (which needs compiled extensions we don't use)."""
    root = ALPHAPOSE / "alphapose" / "models" / "layers"
    if not root.exists():
        raise SystemExit("third_party/AlphaPose missing: run  python third_party/fetch.py")
    if "ap_layers" not in sys.modules:
        pkg = types.ModuleType("ap_layers")
        pkg.__path__ = [str(root)]
        sys.modules["ap_layers"] = pkg
    load = lambda n: importlib.import_module(f"ap_layers.{n}")
    return load("SE_Resnet").SEResnet, load("DUC").DUC


class FastPose(nn.Module):
    """AlphaPose alphapose/models/fastpose.py, ResNet-50 config (256x192, 17 COCO keypoints).
    Same layers and names as the original, so its checkpoint loads; only the ImageNet download
    in the original __init__ is skipped (the checkpoint replaces those weights anyway)."""

    def __init__(self, num_joints=17):
        super().__init__()
        SEResnet, DUC = _alphapose_layers()
        self.preact = SEResnet("resnet50")
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=nn.BatchNorm2d)
        self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=nn.BatchNorm2d)
        self.conv_out = nn.Conv2d(128, num_joints, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv_out(self.duc2(self.duc1(self.suffle1(self.preact(x)))))


def _affine(center, scale_w, out_w, out_h, inv=False):
    """AlphaPose utils/transforms.get_affine_transform with rot=0, shift=0."""
    src = np.array([center, center + [0, -0.5 * scale_w], center + [0.5 * scale_w, -0.5 * scale_w]], np.float32)
    dst = np.array([[out_w / 2, out_h / 2], [out_w / 2, out_h / 2 - out_w / 2], [out_w, out_h / 2 - out_w / 2]],
                   np.float32)
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


class FastPoseKeypoints:
    """AlphaPose single-person pose on each person box (its SimpleTransform.test_transform +
    heatmap_to_coord_simple): box -> 1.25x crop with aspect 3:4 -> 256x192 -> heatmaps -> keypoints."""
    IN_H, IN_W = 256, 192

    def __init__(self, weights: str | None, device):
        self.net = FastPose().to(device).eval()
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.device = device

    @torch.no_grad()
    def __call__(self, img, boxes) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        crops, frames = [], []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            c = np.array([x1 + w / 2, y1 + h / 2], np.float32)
            ar = self.IN_W / self.IN_H
            w, h = (w, w / ar) if w > ar * h else (h * ar, h)
            w, h = w * 1.25, h * 1.25
            t = _affine(c, w, self.IN_W, self.IN_H)
            crop = cv2.warpAffine(img, t, (self.IN_W, self.IN_H), flags=cv2.INTER_LINEAR)
            x = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255
            x[0] -= 0.406; x[1] -= 0.457; x[2] -= 0.480
            crops.append(x)
            frames.append((c, w, h))
        hms = self.net(torch.stack(crops).to(self.device)).float().cpu().numpy()      # (n, 17, 64, 48)
        out = np.zeros((len(boxes), 17, 3), np.float32)
        hm_h, hm_w = hms.shape[2:]
        for k, (hm, (c, w, h)) in enumerate(zip(hms, frames)):
            flat = hm.reshape(17, -1)
            idx, val = flat.argmax(1), flat.max(1)
            xy = np.stack([idx % hm_w, idx // hm_w], 1).astype(np.float32) * (val[:, None] > 0)
            for j in range(17):
                px, py = int(round(xy[j, 0])), int(round(xy[j, 1]))
                if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
                    xy[j] += np.sign([hm[j, py, px + 1] - hm[j, py, px - 1],
                                      hm[j, py + 1, px] - hm[j, py - 1, px]]) * 0.25
            t = _affine(c, w, hm_w, hm_h, inv=True)
            out[k, :, :2] = xy @ t[:, :2].T + t[:, 2]
            out[k, :, 2] = val
        return out


# ============================================================== appearance (model 3)
class ClipAppearance:
    """AI-VAD models/models.py: CLIP ViT-B/16 image embedding (512-d) of each object crop,
    224x224, AI-VAD's normalisation (CLIP mean/std applied to the BGR crop, as in its code)."""
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]

    def __init__(self, device, model=None):
        if model is None:
            import clip
            model, _ = clip.load("ViT-B/16", device=device)
        self.model, self.device = model.eval(), device
        self.net = model.visual

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:            # (n, 224, 224, 3) BGR uint8
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255)
        x = ((x - self.MEAN) / self.STD).to(self.device)
        return self.model.encode_image(x).float().cpu().numpy()
