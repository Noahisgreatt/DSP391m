"""Every neural network the project uses, written here in plain PyTorch.

No model is created through a library ("wrapper") call such as timm.create_model, YOLO(...),
clip.load, detectron2 or torchvision.models.*. Each network below is our own nn.Module whose
layer names follow the published checkpoint, so pretrained weights load directly (transfer
learning). tests/test_nets.py checks each one gives the same output as the reference
implementation when both have the same weights.

  ViT          ViT-B/32-384 image encoder          models 1 (random init) and 2 (ImageNet weights)
  CLIPVisual   CLIP ViT-B/16 image encoder         model 3 (appearance)
  RCNN         ResNet-50-FPN Faster R-CNN detector  model 3 (all COCO objects)
               + keypoint head (Keypoint R-CNN)    model 4 + behavior (people + 17 body keypoints)
  FastPose     AlphaPose ResNet-50 pose network    model 3 (layers from third_party/AlphaPose)
  FlowNet2 / FlowNetC  optical flow                models 3 / 4 (code from third_party/AIVAD;
               its 3 CUDA-only layers are re-written below in plain PyTorch)
Weights: python -m src.models.nets download <name> ...   (files go to weights/)
"""
from __future__ import annotations

import importlib
import math
import os
import re
import sys
import types
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, roi_align

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
WEIGHT_SOURCES = {   # name: (file, url or google-drive id)
    "vit_b32_384": ("vit_base_patch32_384.augreg_in21k_ft_in1k.safetensors",
                    "https://huggingface.co/timm/vit_base_patch32_384.augreg_in21k_ft_in1k/resolve/main/model.safetensors"),
    "maskrcnn": ("maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
                 "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"),
    "keypointrcnn": ("keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
                     "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth"),
    "clip_b16": ("CLIP-ViT-B-16.pt", "https://openaipublic.azureedge.net/clip/models/"
                 "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    # NVIDIA's original Google Drive link (1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da) is closed (403); same file
    # mirrored by EvalCrafter (CVPR 2024). Loaded weights-only and checked layer by layer (flownet2()).
    "flownet2": ("FlowNet2_checkpoint.pth.tar",
                 "https://huggingface.co/RaphaelLiu/EvalCrafter-Models/resolve/main/FlowNet2_checkpoint.pth.tar"),
    "fastpose": ("fast_res50_256x192.pth", "gdrive:1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn"),
}


def weight_path(name: str) -> Path:
    return WEIGHTS / WEIGHT_SOURCES[name][0]


def download(name: str) -> Path | None:
    """Path of a pretrained weight file (downloaded once). With VAD_NO_WEIGHTS=1 (tests, no
    internet) returns None and the networks keep their random weights."""
    if os.environ.get("VAD_NO_WEIGHTS"):
        print(f"VAD_NO_WEIGHTS: {name} keeps random weights")
        return None
    path = weight_path(name)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    src = WEIGHT_SOURCES[name][1]
    print(f"downloading {name} -> {path}")
    part = path.with_name(path.name + ".part")              # never a half-downloaded weight file
    try:
        if src.startswith("gdrive:"):
            import gdown
            if gdown.download(id=src[7:], output=str(part), quiet=False) is None:
                raise RuntimeError("Google Drive refused the download")
        else:
            torch.hub.download_url_to_file(src, str(part))
    except Exception as e:
        part.unlink(missing_ok=True)
        link = f"https://drive.google.com/uc?id={src[7:]}" if src.startswith("gdrive:") else src
        raise RuntimeError(f"could not download {name} ({type(e).__name__}).\n"
                           f"  Download it in your browser: {link}\n"
                           f"  and save it as: {path}") from None
    part.replace(path)
    return path


def safe_load(path):
    """torch.load that only reads tensors and plain values: a weight file can't run code."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as first:
        import numpy as np                                  # old checkpoints store numpy numbers
        core = getattr(np, "_core", None) or np.core
        allow = [np.dtype, core.multiarray.scalar] + [type(np.dtype(t)) for t in ("float64", "float32", "int64")]
        if not hasattr(torch.serialization, "add_safe_globals"):
            raise SystemExit(f"cannot safely read {path}: update PyTorch (>= 2.4)") from first
        torch.serialization.add_safe_globals(allow)
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:
            raise SystemExit(f"{path} contains more than weights, refusing to load it ({e})") from None


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ====================================================================== ViT (models 1, 2)
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), Mlp(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    """Vision Transformer (Dosovitskiy et al., 2021), ViT-B/32 at 384x384 by default.
    forward(x) -> the CLS token (b, 768); forward_features(x) -> all tokens (b, 1 + 12*12, 768).
    Layer names = timm's, so the ImageNet checkpoint (vit_b32_384) loads with load_pretrained()."""

    def __init__(self, img_size=384, patch=32, dim=768, depth=12, heads=12):
        super().__init__()
        n = (img_size // patch) ** 2
        self.patch_embed = PatchEmbed(patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.num_features = dim
        for m in self.modules():                        # same init scheme as the original
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        return self.norm(self.blocks(x))

    def forward(self, x):
        return self.forward_features(x)[:, 0]

    def load_pretrained(self, path=None):
        path = path or download("vit_b32_384")
        if path is None:
            return self
        from safetensors.torch import load_file
        sd = load_file(str(path))
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}   # ImageNet classifier
        self.load_state_dict(sd)
        return self


# ====================================================================== CLIP image encoder (model 3)
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm32(nn.LayerNorm):
    """CLIP's LayerNorm: computed in float32 even when the model runs in float16."""
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads)
        self.ln_1, self.ln_2 = LayerNorm32(d), LayerNorm32(d)
        self.mlp = nn.Sequential(OrderedDict(c_fc=nn.Linear(d, d * 4), gelu=QuickGELU(), c_proj=nn.Linear(d * 4, d)))

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class CLIPTransformer(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d, heads) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class CLIPVisual(nn.Module):
    """CLIP ViT-B/16 image encoder (Radford et al., 2021) -> 512-d embedding.
    Layer names = OpenAI's 'visual.*', so the released checkpoint (clip_b16) loads directly."""

    def __init__(self, res=224, patch=16, width=768, layers=12, heads=12, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch, patch, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width) * width ** -0.5)
        self.positional_embedding = nn.Parameter(torch.randn((res // patch) ** 2 + 1, width) * width ** -0.5)
        self.ln_pre, self.ln_post = LayerNorm32(width), LayerNorm32(width)
        self.transformer = CLIPTransformer(width, layers, heads)
        self.proj = nn.Parameter(torch.randn(width, out_dim) * width ** -0.5)

    def forward(self, x):
        x = self.conv1(x.to(self.conv1.weight.dtype)).flatten(2).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(len(x), 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = self.ln_pre(torch.cat([cls, x], 1) + self.positional_embedding.to(x.dtype))
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return self.ln_post(x[:, 0]) @ self.proj

    def load_pretrained(self, path=None):
        path = path or download("clip_b16")
        if path is None:
            return self
        archive = torch.jit.load(str(path), map_location="cpu")
        sd = {k[7:]: v for k, v in archive.state_dict().items() if k.startswith("visual.")}
        self.load_state_dict(sd)
        return self


# ====================================================================== R-CNN detector (models 3, 4, behavior)
class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with fixed statistics (detection backbones never update them)."""
    def __init__(self, n):
        super().__init__()
        for k, v in (("weight", torch.ones(n)), ("bias", torch.zeros(n)),
                     ("running_mean", torch.zeros(n)), ("running_var", torch.ones(n))):
            self.register_buffer(k, v)

    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        return x * scale[None, :, None, None] + (self.bias - self.running_mean * scale)[None, :, None, None]


class Bottleneck(nn.Module):
    def __init__(self, cin, width, stride=1, down=False):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, width, 1, bias=False), FrozenBatchNorm2d(width)
        self.conv2, self.bn2 = nn.Conv2d(width, width, 3, stride, 1, bias=False), FrozenBatchNorm2d(width)
        self.conv3, self.bn3 = nn.Conv2d(width, width * 4, 1, bias=False), FrozenBatchNorm2d(width * 4)
        self.downsample = (nn.Sequential(nn.Conv2d(cin, width * 4, 1, stride, bias=False), FrozenBatchNorm2d(width * 4))
                           if down else None)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class ResNet50(nn.Module):
    """ResNet-50 (He et al., 2016) returning the outputs of its 4 stages (strides 4, 8, 16, 32)."""
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False), FrozenBatchNorm2d(64)
        cin = 64
        for i, (n, width, stride) in enumerate(((3, 64, 1), (4, 128, 2), (6, 256, 2), (3, 512, 2)), 1):
            blocks = [Bottleneck(cin, width, stride, down=True)] + [Bottleneck(width * 4, width) for _ in range(n - 1)]
            setattr(self, f"layer{i}", nn.Sequential(*blocks))
            cin = width * 4

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        out = []
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
            out.append(x)
        return out


class FPN(nn.Module):
    """Feature Pyramid Network (Lin et al., 2017): 5 levels of 256 channels."""
    def __init__(self, in_channels=(256, 512, 1024, 2048), ch=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(c, ch, 1)) for c in in_channels)
        self.layer_blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1)) for _ in in_channels)

    def forward(self, xs):
        last = self.inner_blocks[-1](xs[-1])
        outs = [self.layer_blocks[-1](last)]
        for i in range(len(xs) - 2, -1, -1):
            lateral = self.inner_blocks[i](xs[i])
            last = lateral + F.interpolate(last, size=lateral.shape[-2:], mode="nearest")
            outs.insert(0, self.layer_blocks[i](last))
        outs.append(F.max_pool2d(outs[-1], 1, 2, 0))        # extra level ("pool")
        return outs


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body, self.fpn = ResNet50(), FPN()

    def forward(self, x):
        return self.fpn(self.body(x))


class RPNHead(nn.Module):
    def __init__(self, ch=256, anchors=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()))
        self.cls_logits, self.bbox_pred = nn.Conv2d(ch, anchors, 1), nn.Conv2d(ch, anchors * 4, 1)


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = RPNHead()


class BoxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc6, self.fc7 = nn.Linear(256 * 7 * 7, 1024), nn.Linear(1024, 1024)

    def forward(self, x):
        return F.relu(self.fc7(F.relu(self.fc6(x.flatten(1)))))


class BoxPredictor(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.cls_score, self.bbox_pred = nn.Linear(1024, classes), nn.Linear(1024, classes * 4)


class KeypointPredictor(nn.Module):
    def __init__(self, n=17):
        super().__init__()
        self.kps_score_lowres = nn.ConvTranspose2d(512, n, 4, stride=2, padding=1)

    def forward(self, x):
        return F.interpolate(self.kps_score_lowres(x), scale_factor=2.0, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False)


class ROIHeads(nn.Module):
    def __init__(self, classes, keypoints):
        super().__init__()
        self.box_head, self.box_predictor = BoxHead(), BoxPredictor(classes)
        if keypoints:
            layers, c = [], 256
            for _ in range(8):
                layers += [nn.Conv2d(c, 512, 3, padding=1), nn.ReLU()]
                c = 512
            self.keypoint_head = nn.Sequential(*layers)
            self.keypoint_predictor = KeypointPredictor(keypoints)


def decode_boxes(deltas, boxes, weights):
    """R-CNN box regression: deltas (n, k*4) relative to boxes (n, 4) -> (n, k, 4)."""
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx, cy = boxes[:, 0] + 0.5 * w, boxes[:, 1] + 0.5 * h
    wx, wy, ww, wh = weights
    dx, dy = deltas[:, 0::4] / wx, deltas[:, 1::4] / wy
    clip = math.log(1000.0 / 16)
    dw, dh = (deltas[:, 2::4] / ww).clamp(max=clip), (deltas[:, 3::4] / wh).clamp(max=clip)
    pcx, pcy = dx * w[:, None] + cx[:, None], dy * h[:, None] + cy[:, None]
    pw, ph = torch.exp(dw) * w[:, None], torch.exp(dh) * h[:, None]
    return torch.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], 2)


def clip_boxes(b, hw):
    return torch.stack([b[..., 0].clamp(0, hw[1]), b[..., 1].clamp(0, hw[0]),
                        b[..., 2].clamp(0, hw[1]), b[..., 3].clamp(0, hw[0])], -1)


def keep_min_size(b, s):
    return torch.where(((b[:, 2] - b[:, 0]) >= s) & ((b[:, 3] - b[:, 1]) >= s))[0]


class RCNN(nn.Module):
    """Faster R-CNN (Ren et al., 2015) with a ResNet-50-FPN backbone; with keypoints=17 it is
    Keypoint R-CNN (He et al., 2017). Inference only. Layer names = torchvision's checkpoints:
      RCNN(classes=91)                 + load_pretrained("maskrcnn")      COCO, 80 object classes
      RCNN(classes=2, keypoints=17)    + load_pretrained("keypointrcnn")  people + body keypoints
    __call__(list of RGB float tensors (3, H, W) in [0, 1]) -> list of dicts: boxes, scores,
    labels (+ keypoints (n, 17, 3), keypoints_scores)."""

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, classes=91, keypoints=0, min_size=800, max_size=1333, score_thresh=0.05,
                 nms_thresh=0.5, detections=100):
        super().__init__()
        self.backbone, self.rpn, self.roi_heads = Backbone(), RPN(), ROIHeads(classes, keypoints)
        self.min_size, self.max_size = min_size, max_size
        self.score_thresh, self.nms_thresh, self.detections = score_thresh, nms_thresh, detections
        self.keypoints = keypoints

    def load_pretrained(self, name, path=None):
        path = path or download(name)
        if path is None:
            return self
        sd, own = safe_load(path), self.state_dict()
        # torchvision's 2019 files use older names: fpn.inner_blocks.0.weight -> .0.0.weight,
        # rpn.head.conv.weight -> .conv.0.0.weight (torchvision renames them the same way when loading)
        old = lambda k: [k, re.sub(r"\.0\.(weight|bias)$", r".\1", k), re.sub(r"\.0\.0\.(weight|bias)$", r".\1", k)]
        new = {}
        for k in own:                                         # the mask head of Mask R-CNN is not used
            src = next((c for c in old(k) if c in sd), None)
            if src is None or sd[src].shape != own[k].shape:
                raise SystemExit(f"{path}: no weights for layer {k}; delete the file and download it again")
            new[k] = sd[src]
        self.load_state_dict(new)
        return self

    # ---- input: normalise, resize (shorter side min_size), pad to a multiple of 32
    def _prepare(self, images):
        sizes, out = [], []
        mean = torch.tensor(self.MEAN, device=images[0].device)[:, None, None]
        std = torch.tensor(self.STD, device=images[0].device)[:, None, None]
        for im in images:
            im = (im - mean) / std
            h, w = im.shape[-2:]
            scale = min(self.min_size / min(h, w), self.max_size / max(h, w))
            im = F.interpolate(im[None], scale_factor=scale, mode="bilinear", recompute_scale_factor=True,
                               align_corners=False)[0]
            sizes.append(tuple(im.shape[-2:]))
            out.append(im)
        H = int(math.ceil(max(s[0] for s in sizes) / 32) * 32)
        W = int(math.ceil(max(s[1] for s in sizes) / 32) * 32)
        batch = out[0].new_zeros(len(out), 3, H, W)
        for i, im in enumerate(out):
            batch[i, :, :im.shape[1], :im.shape[2]] = im
        return batch, sizes

    # ---- region proposals
    def _anchors(self, feats, image_hw):
        anchors = []
        for f, size in zip(feats, (32, 64, 128, 256, 512)):
            ratios = torch.tensor((0.5, 1.0, 2.0), device=f.device)
            hr = torch.sqrt(ratios)
            ws, hs = (1 / hr) * size, hr * size
            base = (torch.stack([-ws, -hs, ws, hs], 1) / 2).round()
            gh, gw = f.shape[-2:]
            sy, sx = image_hw[0] // gh, image_hw[1] // gw
            yy, xx = torch.meshgrid(torch.arange(gh, device=f.device, dtype=torch.int32) * sy,
                                    torch.arange(gw, device=f.device, dtype=torch.int32) * sx, indexing="ij")
            shifts = torch.stack([xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1)], 1)
            anchors.append((shifts.view(-1, 1, 4) + base.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def _proposals(self, feats, batch_hw, sizes, pre_n=1000, post_n=1000):
        head = self.rpn.head
        anchors = self._anchors(feats, batch_hw)
        logits, deltas, per_level = [], [], []
        for f in feats:
            t = head.conv(f)
            n, a, h, w = head.cls_logits(t).shape
            logits.append(head.cls_logits(t).view(n, a, 1, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1))
            deltas.append(head.bbox_pred(t).view(n, a, 4, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1, 4))
            per_level.append(a * h * w)
        logits, deltas, anchors = torch.cat(logits, 1), torch.cat(deltas, 1), torch.cat(anchors, 0)
        props = []
        for i, hw in enumerate(sizes):
            boxes = decode_boxes(deltas[i], anchors, (1.0, 1.0, 1.0, 1.0))[:, 0]
            idx, off = [], 0
            for n_lvl in per_level:                              # top pre_n per level
                k = min(pre_n, n_lvl)
                idx.append(logits[i, off:off + n_lvl].topk(k).indices + off)
                off += n_lvl
            idx = torch.cat(idx)
            lvl = torch.cat([torch.full((min(pre_n, n),), j, device=idx.device) for j, n in enumerate(per_level)])
            b, s = clip_boxes(boxes[idx], hw), torch.sigmoid(logits[i, idx])
            keep = keep_min_size(b, 1e-3)
            b, s, lvl = b[keep], s[keep], lvl[keep]
            keep = torch.where(s >= 0.0)[0]
            b, s, lvl = b[keep], s[keep], lvl[keep]
            keep = batched_nms(b, s, lvl, 0.7)[:post_n]
            props.append(b[keep])
        return props

    # ---- pooling from the right pyramid level (FPN paper rule)
    def _pool(self, feats, boxes, sizes, out):
        feats = feats[:4]
        max_hw = (max(s[0] for s in sizes), max(s[1] for s in sizes))
        scales = [2 ** round(math.log2(f.shape[-2] / max_hw[0])) for f in feats]
        k_min, k_max = -math.log2(scales[0]), -math.log2(scales[-1])
        rois = torch.cat([torch.cat([torch.full_like(b[:, :1], i), b], 1) for i, b in enumerate(boxes)])
        area = torch.cat([(b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) for b in boxes])
        lvl = torch.floor(4 + torch.log2(torch.sqrt(area) / 224) + 1e-6).clamp(k_min, k_max) - k_min
        res = rois.new_zeros(len(rois), feats[0].shape[1], out, out)
        for j, (f, sc) in enumerate(zip(feats, scales)):
            idx = torch.where(lvl == j)[0]
            res[idx] = roi_align(f, rois[idx], (out, out), spatial_scale=sc, sampling_ratio=2, aligned=False).to(res.dtype)
        return res

    @staticmethod
    def _heatmaps_to_keypoints(maps, rois):
        x0, y0 = rois[:, 0], rois[:, 1]
        w, h = (rois[:, 2] - rois[:, 0]).clamp(min=1), (rois[:, 3] - rois[:, 1]).clamp(min=1)
        wc, hc = w.ceil(), h.ceil()
        k = maps.shape[1]
        xy = torch.zeros(len(rois), k, 3, device=maps.device)
        scores = torch.zeros(len(rois), k, device=maps.device)
        for i in range(len(rois)):
            rw, rh = int(wc[i]), int(hc[i])
            m = F.interpolate(maps[i][:, None], size=(rh, rw), mode="bicubic", align_corners=False)[:, 0]
            pos = m.reshape(k, -1).argmax(1)
            xi = pos % rw
            yi = torch.div(pos - xi, rw, rounding_mode="floor")
            xy[i, :, 0] = (xi.float() + 0.5) * (w[i] / rw) + x0[i]
            xy[i, :, 1] = (yi.float() + 0.5) * (h[i] / rh) + y0[i]
            xy[i, :, 2] = 1
            scores[i] = m[torch.arange(k), yi, xi]
        return xy, scores

    @torch.no_grad()
    def forward(self, images):
        orig = [tuple(im.shape[-2:]) for im in images]
        x, sizes = self._prepare(images)
        feats = self.backbone(x)
        props = self._proposals(feats, x.shape[-2:], sizes)
        rh = self.roi_heads
        t = rh.box_head(self._pool(feats, props, sizes, 7))
        logits, deltas = rh.box_predictor.cls_score(t), rh.box_predictor.bbox_pred(t)
        boxes_all = decode_boxes(deltas, torch.cat(props), (10.0, 10.0, 5.0, 5.0))
        probs = F.softmax(logits, -1)
        results, start = [], 0
        for i, hw in enumerate(sizes):
            n = len(props[i])
            b = clip_boxes(boxes_all[start:start + n], hw)[:, 1:].reshape(-1, 4)
            c = probs[start:start + n].shape[1]
            s = probs[start:start + n, 1:].reshape(-1)
            lab = torch.arange(1, c, device=s.device).repeat(n)
            start += n
            keep = torch.where(s > self.score_thresh)[0]
            b, s, lab = b[keep], s[keep], lab[keep]
            keep = keep_min_size(b, 1e-2)
            b, s, lab = b[keep], s[keep], lab[keep]
            keep = batched_nms(b, s, lab, self.nms_thresh)[:self.detections]
            results.append({"boxes": b[keep], "scores": s[keep], "labels": lab[keep]})
        if self.keypoints:
            boxes = [r["boxes"] for r in results]
            if sum(len(b) for b in boxes):
                maps = rh.keypoint_predictor(rh.keypoint_head(self._pool(feats, boxes, sizes, 14)))
                off = 0
                for r in results:
                    n = len(r["boxes"])
                    r["keypoints"], r["keypoints_scores"] = self._heatmaps_to_keypoints(maps[off:off + n], r["boxes"])
                    off += n
            else:
                for r in results:
                    r["keypoints"] = torch.zeros(0, self.keypoints, 3, device=x.device)
                    r["keypoints_scores"] = torch.zeros(0, self.keypoints, device=x.device)
        for r, hw, o in zip(results, sizes, orig):          # back to the original image size
            ry, rx = torch.tensor(o[0] / hw[0]), torch.tensor(o[1] / hw[1])
            r["boxes"] = r["boxes"] * torch.stack([rx, ry, rx, ry]).to(r["boxes"])
            if "keypoints" in r:
                r["keypoints"][..., 0] *= rx.to(r["keypoints"])
                r["keypoints"][..., 1] *= ry.to(r["keypoints"])
        return results


def to_rgb_tensor(bgr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)


# COCO category ids (1..90, with gaps) -> the 80 contiguous classes AI-VAD uses (person = 0)
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
            86, 87, 88, 89, 90]
COCO_TO_CONTIGUOUS = {c: i for i, c in enumerate(COCO_IDS)}


# ====================================================================== FastPose (model 3)
ALPHAPOSE = ROOT / "third_party" / "AlphaPose"


def _alphapose_layers():
    """AlphaPose's layer files, loaded as a package without AlphaPose's __init__ (which needs
    compiled extensions we don't use)."""
    root = ALPHAPOSE / "alphapose" / "models" / "layers"
    if not root.exists():
        raise SystemExit("third_party/AlphaPose missing: run  python third_party/fetch.py")
    if "ap_layers" not in sys.modules:
        pkg = types.ModuleType("ap_layers")
        pkg.__path__ = [str(root)]
        sys.modules["ap_layers"] = pkg
    return (importlib.import_module("ap_layers.SE_Resnet").SEResnet,
            importlib.import_module("ap_layers.DUC").DUC)


class FastPose(nn.Module):
    """AlphaPose FastPose ResNet-50 (Fang et al., TPAMI 2022): person crop 256x192 -> 17 heatmaps.
    Same layers and names as alphapose/models/fastpose.py, so its checkpoint loads."""
    IN_H, IN_W = 256, 192

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

    def load_pretrained(self, path=None):
        path = path or download("fastpose")
        if path is not None:
            self.load_state_dict(safe_load(path))
        return self

    @staticmethod
    def _affine(center, scale_w, out_w, out_h, inv=False):
        """AlphaPose get_affine_transform (rotation 0)."""
        src = np.array([center, center + [0, -0.5 * scale_w], center + [0.5 * scale_w, -0.5 * scale_w]], np.float32)
        dst = np.array([[out_w / 2, out_h / 2], [out_w / 2, out_h / 2 - out_w / 2],
                        [out_w, out_h / 2 - out_w / 2]], np.float32)
        return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)

    @torch.no_grad()
    def keypoints(self, bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """AlphaPose test transform + heatmap decoding -> (n, 17, 3) x, y, score in the image."""
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        dev = next(self.parameters()).device
        crops, frames = [], []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            c = np.array([x1 + w / 2, y1 + h / 2], np.float32)
            ar = self.IN_W / self.IN_H
            w, h = (w, w / ar) if w > ar * h else (h * ar, h)
            w = w * 1.25
            crop = cv2.warpAffine(bgr, self._affine(c, w, self.IN_W, self.IN_H), (self.IN_W, self.IN_H),
                                  flags=cv2.INTER_LINEAR)
            t = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255
            t[0] -= 0.406; t[1] -= 0.457; t[2] -= 0.480
            crops.append(t)
            frames.append((c, w))
        hms = self(torch.stack(crops).to(dev)).float().cpu().numpy()
        out = np.zeros((len(boxes), 17, 3), np.float32)
        hh, hw = hms.shape[2:]
        for k, (hm, (c, w)) in enumerate(zip(hms, frames)):
            flat = hm.reshape(17, -1)
            idx, val = flat.argmax(1), flat.max(1)
            xy = np.stack([idx % hw, idx // hw], 1).astype(np.float32) * (val[:, None] > 0)
            for j in range(17):
                px, py = int(round(xy[j, 0])), int(round(xy[j, 1]))
                if 1 < px < hw - 1 and 1 < py < hh - 1:
                    xy[j] += np.sign([hm[j, py, px + 1] - hm[j, py, px - 1], hm[j, py + 1, px] - hm[j, py - 1, px]]) * 0.25
            t = self._affine(c, w, hw, hh, inv=True)
            out[k, :, :2] = xy @ t[:, :2].T + t[:, 2]
            out[k, :, 2] = val
        return out


# ====================================================================== FlowNet2 / FlowNetC (models 3, 4)
def _correlation_forward(input1, input2, rbot1, rbot2, output, pad_size, kernel_size, max_displacement,
                         stride1, stride2, corr_multiply):
    """= correlation_cuda_kernel.cu: out[b, (tj+r)*(2r+1)+(ti+r), y, x] =
    sum_c in1[c,y,x] * in2[c, y+tj*stride2, x+ti*stride2] / C, zero padding."""
    assert kernel_size == 1 and stride1 == 1 and pad_size == max_displacement, "FlowNetC settings"
    b, c, h, w = input1.shape
    r = max_displacement // stride2
    p2 = F.pad(input2, (pad_size,) * 4)
    out = input1.new_empty(b, (2 * r + 1) ** 2, h, w)
    k = 0
    for tj in range(-r, r + 1):
        for ti in range(-r, r + 1):
            dy, dx = pad_size + tj * stride2, pad_size + ti * stride2
            out[:, k] = (input1 * p2[:, :, dy:dy + h, dx:dx + w]).sum(1) / c
            k += 1
    output.resize_(out.shape).copy_(out)
    return 1


def _resample2d_forward(input1, input2, output, kernel_size, bilinear):
    """= resample2d_kernel.cu: warp input1 by the flow input2, bilinear, neighbours clamped."""
    assert kernel_size == 1 and bilinear
    b, c, h, w = input1.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=input1.device, dtype=input1.dtype),
                            torch.arange(w, device=input1.device, dtype=input1.dtype), indexing="ij")
    xf, yf = xs + input2[:, 0], ys + input2[:, 1]
    x0, y0 = torch.floor(xf), torch.floor(yf)
    al, be = (xf - x0).unsqueeze(1), (yf - y0).unsqueeze(1)
    xl, xr = x0.clamp(0, w - 1).long(), (x0 + 1).clamp(0, w - 1).long()
    yt, yb = y0.clamp(0, h - 1).long(), (y0 + 1).clamp(0, h - 1).long()
    flat = input1.reshape(b, c, h * w)
    g = lambda yy, xx: flat.gather(2, (yy * w + xx).reshape(b, 1, h * w).expand(b, c, h * w)).reshape(b, c, h, w)
    output.copy_((1 - al) * (1 - be) * g(yt, xl) + al * (1 - be) * g(yt, xr) + (1 - al) * be * g(yb, xl) + al * be * g(yb, xr))
    return 1


def _channelnorm_forward(input1, output, norm_deg):
    """= channelnorm_kernel.cu: per-pixel L2 norm over channels."""
    output.copy_(input1.pow(norm_deg).sum(1, keepdim=True).pow(1.0 / norm_deg))
    return 1


def _install_flownet_ops():
    """FlowNet2's code imports 3 compiled CUDA modules; register plain-PyTorch versions instead."""
    def no_backward(*_):
        raise NotImplementedError("inference only")
    for name, fwd in (("correlation_cuda", _correlation_forward), ("resample2d_cuda", _resample2d_forward),
                      ("channelnorm_cuda", _channelnorm_forward)):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.forward, m.backward = fwd, no_backward
            sys.modules[name] = m


def flownet2(weights: bool = True) -> nn.Module:
    """FlowNet2 (Ilg et al., CVPR 2017), AI-VAD's code (third_party/AIVAD), 162.5M parameters."""
    aivad = ROOT / "third_party" / "AIVAD"
    if not aivad.exists():
        raise SystemExit("third_party/AIVAD missing: run  python third_party/fetch.py")
    if str(aivad) not in sys.path:
        sys.path.insert(0, str(aivad))
    _install_flownet_ops()
    net = importlib.import_module("pre_processing.flownet_networks.flownet2_models").FlowNet2()
    path = download("flownet2") if weights else None
    if path is not None:
        pre = safe_load(path)["state_dict"]
        own = net.state_dict()
        bad = [k for k in own if k not in pre or pre[k].shape != own[k].shape]
        if bad:
            raise SystemExit(f"{path} is not the FlowNet2 checkpoint ({len(bad)} layers missing or "
                             f"different, e.g. {bad[0]}); delete it and download it again")
        net.load_state_dict({k: pre[k] for k in own})
    return net.eval()


class FlowNetC(nn.Module):
    """The FIRST stage of FlowNet2 alone (FlowNetC + FlowNet2's input normalisation and x4
    upsampling), with FlowNet2's trained weights. Model 4 uses it instead of the full 5-network
    FlowNet2: ~4x fewer parameters, several times faster."""

    def __init__(self, weights: bool = True):
        super().__init__()
        full = flownet2(weights)
        self.flownetc, self.div_flow, self.rgb_max = full.flownetc, full.div_flow, full.rgb_max

    @torch.no_grad()
    def forward(self, inputs):                          # (b, 3, 2, h, w), 0-255, like FlowNet2
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1)
        flow = self.flownetc(x)[0] * self.div_flow
        return F.interpolate(flow, scale_factor=4, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="download pretrained weights into weights/ and check each loads")
    ap.add_argument("cmd", choices=["download"])
    ap.add_argument("names", nargs="+", help=f"any of {list(WEIGHT_SOURCES)}, or: astt_improved aivad aivad_lite behavior")
    a = ap.parse_args()
    groups = {"astt_improved": ["vit_b32_384"], "aivad": ["maskrcnn", "flownet2", "fastpose", "clip_b16"],
              "aivad_lite": ["keypointrcnn", "flownet2"], "behavior": ["keypointrcnn"]}
    load = {"vit_b32_384": lambda: ViT().load_pretrained(), "maskrcnn": lambda: RCNN(91).load_pretrained("maskrcnn"),
            "keypointrcnn": lambda: RCNN(2, 17).load_pretrained("keypointrcnn"), "flownet2": lambda: flownet2(),
            "fastpose": lambda: FastPose().load_pretrained(), "clip_b16": lambda: CLIPVisual().load_pretrained()}
    failed = []
    for w in dict.fromkeys(w for n in a.names for w in groups.get(n, [n])):   # each file once
        try:                                     # download, then load it into its network (a bad file shows up now)
            print("ok", download(w))
            load[w]()
            print(f"   loads into {w}: ok")
        except (RuntimeError, SystemExit) as e:
            print("FAILED", e)
            failed.append(w)
    if failed:
        raise SystemExit(f"\nproblem with: {', '.join(failed)} (see above); everything else is ready")
