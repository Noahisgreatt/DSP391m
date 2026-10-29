"""Every neural network the project uses, written here in plain PyTorch.

No model is created through a library ("wrapper") call such as timm.create_model, YOLO(...),
torch.hub.load or torchvision.models.*. Each network below is our own nn.Module whose layer names
follow the published checkpoint, so pretrained weights load directly (transfer learning).
tests/test_nets.py checks each one gives the same output as the official implementation when both
have the same weights.

  ViT     ViT-B/32-384 image encoder              models 1 (random init) and 2 (ImageNet weights)
  Hiera   Hiera-B video transformer (Kinetics-400) models 3 and 4 (MULDE features)
  RCNN    ResNet-50-FPN Keypoint R-CNN            behavior rules (people + 17 body keypoints)
Weights: python -m src.models.nets download <name> ...   (files go to weights/)
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, roi_align

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
WEIGHT_SOURCES = {   # name: (file, url)
    "vit_b32_384": ("vit_base_patch32_384.augreg_in21k_ft_in1k.safetensors",
                    "https://huggingface.co/timm/vit_base_patch32_384.augreg_in21k_ft_in1k/resolve/main/model.safetensors"),
    "hiera_b16": ("hiera_base_16x224.pth", "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth"),   # K400
    "keypointrcnn": ("keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
                     "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth"),
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
        torch.hub.download_url_to_file(src, str(part))
    except Exception as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {name} ({type(e).__name__}).\n"
                           f"  Download it in your browser: {src}\n  and save it as: {path}") from None
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


def half_dtype(device):
    """bf16 on GPUs that have it (RTX 30xx / 40xx, A100 ...), else fp16 (Kaggle's T4 / P100)."""
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def autocast(device):
    """Mixed precision on a GPU (faster, less memory); plain fp32 on the CPU."""
    return torch.autocast("cuda", dtype=half_dtype(device)) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


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


# ====================================================================== Hiera-B video (models 3, 4)
def unroll(x, size, strides, times):
    """Hiera's token order: apply `strides` `times` times so each pooling window is contiguous."""
    b, _, c = x.shape
    x = x.view(b, *size, c)
    for _ in range(times):
        size = [n // s for n, s in zip(size, strides)]
        x = x.view(b, *[v for n, s in zip(size, strides) for v in (n, s)], c)
        k = x.dim()
        x = x.permute(0, *range(2, k - 1, 2), *range(1, k - 1, 2), k - 1).flatten(0, len(strides))
        b *= math.prod(strides)
    return x.reshape(-1, math.prod(size) * math.prod(strides) ** times, c)


class HieraAttention(nn.Module):
    """Mask-unit attention (attention inside windows) or global attention, with optional q max-pooling."""

    def __init__(self, dim, dim_out, heads, q_stride, window, windowed):
        super().__init__()
        self.heads, self.q_stride, self.window, self.windowed = heads, q_stride, window, windowed
        self.qkv, self.proj = nn.Linear(dim, 3 * dim_out), nn.Linear(dim_out, dim_out)

    def forward(self, x):
        b, n, _ = x.shape
        w = n // (self.q_stride * self.window) if self.windowed else 1
        q, k, v = self.qkv(x).reshape(b, -1, w, 3, self.heads, self.qkv.out_features // 3 // self.heads).permute(3, 0, 4, 2, 1, 5)
        if self.q_stride > 1:
            q = q.view(b, self.heads, w, self.q_stride, -1, q.shape[-1]).max(dim=3).values
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 3).reshape(b, -1, self.proj.out_features))


class HieraBlock(nn.Module):
    def __init__(self, dim, dim_out, heads, q_stride, window, windowed):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim_out, eps=1e-6)
        self.attn, self.mlp = HieraAttention(dim, dim_out, heads, q_stride, window, windowed), Mlp(dim_out, 4 * dim_out)
        self.proj = nn.Linear(dim, dim_out) if dim != dim_out else None

    def forward(self, x):
        xn = self.norm1(x)
        if self.proj is not None:                                   # next stage: wider, fewer tokens
            x = self.proj(xn).view(x.shape[0], self.attn.q_stride, -1, self.proj.out_features).max(dim=1).values
        x = x + self.attn(xn)
        return x + self.mlp(self.norm2(x))


class Hiera(nn.Module):
    """Hiera-B video (Ryali et al., ICML 2023), Kinetics-400: a clip of 16 frames x 224 x 224
    (every 4th frame, RGB, mean 0.45 / std 0.225) -> one 768-d feature (mean of the last stage's
    tokens + LayerNorm, the classifier input). Layer names = the official checkpoint."""

    def __init__(self, embed_dim=96, heads=1, stages=(2, 3, 16, 3)):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(3, embed_dim, (3, 7, 7), (2, 4, 4), (1, 3, 3))
        self.tokens = (8, 56, 56)                                   # (16, 224, 224) / patch stride (2, 4, 4)
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 56 * 56, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 8, embed_dim))
        ends = [sum(stages[:i + 1]) - 1 for i in range(len(stages))]           # last block of each stage
        window, stage, dim, self.blocks = 64, 0, embed_dim, nn.ModuleList()    # mask unit (1, 8, 8) = 64 tokens
        for i in range(sum(stages)):
            dim_out, pool = dim, i - 1 in ends[:3]                 # q-pooling at the first block of stages 2-4
            if i - 1 in ends:
                dim_out, heads, stage = dim * 2, heads * 2, stage + 1
                window //= 4 if pool else 1
            self.blocks.append(HieraBlock(dim, dim_out, heads, 4 if pool else 1, window, stage < 2 or (stage == 2 and i - 1 in ends)))
            dim = dim_out
        self.norm, self.dim = nn.LayerNorm(dim, eps=1e-6), dim

    def forward(self, x):                                          # (b, 3, 16, 224, 224) -> (b, 768)
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        t, hw = self.tokens[0], self.tokens[1] * self.tokens[2]
        x = x + self.pos_embed_spatial.repeat(1, t, 1) + self.pos_embed_temporal.repeat_interleave(hw, dim=1)
        x = unroll(x, list(self.tokens), (1, 2, 2), 3)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x.mean(dim=1))

    def load_pretrained(self, path=None):
        path = path or download("hiera_b16")
        if path is not None:
            sd = safe_load(path)["model_state"]
            self.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.")})
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
      RCNN(classes=2, keypoints=17)    + load_pretrained("keypointrcnn")  people + body keypoints
      RCNN(classes=91)                 Faster R-CNN, 80 COCO classes (tested, not used)
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="download pretrained weights into weights/ and check each loads")
    ap.add_argument("cmd", choices=["download"])
    ap.add_argument("names", nargs="+", help=f"any of {list(WEIGHT_SOURCES)}, or a model / behavior")
    a = ap.parse_args()
    groups = {"astt_baseline": [], "astt_improved": ["vit_b32_384"], "mulde": ["hiera_b16"], "mulde_improved": ["hiera_b16"],
              "behavior": ["keypointrcnn"]}
    load = {"vit_b32_384": lambda: ViT().load_pretrained(), "hiera_b16": lambda: Hiera().load_pretrained(),
            "keypointrcnn": lambda: RCNN(2, 17).load_pretrained("keypointrcnn")}
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
