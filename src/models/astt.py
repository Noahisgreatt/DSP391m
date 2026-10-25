"""Models 1 and 2: ASTT, a transformer that PREDICTS the next frame from the 4 previous ones.
Trained on normal videos only, so a frame it predicts badly (high error) is abnormal.

Model 1  astt_baseline  the thesis model, UNCHANGED (Tran Minh Tung, IEEE TCSVT 2024; code
         third_party/ASTT @ 1d05305). 150,579,876 parameters.
Model 2  astt_improved  our upgrade. 100,379,299 parameters.

Model 1, why a small adapter is needed: the repo never uploaded its final model.py. It has
  model_ste_tte.py      STE (ViT) + TTE + decoder, the cross-attention line commented out
  model_baseline_ca.py  how cross-attention is applied: feat[:, 0] + self.cross_att(feat)
so ASTTThesis = model_ste_tte.VisionTransformer with that line switched on. The repo builds its
ViT with timm.create_model('vit_base_patch32_384'); that library call is replaced by our own ViT
(src/models/nets.py) with the same layers, verified to give identical outputs.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from src.models.nets import ViT, n_params

# ---------------------------------------------------------------- the thesis repo, with our ViT
_saved = sys.modules.get("timm")
sys.modules["timm"] = types.SimpleNamespace(              # the repo does `import timm`: give it a
    create_model=lambda name, num_classes=0, pretrained=False, **kw: ViT())   # stand-in = our ViT
import third_party.ASTT.model_ste_tte as repo             # noqa: E402

if _saved is None:                                        # leave sys.modules as it was
    del sys.modules["timm"]
else:
    sys.modules["timm"] = _saved


class ASTTThesis(repo.VisionTransformer):
    """Model 1. model_ste_tte.VisionTransformer + the cross-attention of model_baseline_ca.py.
    Only edit: `x.float().cuda()` -> `x.float().to(<model device>)` (same on a GPU)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cross_att = repo.CrossAttention(768, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)

    def forward(self, x):                                 # x: (b, 4, 3, 384, 384)
        x = x.float().to(self.cls_token.device)
        all_tok = None
        for i in range(x.shape[1]):                       # model_ste_tte.py lines 355-366
            tok = self.spatial_transformer(x[:, i]).unsqueeze(1)
            all_tok = tok if all_tok is None else torch.cat((tok, all_tok), 1)    # newest frame first
        emb = torch.cat([self.cls_token.repeat(len(x), 1, 1), all_tok], dim=1)
        emb = self.temporal_transformer(emb)
        emb = emb[:, 0, :].unsqueeze(1) + self.cross_att(emb)          # model_baseline_ca.py line 386
        return self.decoder(emb)


# ---------------------------------------------------------------- model 2
class ASTTImproved(nn.Module):
    """Model 2: same task and the same building blocks (ViT-B/32 STE, the repo's temporal
    Encoder, CrossAttention and decoder convolutions). Changes, all in configs/astt_improved.yaml:
      * ImageNet-pretrained ViT (transfer learning)
      * keep each frame's 12x12 patch tokens; the temporal transformer runs on every position
        (model 1 squeezes a frame into ONE vector and draws 256x256 pixels from it)
      * the repo's cross-attention adds one scene-level vector to every position
      * output = last input frame + predicted change (motion is where anomalies are)"""

    def __init__(self, image_size=384, num_frames=4, emb_dim=768, mlp_dim=1536, num_heads=6, num_layers=2,
                 dropout_rate=0.1, attn_dropout_rate=0.0, pretrained=True, residual_last_frame=True):
        super().__init__()
        self.spatial_transformer = ViT(image_size)
        if pretrained:
            self.spatial_transformer.load_pretrained()
        self.grid, self.residual = image_size // 32, residual_last_frame
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.temporal_transformer = repo.Encoder(num_patches=num_frames, emb_dim=emb_dim, mlp_dim=mlp_dim,
                                                 num_layers=num_layers, num_heads=num_heads,
                                                 dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate)
        self.cross_att = repo.CrossAttention(emb_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.to_map = nn.Sequential(nn.Conv2d(emb_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.decoder = repo.Decoder().decoder              # the repo's conv stack (256x16x16 -> 256x256)

    def forward(self, x):                                  # (b, t, 3, h, w), oldest first
        x = x.float()
        b, t = x.shape[:2]
        tok = self.spatial_transformer.forward_features(x.flatten(0, 1))     # (b*t, 1+g*g, d)
        n, d = tok.shape[1:]
        seq = tok.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        seq = torch.cat([self.cls_token.expand(b * n, -1, -1), seq], 1)
        tok = self.temporal_transformer(seq)[:, 0].view(b, n, d)
        tok = tok + self.cross_att(tok)
        fmap = tok[:, 1:].transpose(1, 2).reshape(b, d, self.grid, self.grid)
        out = self.decoder(self.to_map(F.interpolate(fmap, size=(16, 16), mode="bilinear", align_corners=False)))
        if self.residual:
            last = F.interpolate(x[:, -1], size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
            out = (last + 2 * out).clamp(-1, 1)
        return out


# ---------------------------------------------------------------- build / count / load
def build(cfg, pretrained=None):
    m = cfg["model"]
    if m["name"] == "astt_baseline":     # arguments exactly as the repo's config.py get_b16_config
        return ASTTThesis(image_size=(m["image_size"],) * 2, patch_size=(m["patch_size"],) * 2, emb_dim=m["emb_dim"],
                          mlp_dim=m["mlp_dim"], num_heads=m["num_heads"], num_layers=m["num_layers"],
                          num_classes=m["num_classes"], attn_dropout_rate=m["attn_dropout_rate"],
                          dropout_rate=m["dropout_rate"], num_frames=m["num_frames"])
    return ASTTImproved(m["image_size"], m["num_frames"], m["emb_dim"], m["mlp_dim"], m["num_heads"], m["num_layers"],
                        m["dropout_rate"], m["attn_dropout_rate"], m["pretrained"] if pretrained is None else pretrained,
                        m.get("residual_last_frame", True))


def count_parameters(model) -> dict:
    out = {"STE (ViT-B/32-384)": n_params(model.spatial_transformer),
           "TTE (+ temporal CLS)": n_params(model.temporal_transformer) + model.cls_token.numel(),
           "Cross-attention": n_params(model.cross_att)}
    if isinstance(model, ASTTThesis):
        out["Decoder"] = n_params(model.decoder)
        out["Sphere (unused, kept from repo)"] = model.center.numel() + model.radius.numel()
    else:
        out["Decoder"] = n_params(model.to_map) + n_params(model.decoder)
    out["Total"] = n_params(model)
    return out


def load_trained(cfg, checkpoint, device):
    model = build(cfg, pretrained=False).to(device)          # weights come from the checkpoint
    sd = torch.load(checkpoint, map_location=device)
    model.load_state_dict(sd.get("state_dict", sd))
    return model.eval()


# ---------------------------------------------------------------- scoring
def autocast(cfg, device):
    if device.type == "cuda" and cfg["train"].get("amp", False):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def gradient_loss(pred, target):
    """Image-gradient difference loss (Liu et al., CVPR 2018): keeps predicted edges sharp."""
    g = lambda x: ((x[..., :, 1:] - x[..., :, :-1]).abs(), (x[..., 1:, :] - x[..., :-1, :]).abs())
    (px, py), (tx, ty) = g(pred), g(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


@torch.no_grad()
def score_video(model, cfg, video_dir, device, maps=False, num_workers=None):
    """The repo's test_all_scenes: predict frame t from t-4..t-1, score = MSE at 256x256.
    -> mse (n_frames,), NaN for the first 4 frames [, error maps for drawing]."""
    from src.data import video_dataset
    ds, n = video_dataset(cfg, Path(video_dir))
    dl = tud.DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"] if num_workers is None else num_workers)
    T = cfg["model"]["num_frames"]
    mse = np.full(n, np.nan, np.float32)
    emaps = np.zeros((n, 256, 256), np.float16) if maps else None
    model.eval()
    t = T
    for b in dl:
        target, inp = b["256"][:, T].to(device).float(), b["standard"][:, :T].to(device)
        with autocast(cfg, device):
            pred = model(inp)
        err = (pred.float() - target) ** 2
        k = len(err)
        mse[t:t + k] = err.mean(dim=(1, 2, 3)).cpu().numpy()
        if maps:
            emaps[t:t + k] = err.mean(1).cpu().numpy()
        t += k
    return (mse, emaps) if maps else mse


class Scorer:
    """Score one video with a trained model 1 / 2 (used by evaluate, demo, API)."""
    kind = "mse"

    def __init__(self, cfg, checkpoint, device, **_):
        self.cfg, self.device = cfg, device
        self.model = load_trained(cfg, checkpoint, device)

    def __call__(self, video_dir, vid=None, details=False, num_workers=None, **_):
        if details:
            mse, emaps = score_video(self.model, self.cfg, video_dir, self.device, True, num_workers)
            return mse, {"maps": emaps}
        return score_video(self.model, self.cfg, video_dir, self.device, False, num_workers)
