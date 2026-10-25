"""The thesis model (STE + TTE + temporal cross-attention), built ONLY from the original repo code.

Source: https://github.com/Tungufm/ASTT, commit 1d05305 (copied unchanged to models/thesis_astt/original_repo).
Thesis: Tran Minh Tung, Chapter 3, eq. 3.6-3.12; paper: IEEE TCSVT 34 (2024) 8292-8309.

Why this file exists: the repo's final model (`model.py`, imported by train_val_UIT_ADrone.py)
was never uploaded. The repo has its two halves:
  model_ste_tte.py      VisionTransformer = STE + TTE + Decoder, with the cross-attention line
                        commented out (line 351)
  model_baseline_ca.py  shows how cross-attention is applied:
                        feat = feat[:,0,:].unsqueeze(1) + self.cross_att(feat)     (line 386)
This class is model_ste_tte.VisionTransformer with that commented line switched on and that one
forward line applied after the temporal encoder. Every layer, size, initialisation and setting is
the repo's: ViT-B/32-384 created with pretrained=False, frames concatenated newest-first,
the unused Deep-SVDD "sphere" parameters kept.

The only edit: the repo's forward starts with `x = x.float().cuda()`; here it is
`x.float().to(<model device>)`, which is identical on a GPU and also allows CPU testing.
"""
from __future__ import annotations

import torch

from models.thesis_astt.original_repo.model_ste_tte import CrossAttention
from models.thesis_astt.original_repo.model_ste_tte import VisionTransformer as RepoSTETTE


class ASTTThesis(RepoSTETTE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # model_ste_tte.py line 351 (commented out there), as used in model_baseline_ca.py line 358
        self.cross_att = CrossAttention(768, num_heads=8, qkv_bias=False, qk_scale=None,
                                        attn_drop=0., proj_drop=0.)

    def forward(self, x):  # x: (b, n, c, h, w)
        x = x.float().to(self.cls_token.device)          # repo: x.float().cuda()
        for i in range(x.shape[1]):                       # --- model_ste_tte.py lines 355-366
            spatial_cls_token = self.spatial_transformer(x[:, i]).unsqueeze(1)
            if i == 0:
                all_spatial_cls_token = spatial_cls_token
            else:
                all_spatial_cls_token = torch.cat((spatial_cls_token, all_spatial_cls_token), 1)

        b, n, c = all_spatial_cls_token.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, all_spatial_cls_token], dim=1)
        emb = self.temporal_transformer(emb)
        emb = emb[:, 0, :].unsqueeze(1) + self.cross_att(emb)   # model_baseline_ca.py line 386
        output = self.decoder(emb)
        return output


def build_thesis_model(cfg: dict) -> ASTTThesis:
    """Arguments exactly as the repo's train script passes them (config.py get_b16_config)."""
    m, d = cfg["model"], cfg["data"]
    return ASTTThesis(image_size=(d["image_size"], d["image_size"]),
                      patch_size=(m["patch_size"], m["patch_size"]), emb_dim=m["emb_dim"],
                      mlp_dim=m["mlp_dim"], num_heads=m["num_heads"], num_layers=m["num_layers"],
                      num_classes=m["num_classes"], attn_dropout_rate=m["attn_dropout_rate"],
                      dropout_rate=m["dropout_rate"], num_frames=d["num_frames"])


def count_parameters(model: ASTTThesis) -> dict:
    c = lambda mod: sum(p.numel() for p in mod.parameters())
    out = {"STE (ViT-B/32-384)": c(model.spatial_transformer),
           "TTE (+ temporal CLS)": c(model.temporal_transformer) + model.cls_token.numel(),
           "Cross-attention": c(model.cross_att), "Decoder": c(model.decoder),
           "Sphere (unused, kept from repo)": model.center.numel() + model.radius.numel()}
    out["Total"] = c(model)
    return out
