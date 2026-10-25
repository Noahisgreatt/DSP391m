"""Model 2: improved ASTT (our upgrade of the thesis model).

Same task and same building blocks as model 1 (ViT-B/32-384 spatial encoder = STE, the repo's
temporal transformer = TTE, the repo's cross-attention, the repo's convolutional decoder), so the
comparison isolates the changes:

  1. Transfer learning   the ViT starts from ImageNet weights (model 1: random init).
  2. Keep WHERE things are   model 1 squeezes each frame into ONE 768-d token, then draws the
                         256x256 frame from one vector (Linear 768 -> 256x16x16, 50M params).
                         Here every frame keeps its 12x12 patch tokens; the TTE runs on each
                         patch position over time (factorised space-time attention), so the
                         decoder starts from a real 12x12 feature map.
  3. Global context      the repo's cross-attention (frame CLS = query, all positions = keys)
                         adds one scene-level vector to every position, like model 1 does
                         (emb[:, 0] + cross_att(emb)).
  4. Predict the change  output = last input frame + predicted change (residual_last_frame),
                         so the model spends its capacity on motion, where anomalies are.
Input: oldest -> newest frames, RGB in [-1, 1]. Output: (b, 3, 256, 256) in [-1, 1].
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.ASTT.model_ste_tte import CrossAttention, Decoder, Encoder


class ASTTImproved(nn.Module):
    def __init__(self, image_size=384, output_size=256, num_frames=4, emb_dim=768, mlp_dim=1536,
                 num_heads=6, num_layers=2, dropout_rate=0.1, attn_dropout_rate=0.0,
                 decoder_channels=256, pretrained=True, residual_last_frame=True):
        super().__init__()
        assert output_size == 256 and decoder_channels == 256, "repo decoder: 256x16x16 -> 256x256"
        self.output_size, self.residual = output_size, residual_last_frame
        # STE: ViT-B/32-384, all tokens kept (1 CLS + 12x12 patches)
        self.spatial_transformer = timm.create_model("vit_base_patch32_384", pretrained=pretrained,
                                                     num_classes=0, img_size=image_size)
        self.grid = image_size // 32
        # TTE: the repo's Encoder, run on every patch position over the T frames
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.temporal_transformer = Encoder(num_patches=num_frames, emb_dim=emb_dim, mlp_dim=mlp_dim,
                                            num_layers=num_layers, num_heads=num_heads,
                                            dropout_rate=dropout_rate,
                                            attn_dropout_rate=attn_dropout_rate)
        self.cross_att = CrossAttention(emb_dim, num_heads=8, qkv_bias=False, qk_scale=None,
                                        attn_drop=0., proj_drop=0.)
        # token map -> the repo decoder's input (256 x 16 x 16)
        self.to_map = nn.Sequential(nn.Conv2d(emb_dim, decoder_channels, 1),
                                    nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=False))
        self.decoder = Decoder().decoder            # the repo's conv stack, minus its Linear layer

    def forward(self, x):                            # x: (b, t, 3, h, w), oldest first
        x = x.float()
        b, t = x.shape[:2]
        tok = self.spatial_transformer.forward_features(x.flatten(0, 1))      # (b*t, 1+g*g, d)
        n, d = tok.shape[1], tok.shape[2]
        seq = tok.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)  # per position
        seq = torch.cat([self.cls_token.expand(b * n, -1, -1), seq], 1)
        tok = self.temporal_transformer(seq)[:, 0].view(b, n, d)              # (b, 1+g*g, d)
        tok = tok + self.cross_att(tok)                                       # global context
        fmap = tok[:, 1:].transpose(1, 2).reshape(b, d, self.grid, self.grid)
        fmap = F.interpolate(fmap, size=(16, 16), mode="bilinear", align_corners=False)
        out = self.decoder(self.to_map(fmap))                                 # (b, 3, 256, 256)
        if self.residual:
            last = F.interpolate(x[:, -1], size=(self.output_size, self.output_size),
                                 mode="bilinear", align_corners=False, antialias=True)
            out = (last + 2 * out).clamp(-1, 1)
        return out


def build(cfg: dict) -> ASTTImproved:
    m = cfg["model"]
    return ASTTImproved(image_size=m["image_size"], output_size=m["output_size"],
                        num_frames=m["num_frames"], emb_dim=m["emb_dim"], mlp_dim=m["mlp_dim"],
                        num_heads=m["num_heads"], num_layers=m["num_layers"],
                        dropout_rate=m["dropout_rate"], attn_dropout_rate=m["attn_dropout_rate"],
                        decoder_channels=m["decoder_channels"], pretrained=m["pretrained"],
                        residual_last_frame=m.get("residual_last_frame", True))


def parameter_groups(model: ASTTImproved) -> dict:
    c = lambda mod: sum(p.numel() for p in mod.parameters())
    return {"STE (ViT-B/32-384, pretrained)": c(model.spatial_transformer),
            "TTE (+ temporal CLS)": c(model.temporal_transformer) + model.cls_token.numel(),
            "Cross-attention": c(model.cross_att),
            "Decoder (token map + conv)": c(model.to_map) + c(model.decoder)}


def backbone_parameters(model: ASTTImproved):
    return model.spatial_transformer.parameters()
