"""Model registry: the config's model.name picks the model.

    model = build_model(cfg)            # cfg["model"]["name"] = astt_baseline | astt_improved
"""
from __future__ import annotations

import importlib

MODELS = ("astt_baseline", "astt_improved")


def _module(name: str):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
    return importlib.import_module(f"src.models.{name}")


def build_model(cfg: dict):
    return _module(cfg["model"]["name"]).build(cfg)


def count_parameters(cfg: dict, model) -> dict:
    out = _module(cfg["model"]["name"]).parameter_groups(model)
    out["Total"] = sum(p.numel() for p in model.parameters())
    out["Trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return out


def backbone_parameters(cfg: dict, model):
    return _module(cfg["model"]["name"]).backbone_parameters(model)


def load_trained(cfg: dict, checkpoint, device):
    """Model with trained weights (model.pth or a training checkpoint), in eval mode."""
    import torch
    if cfg["model"].get("pretrained"):
        cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}   # weights come from the file
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    return model.eval()
