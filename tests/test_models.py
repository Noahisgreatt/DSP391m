"""Model 1 must stay the thesis model exactly; model 2 must stay >= 35M parameters."""
import pytest
import torch

from src.config import load_config
from src.models import build_model, count_parameters


def build(name):
    cfg = load_config(f"configs/{name}.yaml")
    cfg["model"]["pretrained"] = False           # no download in tests
    return cfg, build_model(cfg).eval()


def test_baseline_is_the_thesis_model():
    cfg, m = build("astt_baseline")
    p = count_parameters(cfg, m)
    assert p["Total"] == 150_579_876
    assert p["STE (ViT-B/32-384)"] == 87_528_192 and p["Decoder"] == 51_230_627


@pytest.mark.parametrize("name", ["astt_baseline", "astt_improved"])
def test_forward_shape_and_size(name):
    cfg, m = build(name)
    assert count_parameters(cfg, m)["Total"] >= 35_000_000
    with torch.no_grad():
        y = m(torch.rand(1, 4, 3, 384, 384) * 2 - 1)
    assert y.shape == (1, 3, 256, 256) and y.abs().max() <= 1.0 + 1e-5
