"""Model 1 must stay the thesis model exactly; models 1-4 must have >= 35M parameters."""
import pytest
import torch

from src.models.astt import build, count_parameters
from src.utils import load_config


def test_model1_is_the_thesis_model():
    p = count_parameters(build(load_config("configs/astt_baseline.yaml")))
    assert p["Total"] == 150_579_876
    assert p["STE (ViT-B/32-384)"] == 87_528_192 and p["Decoder"] == 51_230_627


@pytest.mark.parametrize("name", ["astt_baseline", "astt_improved"])
def test_forward_shape(name):
    m = build(load_config(f"configs/{name}.yaml"), pretrained=False).eval()
    assert count_parameters(m)["Total"] >= 35_000_000
    with torch.no_grad():
        y = m(torch.rand(1, 4, 3, 384, 384) * 2 - 1)
    assert y.shape == (1, 3, 256, 256) and y.abs().max() <= 1 + 1e-5


def test_mulde_models_size():
    from src.models.mulde import parameters
    m3, m4 = (parameters(load_config(f"configs/{n}.yaml"))["Total"] for n in ("mulde", "mulde_improved"))
    assert m3 == 70_724_289 and m4 == 58_133_185                    # Hiera-B 50,784,960 + density network
    assert 35_000_000 <= m4 < m3
