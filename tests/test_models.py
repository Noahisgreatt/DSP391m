"""Model 1 must stay the thesis model exactly; every model must have >= 35M parameters."""
import pytest
import torch

from src.models.astt import build, count_parameters
from src.utils import load_config


def test_model1_is_the_thesis_model():
    p = count_parameters(build(load_config("configs/astt_baseline.yaml")))
    assert p["Total"] == 150_579_876
    assert p["STE (ViT-B/32-384)"] == 87_528_192 and p["Decoder"] == 51_230_627


@pytest.mark.parametrize("name", ["astt_baseline", "astt_improved", "astt_lite"])
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


def test_model5_is_model2_with_7_vit_blocks():
    p5 = count_parameters(build(load_config("configs/astt_lite.yaml"), pretrained=False))
    assert p5["Total"] == 64_939_939 and 60_000_000 <= p5["Total"] <= 70_000_000
    assert p5["STE (ViT-B/32-384), first 7 blocks"] == 52_088_832


def test_shortened_vit_loads_the_first_blocks_of_the_pretrained_one(tmp_path):
    """Model 5's ViT gets exactly the pretrained weights of blocks 0-6 (8-11 are dropped)."""
    from safetensors.torch import save_file
    from src.models.nets import ViT
    full = ViT()
    save_file({k: v.contiguous() for k, v in full.state_dict().items()}, str(tmp_path / "vit.safetensors"))
    short = ViT(depth=7).load_pretrained(tmp_path / "vit.safetensors")
    assert len(short.blocks) == 7
    assert torch.equal(short.blocks[6].mlp.fc2.weight, full.blocks[6].mlp.fc2.weight)
    assert torch.equal(short.patch_embed.proj.weight, full.patch_embed.proj.weight)


def test_model6_uses_model5_weights_and_smooths():
    """Model 6 = model 5's network (same training folder) + Gaussian smoothing of the score."""
    import numpy as np
    from src.models import trained_dir
    from src.models.astt import smooth
    from src.utils import out_dir
    c5, c6 = load_config("configs/astt_lite.yaml"), load_config("configs/astt_lite_smooth.yaml")
    assert out_dir(c6, "training") == out_dir(c5, "training") and trained_dir("astt_lite_smooth") == trained_dir("astt_lite")
    assert out_dir(c6, "evaluation") != out_dir(c5, "evaluation")
    assert {k: v for k, v in c6["model"].items() if k not in ("name", "weights_from")} == \
           {k: v for k, v in c5["model"].items() if k != "name"}
    s = np.r_[[np.nan] * 4, np.zeros(20), 1.0, np.zeros(20)]                # one noisy frame
    out = smooth(s, c6["scoring"]["smoothing_sigma"])
    assert np.isnan(out[:4]).all() and out[24] < 0.1 and 0.95 < np.nansum(out) <= 1.0     # spread, not lost


@pytest.mark.parametrize("name", ["astt_improved", "astt_lite"])
def test_cached_scoring_gives_the_same_scores(tmp_path, name):
    """astt_fast (API / app: each frame through the ViT once) == astt.score_video (evaluation)."""
    import cv2
    import numpy as np
    from src.models.astt import score_video
    from src.models.astt_fast import score_video_cached
    rng = np.random.default_rng(0)
    for i in range(11):
        cv2.imwrite(str(tmp_path / f"{i:06d}.jpg"), rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    cfg = load_config(f"configs/{name}.yaml")
    cfg["eval"]["batch_size"] = 3                         # several batches: the cache must carry over
    torch.manual_seed(0)
    model = build(cfg, pretrained=False).eval()
    ref, ref_maps = score_video(model, cfg, tmp_path, torch.device("cpu"), maps=True, num_workers=0)
    new, new_maps = score_video_cached(model, cfg, tmp_path, torch.device("cpu"), maps=True)
    assert np.isnan(new[:4]).all() and np.isnan(ref[:4]).all()
    np.testing.assert_allclose(new[4:], ref[4:], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(new_maps.astype(np.float32), ref_maps.astype(np.float32), atol=2e-3)
