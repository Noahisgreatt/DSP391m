"""MULDE (models 3, 4): the density network learns where normal data is; features per frame."""
import numpy as np
import torch

from src.models import mulde
from src.utils import load_config


def test_density_network_learns_where_the_data_is():
    torch.manual_seed(0)
    X = np.random.RandomState(0).randn(4000, 4).astype(np.float32) * 0.3        # "normal" data around 0
    d = {"lr": 1e-3, "adam_betas": [0.5, 0.9], "sigma_low": 1e-3, "sigma_high": 1.0, "batch_size": 256, "beta": 0.1}
    net = mulde.train_density(X, d, 64, torch.device("cpu"), 600)
    F = mulde.levels(net, np.array([[0, 0, 0, 0], [3, 3, 3, 3]], np.float32), np.geomspace(1e-3, 1, 16), torch.device("cpu"))
    assert (F[1] > F[0]).mean() > 0.8                     # f is higher far from the data, at most noise levels
    gmm = mulde.fit_mixture(mulde.levels(net, X[:500], np.geomspace(1e-3, 1, 16), torch.device("cpu")), 1)
    assert -gmm.score_samples(F[1:]) > -gmm.score_samples(F[:1])          # the far point is more abnormal


def test_one_feature_per_frame(tmp_path, monkeypatch):
    import cv2
    monkeypatch.setenv("VAD_NO_WEIGHTS", "1")
    for t in range(5):
        cv2.imwrite(str(tmp_path / f"{t:03d}.jpg"), np.full((60, 80, 3), 40 * t, np.uint8))
    cfg = load_config("configs/mulde_improved.yaml")
    cfg["model"]["batch_clips"] = 5
    centres, X = mulde.Features(cfg, torch.device("cpu"), weights=False).video(tmp_path, "v")
    assert list(centres) == [0, 1, 2, 3, 4] and X.shape == (5, 768 * 2)       # two clip lengths side by side
