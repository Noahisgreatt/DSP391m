import numpy as np

from src.evaluate import binary_metrics, frame_metrics, macro_auc, psnr_norm


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 0])
    s = np.array([np.nan, .01, .01, .5, .6, .02])        # NaN = frame without a prediction
    res, per_video = frame_metrics({"v": s}, {"v": y})
    assert res["auc_raw"] == res["auc_norm"] == res["auc_macro"] == 1.0 and res["eer"] == 0.0
    assert per_video["v"][0][0] == np.nanmin(s)          # NaN filled with the video minimum


def test_psnr_norm_direction():
    n = psnr_norm(np.array([0.001, 0.01, 0.1]))
    assert n.min() == 0 and n.max() == 1 and n[2] > n[0]  # larger error = more abnormal


def test_macro_auc_all_normal_video():
    assert 0 <= macro_auc([(np.array([.1, .2]), np.array([0, 0]))]) <= 1


def test_binary_metrics():
    m = binary_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0], bool))
    assert m["tpr"] == 0.5 and m["fpr"] == 0.0
