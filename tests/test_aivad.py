"""Models 3/4 building blocks: FlowNet2 replacement ops, velocity histogram, exact 1-NN,
and the density scorer ranking an unusual object above normal ones."""
import math

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter

from src.models.aivad.aivad_code import velocity_histogram
from src.models.aivad.density import AttributeScorer, nn_distance
from src.models.aivad.features import normalise_pose
from src.models.aivad.flownet2_ops import correlation_forward, resample2d_forward


def test_correlation_matches_cuda_kernel_definition():
    torch.manual_seed(0)
    a, b = torch.randn(1, 3, 7, 9), torch.randn(1, 3, 7, 9)
    out = torch.empty(0)
    correlation_forward(a, b, None, None, out, 4, 1, 4, 1, 2, 1)
    r, ref = 2, torch.zeros(1, 25, 7, 9)
    for y in range(7):
        for x in range(9):
            for tj in range(-r, r + 1):
                for ti in range(-r, r + 1):
                    yy, xx = y + tj * 2, x + ti * 2
                    if 0 <= yy < 7 and 0 <= xx < 9:
                        ref[0, (tj + r) * 5 + ti + r, y, x] = (a[0, :, y, x] * b[0, :, yy, xx]).sum() / 3
    assert torch.allclose(out, ref, atol=1e-6)


def test_resample_matches_cuda_kernel_definition():
    torch.manual_seed(1)
    img, flow = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8) * 3
    out = torch.zeros_like(img)
    resample2d_forward(img, flow, out, 1, True)
    cl = lambda v, m: max(min(v, m - 1), 0)
    for y in range(6):
        for x in range(8):
            xf, yf = x + float(flow[0, 0, y, x]), y + float(flow[0, 1, y, x])
            al, be = xf - math.floor(xf), yf - math.floor(yf)
            xl, xr, yt, yb = cl(math.floor(xf), 8), cl(math.floor(xf) + 1, 8), cl(math.floor(yf), 6), cl(math.floor(yf) + 1, 6)
            ref = ((1 - al) * (1 - be) * img[0, :, yt, xl] + al * (1 - be) * img[0, :, yt, xr]
                   + (1 - al) * be * img[0, :, yb, xl] + al * be * img[0, :, yb, xr])
            assert torch.allclose(out[0, :, y, x], ref, atol=1e-5)


def test_velocity_histogram_equals_aivad_uniform_filter():
    f = np.random.default_rng(0).normal(size=(224, 224, 2)).astype(np.float32) * 3
    _, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
    mag = np.abs(f[..., 0]) + np.abs(f[..., 1])
    o, ref = ang * (180 / np.pi), np.zeros(8)
    for i in range(8):
        t = np.where(o < 45 * (i + 1), o, -1)
        t = np.where(o >= 45 * i, t, -1)
        ref[i] = uniform_filter(np.where(t > -1, mag, 0), size=(224, 224))[112, 112]
    assert np.allclose(velocity_histogram(f, mag, ang.copy()), ref, rtol=1e-5)


def test_nearest_neighbour_and_leave_one_video_out():
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(600, 34)).astype(np.float32)
    q = rng.normal(size=(50, 34)).astype(np.float32)
    ref = np.array([((bank - x) ** 2).sum(1).min() for x in q])
    assert np.allclose(nn_distance(q, bank, b_chunk=97), ref, rtol=1e-4)
    g = np.repeat(np.arange(6), 100)
    lovo = nn_distance(bank, bank, g, g, b_chunk=97)
    ref = np.array([((bank[g != g[i]] - bank[i]) ** 2).sum(1).min() for i in range(0, 600, 60)])
    assert np.allclose(lovo[::60], ref, rtol=1e-4)


def fake_video(rng, n=40, fast_frame=None):
    rows = 3
    vel = np.abs(rng.normal(0.2, 0.05, size=(n * rows, 8))).astype(np.float32)
    if fast_frame is not None:
        vel[fast_frame * rows] = 3.0                         # one object moves far too fast
    pose = rng.normal(20, 2, size=(n, 34)).astype(np.float32)
    return {"n_frames": np.array(n), "frame": np.repeat(np.arange(n), rows).astype(np.int32),
            "velocity": vel, "mean_flow": np.zeros((n * rows, 2), np.float32),
            "ids": -np.ones(n * rows, np.int64), "pose_frame": np.arange(n, dtype=np.int32),
            "pose": pose, "pose_object": np.arange(n) * rows}


def test_density_scorer_finds_the_unusual_object():
    rng = np.random.default_rng(0)
    cfg = {"scoring": {"attributes": ["velocity", "pose"], "gmm_components": 2,
                       "calibration_percentile": 99.9, "smoothing_sigma": 0}}
    sc = AttributeScorer(cfg)
    sc.fit({f"01_00{i}": fake_video(rng) for i in range(4)})
    r = sc.score("01_0014", fake_video(rng, fast_frame=20))
    assert int(np.argmax(r["score"])) == 20


def test_pose_normalisation():
    kp = np.zeros((1, 17, 3), np.float32)
    kp[0, :, 0], kp[0, :, 1] = 110, 260                      # all at the box's bottom-right corner
    p = normalise_pose(kp, np.array([[10, 60, 110, 260]], np.float32))
    assert np.allclose(p[0, 0::2], 48) and np.allclose(p[0, 1::2], 112)


def test_unusual_objects_report():
    from src.inference.analyze import Analyzer
    f = {"frame": np.array([0, 0, 1]), "boxes": np.array([[0, 0, 10, 20], [5, 5, 9, 9], [1, 1, 4, 4]], np.float32),
         "whole": np.array([False, False, False]), "pose_object": np.array([0])}
    objs = {"velocity": np.array([0.2, 1.5, 0.1]), "pose": np.array([3.0])}
    u = Analyzer.unusual_objects({"features": f, "objects": objs})
    assert [(o["frame"], o["attribute"]) for o in u] == [(0, "pose"), (0, "velocity")]
