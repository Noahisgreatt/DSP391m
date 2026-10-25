"""Models 3/4 scoring: AI-VAD's velocity histogram, exact 1-NN, pose normalisation, and the
density model ranking an unusual object above normal ones."""
import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from src.models.aivad import Density, nn_distance, normalise_pose, velocity_histogram


def test_velocity_histogram_equals_aivad():
    f = np.random.default_rng(0).normal(size=(224, 224, 2)).astype(np.float32) * 3
    _, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
    mag = np.abs(f[..., 0]) + np.abs(f[..., 1])
    o, ref = ang * (180 / np.pi), np.zeros(8)
    for i in range(8):                                       # AI-VAD's exact code path
        t = np.where(o < 45 * (i + 1), o, -1)
        t = np.where(o >= 45 * i, t, -1)
        ref[i] = uniform_filter(np.where(t > -1, mag, 0), size=(224, 224))[112, 112]
    assert np.allclose(velocity_histogram(f, mag, ang.copy()), ref, rtol=1e-5)


def test_nearest_neighbour_and_leave_one_video_out():
    rng = np.random.default_rng(0)
    bank, q = rng.normal(size=(600, 34)).astype(np.float32), rng.normal(size=(50, 34)).astype(np.float32)
    assert np.allclose(nn_distance(q, bank, b_chunk=97), [((bank - x) ** 2).sum(1).min() for x in q], rtol=1e-4)
    g = np.repeat(np.arange(6), 100)
    lovo = nn_distance(bank, bank, g, g, b_chunk=97)
    assert np.allclose(lovo[::60], [((bank[g != g[i]] - bank[i]) ** 2).sum(1).min() for i in range(0, 600, 60)], rtol=1e-4)


def fake_video(rng, n=40, fast_frame=None):
    vel = np.abs(rng.normal(0.2, 0.05, size=(n * 3, 8))).astype(np.float32)
    if fast_frame is not None:
        vel[fast_frame * 3] = 3.0                            # one object far too fast
    return {"n_frames": np.array(n), "frame": np.repeat(np.arange(n), 3).astype(np.int32), "velocity": vel,
            "mean_flow": np.zeros((n * 3, 2), np.float32), "ids": -np.ones(n * 3, np.int64),
            "pose_frame": np.arange(n, dtype=np.int32), "pose": rng.normal(20, 2, size=(n, 34)).astype(np.float32),
            "pose_object": np.arange(n) * 3}


def test_density_finds_the_unusual_object():
    rng = np.random.default_rng(0)
    d = Density({"scoring": {"attributes": ["velocity", "pose"], "gmm_components": 2,
                             "calibration_percentile": 99.9, "smoothing_sigma": 0}})
    d.fit({f"01_00{i}": fake_video(rng) for i in range(4)})
    assert int(np.argmax(d.score("01_0014", fake_video(rng, fast_frame=20))["score"])) == 20


def test_pose_normalisation():
    kp = np.zeros((1, 17, 3), np.float32)
    kp[0, :, 0], kp[0, :, 1] = 110, 260                      # bottom-right corner of the box
    p = normalise_pose(kp, np.array([[10, 60, 110, 260]], np.float32))
    assert np.allclose(p[0, 0::2], 48) and np.allclose(p[0, 1::2], 112)


def test_unusual_objects_report():
    from src.demo import unusual_objects
    f = {"frame": np.array([0, 0, 1]), "boxes": np.array([[0, 0, 10, 20], [5, 5, 9, 9], [1, 1, 4, 4]], np.float32),
         "whole": np.array([False, False, False]), "pose_object": np.array([0])}
    u = unusual_objects({"features": f, "objects": {"velocity": np.array([0.2, 1.5, 0.1]), "pose": np.array([3.0])}})
    assert [(o["frame"], o["attribute"]) for o in u] == [(0, "pose"), (0, "velocity")]
