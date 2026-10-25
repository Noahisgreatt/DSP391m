"""The rules on hand-made tracks: a crowd walking right, one person walking left, one running,
one standing still for 12 s. And the tracker keeps ids across frames."""
import numpy as np

from src.behavior import COUNTER, FAST, LOITER, IoUTracker, detect_events, track_velocities
from src.utils import load_config


def make_tracks(fps=25, seconds=12):
    rows = []
    for t in range(fps * seconds):
        for k in range(5):                                   # crowd: ids 1-5 walk right
            x = 50 + k * 30 + 1.0 * t
            rows.append([t, k + 1, x, 100, x + 20, 160, .9])
        x = 500 - 1.0 * t                                    # id 6 walks LEFT through them
        rows.append([t, 6, x, 110, x + 20, 170, .9])
        x = 60 + 5.0 * t                                     # id 7 runs right
        rows.append([t, 7, x, 120, x + 20, 180, .9])
        rows.append([t, 8, 300, 250, 320, 310, .9])          # id 8 stands still
    return np.array(rows, np.float32)


def test_rules():
    cfg = load_config("configs/behavior.yaml")
    tr = make_tracks()
    vel, moving = track_velocities(tr, 8, 5, 0.4)
    R, C = cfg["optical_flow"]["grid"]
    flags = detect_events(cfg, tr, vel, moving, np.zeros((R, C, 2), np.float32), np.zeros((R, C), np.float32), 25.0)
    who = lambda bit: set(tr[(flags & bit) > 0, 1].astype(int))
    assert 6 in who(COUNTER) and 7 in who(FAST) and 8 in who(LOITER)
    assert not ({1, 2, 3, 4, 5} & (who(COUNTER) | who(FAST)))


def test_tracker_keeps_ids():
    tr, seen = IoUTracker(), []
    for t in range(10):
        boxes = np.array([[10 + 3 * t, 10, 40 + 3 * t, 80], [200 - 2 * t, 50, 230 - 2 * t, 120]], np.float32)
        seen.append(tuple(tr.update(boxes, np.array([.9, .9]))))
    assert len(set(seen)) == 1 and -1 not in seen[0]
