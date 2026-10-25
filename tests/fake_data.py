"""Tiny synthetic ShanghaiTech-like zips, for testing the pipeline without the real 20 GB.

    python -m tests.fake_data /tmp/fake_raw        (never writes into data/raw by default)

Layout copies the Kaggle parts: SHANGHAI/<part>/frames/<video>/<n>.jpg + label/<video>.npy,
with train- and test-named videos mixed across the two zips (like the weakly-supervised copy)
and a pixel mask that must be skipped. Test videos have a fast bright square = "anomaly".
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

TRAIN = ["01_001", "01_002", "02_001"]
TEST = ["01_0014", "02_0015"]


def frame(t: int, abnormal: bool, w=160, h=96) -> bytes:
    img = np.full((h, w, 3), 90, np.uint8)
    x = int(10 + 2 * t) % (w - 20)                       # slow walker
    cv2.rectangle(img, (x, 40), (x + 8, 70), (30, 30, 200), -1)
    if abnormal:                                         # fast object
        x2 = int(15 * t) % (w - 20)
        cv2.rectangle(img, (x2, 10), (x2 + 16, 30), (250, 250, 250), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def make(raw: Path, n_frames: int = 30) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    parts = {"train": ("SHANGHAI_TRAIN", TRAIN[:2] + TEST[:1]), "test": ("SHANGHAI_Test", TRAIN[2:] + TEST[1:])}
    for part, (top, vids) in parts.items():
        with zipfile.ZipFile(raw / f"{part}.zip", "w") as z:
            for v in vids:
                test = v in TEST
                lab = np.zeros(n_frames, np.int64)
                if test:
                    lab[n_frames // 2: n_frames // 2 + 8] = 1
                for t in range(n_frames):
                    z.writestr(f"SHANGHAI/{top}/frames/{v}/{t:03d}.jpg", frame(t, bool(lab[t])))
                buf = io.BytesIO(); np.save(buf, lab)
                z.writestr(f"SHANGHAI/{top}/label/{v}.npy", buf.getvalue())
                if test:
                    buf = io.BytesIO(); np.save(buf, np.zeros((n_frames, 4, 4), np.uint8))
                    z.writestr(f"SHANGHAI/{top}/test_pixel_mask/{v}.npy", buf.getvalue())


if __name__ == "__main__":
    make(Path(sys.argv[1]))
    print("wrote", sys.argv[1])
