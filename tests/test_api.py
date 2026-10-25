"""API with the behavior rules only (no trained model needed): upload -> JSON + metrics."""

import cv2
import numpy as np
from fastapi.testclient import TestClient


def test_api(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 96))
    for t in range(20):
        img = np.full((96, 160, 3), 80, np.uint8)
        cv2.rectangle(img, (5 + 3 * t, 30), (15 + 3 * t, 60), (0, 0, 255), -1)
        w.write(img)
    w.release()

    def no_yolo(cfg, frames, out):                      # tracking without downloading YOLO
        np.savez(out, tracks=np.zeros((0, 7), np.float32), velocity=np.zeros((0, 2), np.float32),
                 moving=np.zeros(0, bool), frame_names=np.array([f.name for f in frames]))
        return np.zeros((0, 7), np.float32)

    monkeypatch.setattr("src.inference.analyze.track_and_save", no_yolo)
    monkeypatch.setenv("VAD_MODEL", "none")
    import api.main as m
    monkeypatch.setattr(m, "WORK", tmp_path / "api")
    with TestClient(m.app) as c:
        assert c.get("/health").json()["model"] == "none"
        r = c.post("/analyze", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["frames"] == 20 and j["frame_scores"] is None
        assert c.get(j["result_video"]).status_code == 200
        assert c.post("/analyze", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 400
        metrics = c.get("/metrics").text
        assert 'vad_videos_analyzed_total{model="none"} 1.0' in metrics
