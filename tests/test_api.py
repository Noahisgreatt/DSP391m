"""API with the behavior rules only (no trained model, no downloads): upload -> JSON + metrics."""
import cv2
import numpy as np
from fastapi.testclient import TestClient


class NoPeople:                                          # instead of the Keypoint R-CNN tracker
    def __init__(self, *a, **k):
        pass

    def video(self, frames):
        return {"tracks": np.zeros((0, 7), np.float32), "velocity": np.zeros((0, 2), np.float32),
                "moving": np.zeros(0, bool), "frame_names": np.array([f.name for f in frames])}


def test_api(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 96))
    for t in range(20):
        img = np.full((96, 160, 3), 80, np.uint8)
        cv2.rectangle(img, (5 + 3 * t, 30), (15 + 3 * t, 60), (0, 0, 255), -1)
        w.write(img)
    w.release()
    monkeypatch.setattr("src.inference.PeopleTracker", NoPeople)
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
        for req, status, code in [(dict(files={"file": ("a.txt", b"x", "text/plain")}), 415, "unsupported_file_type"),
                                  (dict(files={"file": ("clip.mp4", b"x", "video/mp4")}, params={"model": "yolo"}), 400, "unknown_model"),
                                  (dict(files={"file": ("clip.mp4", b"", "video/mp4")}), 400, "empty_file"),
                                  (dict(files={"file": ("clip.mp4", b"not a video", "video/mp4")}), 422, "cannot_analyze"),
                                  (dict(), 422, "invalid_request")]:              # no file at all
            e = c.post("/analyze", **req)
            assert e.status_code == status and e.json()["error"] == code and e.json()["detail"], (code, e.text)
        assert c.get("/results/nope/video").json()["error"] == "not_found"
        ms = c.get("/models").json()
        assert ms["default"] == "none" and [m["name"] for m in ms["models"]][-1] == "none"
        assert j["people"] == [] and j["events"] == []
        assert 'vad_videos_analyzed_total{model="none"} 1.0' in c.get("/metrics").text


def test_api_model_not_trained_yet(tmp_path, monkeypatch):
    """A fresh clone (no outputs/) still starts: rules only, and /health says why."""
    monkeypatch.setattr("src.inference.PeopleTracker", NoPeople)
    monkeypatch.setenv("VAD_MODEL", "astt_improved")
    monkeypatch.chdir(tmp_path)                         # no outputs/training/... here
    (tmp_path / "configs").symlink_to(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs")
    import api.main as m
    monkeypatch.setattr(m, "WORK", tmp_path / "api")
    with TestClient(m.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["model"] == "none" and "astt_improved is not ready" in h["warning"]


def test_people_summary():
    """Main walking direction per person, and which behaviors were flagged for them."""
    from src.inference import people_summary
    tracks = np.array([[t, 1, 0, 0, 10, 10, 1] for t in range(10)] + [[t, 2, 0, 0, 10, 10, 1] for t in range(10)], np.float32)
    vel = np.array([[3, 0]] * 10 + [[0, 0]] * 10, np.float32)                 # #1 walks right, #2 stands
    moving = np.array([True] * 10 + [False] * 10)
    flags = np.array([0] * 10 + [1] * 10)                                      # #2 loiters
    p = people_summary(tracks, vel, moving, flags, fps=5)
    assert [(x["track_id"], x["direction"], x["events"]) for x in p] == [(1, "right", []), (2, "standing", ["LOITERING"])]
    assert p[0]["to_s"] == 1.8


def test_openapi_documents_schemas_and_errors(monkeypatch):
    """/docs shows what /analyze returns and every error it can give."""
    monkeypatch.setattr("src.inference.PeopleTracker", NoPeople)
    monkeypatch.setenv("VAD_MODEL", "none")
    import api.main as m
    with TestClient(m.app) as c:
        spec = c.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert {"AnalyzeResult", "Event", "Person", "ModelList", "ModelCard", "Health", "ErrorResponse"} <= set(schemas)
    responses = spec["paths"]["/analyze"]["post"]["responses"]
    assert {"200", "400", "413", "415", "422", "500"} <= set(responses)
    assert responses["415"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")
    assert schemas["Event"]["properties"]["event"]["enum"] == ["LOITERING", "COUNTER_FLOW", "TOO_FAST"]
