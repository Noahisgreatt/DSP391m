"""The web page turns an API result into the summary and tables (needs: pip install -r requirements-ui.txt)."""
import pytest

pytest.importorskip("gradio")


def test_summary_and_tables():
    from ui.app import summary, tables
    r = {"video": "clip.mp4", "frames": 50, "fps": 25.0, "model": "mulde", "people_tracked": 2, "timings": {"total_s": 3.0},
         "events": [{"track_id": 2, "event": "COUNTER_FLOW", "reason": "moving against the crowd", "start_s": 0.4, "end_s": 1.2}],
         "people": [{"track_id": 1, "from_s": 0, "to_s": 2, "direction": "right", "moving_share": 1.0, "events": []},
                    {"track_id": 2, "from_s": 0, "to_s": 2, "direction": "left", "moving_share": 0.9, "events": ["COUNTER_FLOW"]}],
         "frame_scores": [0.1] * 25 + [0.9] * 25, "abnormal_frames": list(range(25, 50)), "abnormal_ratio": 0.5,
         "threshold": 0.5, "warning": None}
    s = summary(r)
    assert "1 of 2 people" in s and "Person #2" in s and "Counter-flow 0.4-1.2 s" in s and "50.0%" in s
    events, people, curve = tables(r)
    assert events.iloc[0]["Behavior"] == "Counter-flow"
    assert list(people["Main direction"]) == ["right", "left"] and people.iloc[1]["Abnormal"] == "Counter-flow"
    assert set(curve["line"]) == {"anomaly score", "threshold"} and len(curve) == 52
