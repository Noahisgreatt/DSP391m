"""The web page turns an API result into the summary and tables (needs: pip install -r requirements-ui.txt)."""
import pytest

pytest.importorskip("gradio")


def test_summary_and_tables():
    from app.gradio_app import summary, tables
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


def test_model_list_has_no_rules_only_entry(monkeypatch):
    """The dropdown offers the four models only (the rules run with every model anyway)."""
    import app.gradio_app as app

    class R:
        def json(self):
            return {"default": "astt_improved", "models": [
                {"name": "astt_improved", "title": "Model 2", "ready": True, "parameters": 100379299, "auc_macro": 0.81},
                {"name": "mulde", "title": "Model 3", "ready": False},
                {"name": "none", "title": "Behavior rules only", "ready": True}]}
    monkeypatch.setattr(app.requests, "get", lambda *a, **k: R())
    choices, default = app.models()
    assert [v for _, v in choices] == ["astt_improved", "mulde"] and default == "astt_improved"
    assert "81.0%" in choices[0][0] and "not trained yet" in choices[1][0]
