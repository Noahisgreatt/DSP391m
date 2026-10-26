"""Versions: what changed between two metadata cards (src/release.py)."""
from src.release import changes

CARD = {"data": {"shanghaitech": {"md5": "a"}}, "code_md5": "c", "settings": {"astt_baseline": {"train": {"lr": 1e-4}}},
        "models": {"astt_baseline": {"parameters": 150579876, "weights_md5": "w", "test": {"auc_raw": 0.61}}}}


def test_changes():
    assert changes(CARD, CARD) == []
    new = {**CARD, "settings": {"astt_baseline": {"train": {"lr": 2e-4}}},
           "models": {"astt_baseline": {"parameters": 150579876, "weights_md5": "w2", "test": {"auc_raw": 0.63}},
                      "astt_improved": {"parameters": 100379299, "weights_md5": "x", "test": {"auc_raw": 0.68}}}}
    out = changes(CARD, new)
    assert "setting astt_baseline.train.lr: 0.0001 -> 0.0002" in out
    assert "astt_baseline: retrained (new weights)" in out
    assert "astt_baseline: auc_raw 61.0% -> 63.0% (+2.0)" in out
    assert any(o.startswith("astt_improved: ADDED") for o in out)
