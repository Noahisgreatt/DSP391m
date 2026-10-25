"""Fake ShanghaiTech zips -> prepare -> split: the official split is rebuilt from the names,
pixel masks are skipped, and the data card is written."""
import json
import shutil
import sys

from tests import fake_data


def test_prepare_and_split(tmp_path, monkeypatch):
    shutil.copytree("configs", tmp_path / "configs")
    fake_data.make(tmp_path / "data/raw/shanghaitech", n_frames=12)
    monkeypatch.chdir(tmp_path)
    from src.data import prepare_shanghaitech, split
    monkeypatch.setattr(sys, "argv", ["x"])
    prepare_shanghaitech.main()
    split.main()
    card = json.loads((tmp_path / "data/metadata/shanghaitech.json").read_text())
    assert card["videos"] == 5 and card["problems"] == []
    assert card["official_split"]["train"]["videos"] == 3 and card["official_split"]["test"]["videos"] == 2
    s = json.loads((tmp_path / "data/metadata/split.json").read_text())
    assert s["train"] == fake_data.TRAIN and s["test"] == fake_data.TEST
    assert len(list((tmp_path / "data/processed/shanghaitech/labels").glob("*.npy"))) == 5
