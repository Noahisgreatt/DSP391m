"""Fake ShanghaiTech zips -> prepare -> split: the official split is rebuilt from the names,
pixel masks are skipped, the data card is written."""
import json
import shutil

from tests import fake_data


def test_prepare_and_split(tmp_path, monkeypatch):
    shutil.copytree("configs", tmp_path / "configs")
    fake_data.make(tmp_path / "data/raw/shanghaitech", n_frames=12)
    monkeypatch.chdir(tmp_path)
    from src.data import make_split, prepare_shanghaitech
    from src.utils import load_config
    cfg = load_config()
    prepare_shanghaitech(cfg)
    card = json.loads((tmp_path / "data/metadata/shanghaitech.json").read_text())
    assert card["videos"] == 5 and card["problems"] == []
    s = make_split(cfg)
    assert s["train"] == fake_data.TRAIN and s["test"] == fake_data.TEST
    assert len(list((tmp_path / "data/processed/shanghaitech/labels").glob("*.npy"))) == 5
