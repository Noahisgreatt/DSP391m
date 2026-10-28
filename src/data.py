"""Step 1, DATA: get it, unpack it, split it, and feed it to the models.

    python -m src.data download                       # the 2 zips from Kaggle (needs a Kaggle API token)
    python -m src.data prepare                        # unzip ShanghaiTech + data card   (DVC: prepare_shanghaitech)
    python -m src.data manual                         # cut our phone videos into frames (DVC: prepare_manual)
    python -m src.data split                          # official train / test split      (DVC: split)
    python -m src.data link DIR [DIR ...]             # use already-unpacked folders instead (Kaggle)

Files:
  data/raw/shanghaitech/{train,test}.zip   Kaggle parts (238 + 199 videos as frames), DVC-tracked
  data/raw/manual/                         our phone videos, DVC-tracked
  data/processed/<dataset>/frames/<video>/ frames, one folder per video
  data/processed/shanghaitech/labels/      0/1 per frame for the test videos
  data/metadata/shanghaitech.json, manual.json   data cards (what the data is, counts, checks)
  data/metadata/split.json                 which videos train / test
Official ShanghaiTech Campus split, from the names: XX_YYY = 330 normal training videos,
XX_YYYY = 107 test videos (the Kaggle parts mix them: 238 / 199).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
import torch.utils.data as tud
import yaml

from src.utils import IMAGE_EXTS, frames_dir, labels_dir, list_frames, load_config

TRAIN_RE, TEST_RE = re.compile(r"\d{2}_\d{3}"), re.compile(r"\d{2}_\d{4}")


# ====================================================================== download / pack
def download(cfg):
    out = Path(cfg["paths"]["raw_shanghaitech"])
    for part in ("train", "test"):
        if not (out / f"{part}.zip").exists():
            tmp = out / f"_kaggle_{part}"
            subprocess.run(["kaggle", "datasets", "download", "-d", cfg["shanghaitech"][f"kaggle_{part}"], "-p", str(tmp)], check=True)
            next(tmp.glob("*.zip")).replace(out / f"{part}.zip")
            tmp.rmdir()


# ====================================================================== prepare ShanghaiTech
def link_shanghaitech(cfg, folders):
    """Already-unpacked copies (e.g. Kaggle's /kaggle/input/...): link them instead of unzipping."""
    frames, labels = frames_dir(cfg, "shanghaitech"), labels_dir(cfg)
    frames.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
    for root in folders:
        for d, dirs, files in os.walk(root):
            if "pixel" in d.lower():
                continue
            name = Path(d).name
            if (TRAIN_RE.fullmatch(name) or TEST_RE.fullmatch(name)) and any(Path(f).suffix.lower() in IMAGE_EXTS for f in files):
                if not (frames / name).exists():
                    os.symlink(d, frames / name)
                dirs[:] = []
            for f in files:
                if f.endswith(".npy") and not (labels / f).exists():
                    os.symlink(os.path.join(d, f), labels / f)
    data_card(cfg, {"linked_from": [str(f) for f in folders]})


def prepare_shanghaitech(cfg):
    raw, frames, labels = Path(cfg["paths"]["raw_shanghaitech"]), frames_dir(cfg, "shanghaitech"), labels_dir(cfg)
    labels.mkdir(parents=True, exist_ok=True)
    parts = {}
    for part in ("train", "test"):
        zp = raw / f"{part}.zip"
        if not zp.exists():
            raise SystemExit(f"{zp} missing: run `dvc pull` or `python -m src.data download`")
        vids, n_lab = set(), 0
        with zipfile.ZipFile(zp) as z:
            members = [m for m in z.infolist() if not m.is_dir() and "pixel" not in m.filename.lower()]
            for k, m in enumerate(members, 1):
                p = PurePosixPath(m.filename)
                if p.suffix.lower() in IMAGE_EXTS:
                    dst = frames / p.parent.name / p.name
                    vids.add(p.parent.name)
                elif p.suffix == ".npy":
                    dst, n_lab = labels / p.name, n_lab + 1
                else:
                    continue
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(dst, "wb") as f:
                        shutil.copyfileobj(src, f)
                if k % 20000 == 0 or k == len(members):
                    print(f"  {zp.name}: {k}/{len(members)} files")
        parts[part] = {"zip_bytes": zp.stat().st_size, "videos": len(vids), "label_files": n_lab}
        print(f"[{part}] {len(vids)} videos, {n_lab} label files")
    data_card(cfg, parts)


def data_card(cfg, parts):
    """Data card: what the data is, counts, and checks against the official split."""
    raw, frames, labels = Path(cfg["paths"]["raw_shanghaitech"]), frames_dir(cfg, "shanghaitech"), labels_dir(cfg)
    counts = {d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS) for d in frames.iterdir() if d.is_dir()}
    train = sorted(v for v in counts if TRAIN_RE.fullmatch(v))
    test = sorted(v for v in counts if TEST_RE.fullmatch(v))
    problems, ab_vids, ab_frames = [], 0, 0
    for v in test:
        f = labels / f"{v}.npy"
        if not f.exists():
            problems.append(f"test video {v} has no labels")
            continue
        lab = np.load(f)
        if len(lab) != counts[v]:
            problems.append(f"{v}: {len(lab)} labels vs {counts[v]} frames")
        ab_vids, ab_frames = ab_vids + int(lab.max() > 0), ab_frames + int(lab.sum())
    for v in train:
        f = labels / f"{v}.npy"
        if f.exists() and np.load(f).max() > 0:
            problems.append(f"training video {v} has abnormal frames")
    if set(counts) - set(train) - set(test):
        problems.append(f"unexpected folder names: {sorted(set(counts) - set(train) - set(test))[:3]}")
    dvc_file = raw.with_name(raw.name + ".dvc")
    card = {"dataset": "ShanghaiTech Campus",
            "source": {"kaggle_train": cfg["shanghaitech"]["kaggle_train"], "kaggle_test": cfg["shanghaitech"]["kaggle_test"],
                       "raw_dvc_md5": yaml.safe_load(dvc_file.read_text())["outs"][0].get("md5") if dvc_file.exists() else None,
                       "parts": parts},
            "videos": len(counts), "frames": int(sum(counts.values())),
            "official_split": {"train": {"videos": len(train), "frames": int(sum(counts[v] for v in train))},
                               "test": {"videos": len(test), "frames": int(sum(counts[v] for v in test)),
                                        "abnormal_videos": ab_vids, "abnormal_frames": ab_frames}},
            "expected": {"videos": 437, "train_videos": 330, "test_videos": 107, "test_frames": 40791},
            "fps": cfg["shanghaitech"]["fps"], "problems": problems}
    _write_card(cfg, "shanghaitech.json", card)
    s = card["official_split"]
    print(f"\nvideos {card['videos']} (437)  train {s['train']['videos']} (330)  test {s['test']['videos']} (107), "
          f"{s['test']['frames']} frames (40791)\nproblems: {problems or 'none'}")


def _write_card(cfg, name, card):
    meta = Path(cfg["paths"]["metadata"])
    meta.mkdir(parents=True, exist_ok=True)
    (meta / name).write_text(json.dumps(card, indent=2))


# ====================================================================== our phone videos
def extract_video(video_path, out_dir, size=None, step=1, quality=95) -> dict:
    """Video file -> out_dir/000000.jpg ... + meta.json. Also used by the demo / API."""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    orig = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if size:
                frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        i += 1
    cap.release()
    meta = {"source": Path(video_path).name, "fps": fps / step if fps else None, "orig_fps": fps,
            "orig_size": orig, "size": list(size) if size else orig, "frames": saved}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def prepare_manual(cfg):
    f = cfg["frames"]
    src, dst = Path(cfg["paths"]["raw_manual"]), frames_dir(cfg, "manual")
    dst.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in src.rglob("*") if p.suffix in set(cfg["manual"]["video_exts"]))
    metas = {}
    with ProcessPoolExecutor(max_workers=cfg["run"]["workers"]) as pool:
        jobs = {pool.submit(extract_video, v, dst / v.stem, f["resize"], f["step"], f["jpg_quality"]): v for v in videos}
        for job in as_completed(jobs):
            metas[jobs[job].stem] = job.result()
            print(f"{jobs[job].name}: {metas[jobs[job].stem]['frames']} frames")
    _write_card(cfg, "manual.json", {
        "dataset": "manual (own phone videos)", "videos": len(metas), "frames": sum(m["frames"] for m in metas.values()),
        "frame_size": f["resize"], "per_video": {k: {"frames": m["frames"], "fps": m["fps"]} for k, m in sorted(metas.items())}})


# ====================================================================== split
def make_split(cfg) -> dict:
    vids = sorted(d.name for d in frames_dir(cfg, "shanghaitech").iterdir() if d.is_dir())
    return {"train": [v for v in vids if TRAIN_RE.fullmatch(v)], "test": [v for v in vids if TEST_RE.fullmatch(v)]}


def load_split(cfg) -> dict:
    f = Path(cfg["paths"]["metadata"]) / "split.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: run `dvc repro split`")
    return json.loads(f.read_text())


def video_labels(cfg, vid, n_frames) -> np.ndarray:
    """0/1 per frame; zeros for a normal training video without labels."""
    f = labels_dir(cfg) / f"{vid}.npy"
    if not f.exists():
        return np.zeros(n_frames, np.int64)
    lab = np.load(f).astype(np.int64).ravel()
    if len(lab) != n_frames:
        lab = np.pad(lab, (0, max(0, n_frames - len(lab))))[:n_frames]
    return lab


# ====================================================================== PyTorch datasets (models 1, 2)
LINKS = Path("data/processed/_links")      # folders of links the thesis repo's loader can read


def _jpg_view(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in list_frames(src):
        if not (dst / f.name).exists():
            os.symlink(f.resolve(), dst / f.name)
    return dst


def _repo_loader(folder, size):
    """The thesis repo's own data_utils.DataLoader (BGR, [-1, 1]), unchanged."""
    import torchvision.transforms as T
    from third_party.ASTT.data_utils import DataLoader
    return DataLoader(str(folder), T.Compose([T.ToTensor()]), resize_height=size, resize_width=size)


class VideoWindows(tud.Dataset):
    """Model 2: windows of T+1 frames that never cross two videos; each frame read once.
    Item: {"standard": (T+1, 3, S, S), "256": (T+1, 3, 256, 256)}, values in [-1, 1]."""

    def __init__(self, video_dirs, T=4, size=384, out=256, rgb=True):
        self.T, self.size, self.out, self.rgb = T, size, out, rgb
        self.frames = [list_frames(d) for d in video_dirs]
        self.index = [(v, i) for v, fr in enumerate(self.frames) for i in range(len(fr) - T)]

    def __len__(self):
        return len(self.index)

    def _load(self, p):
        img = cv2.imread(str(p))
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f = lambda s: (cv2.resize(img, (s, s)).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
        return f(self.size), f(self.out)

    def __getitem__(self, k):
        v, i = self.index[k]
        pairs = [self._load(p) for p in self.frames[v][i:i + self.T + 1]]
        return {"standard": torch.from_numpy(np.stack([a for a, _ in pairs])),
                "256": torch.from_numpy(np.stack([b for _, b in pairs]))}


def train_dataset(cfg, video_dirs):
    m = cfg["model"]
    if cfg["input"]["loader"] == "repo":          # model 1: one folder with all videos (repo quirk)
        root = LINKS / "train" / m["name"]
        for d in video_dirs:
            _jpg_view(d, root / d.name)
        return _repo_loader(root, m["image_size"])
    return VideoWindows(video_dirs, m["num_frames"], m["image_size"], m["output_size"], cfg["input"]["color"] == "rgb")


def video_dataset(cfg, video_dir):
    """Windows of ONE video (for scoring) and its number of frames."""
    m, video_dir = cfg["model"], Path(video_dir)
    if cfg["input"]["loader"] == "repo":
        ds = _repo_loader(_jpg_view(video_dir, LINKS / "videos" / video_dir.parent.parent.name / video_dir.name),
                          m["image_size"])
        return ds, len(ds.video_frames)
    ds = VideoWindows([video_dir], m["num_frames"], m["image_size"], m["output_size"], cfg["input"]["color"] == "rgb")
    return ds, len(ds.frames[0])


# ====================================================================== command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "prepare", "manual", "split", "link"])
    ap.add_argument("folders", nargs="*", help="link: folders that hold the unpacked videos")
    a = ap.parse_args()
    step, cfg = a.step, load_config()
    if step == "link":
        link_shanghaitech(cfg, a.folders)
    elif step == "split":
        s = make_split(cfg)
        (Path(cfg["paths"]["metadata"]) / "split.json").write_text(json.dumps(s, indent=1))
        print(f"train {len(s['train'])} videos, test {len(s['test'])} videos")
    else:
        {"download": download, "prepare": prepare_shanghaitech, "manual": prepare_manual}[step](cfg)


if __name__ == "__main__":
    main()
