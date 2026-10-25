"""Get the raw ShanghaiTech data: two zip files that DVC then versions.

    python -m src.data.download                          # download both zips from Kaggle
    python -m src.data.download --pack-train DIR --pack-test DIR   # already unzipped? zip them

Result:  data/raw/shanghaitech/train.zip   (Kaggle nikanvasei/shanghaitech-campus-dataset)
         data/raw/shanghaitech/test.zip    (Kaggle nikanvasei/shanghaitech-campus-dataset-test)
Then:    dvc add data/raw/shanghaitech  &&  dvc push

Why zips: ~180,000 small JPEGs are very slow to push to / pull from Google Drive one by one.
Two large files are fast, and src.data.prepare_shanghaitech unpacks them.
Needs a Kaggle API token (KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json) for downloading.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

from src.config import load_config


def kaggle_download(slug: str, dst_zip: Path) -> None:
    if dst_zip.exists():
        print(f"{dst_zip} already exists, skipped")
        return
    tmp = dst_zip.parent / "_kaggle_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"downloading {slug} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(tmp)], check=True)
    zips = sorted(tmp.glob("*.zip"))
    if len(zips) != 1:
        raise SystemExit(f"expected one zip in {tmp}, found {[z.name for z in zips]}")
    zips[0].replace(dst_zip)
    tmp.rmdir()


def pack(src: Path, dst_zip: Path) -> None:
    """Zip an unzipped Kaggle folder without compression (JPEGs don't compress). Only frames and
    .npy labels are taken (a half-downloaded .zip lying in the folder is ignored)."""
    if dst_zip.exists():
        print(f"{dst_zip} already exists, skipped")
        return
    keep = {".jpg", ".jpeg", ".png", ".npy"}
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in keep)
    tmp = dst_zip.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for k, f in enumerate(files, 1):
            z.write(f, f.relative_to(src).as_posix())
            if k % 20000 == 0 or k == len(files):
                print(f"  {dst_zip.name}: {k}/{len(files)} files")
    os.replace(tmp, dst_zip)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-train", type=Path, help="unzipped train part to zip instead of downloading")
    ap.add_argument("--pack-test", type=Path, help="unzipped test part to zip instead of downloading")
    args = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["paths"]["raw_shanghaitech"])
    out.mkdir(parents=True, exist_ok=True)
    for part, packed in (("train", args.pack_train), ("test", args.pack_test)):
        if packed:
            pack(packed, out / f"{part}.zip")
        else:
            kaggle_download(cfg["shanghaitech"][f"kaggle_{part}"], out / f"{part}.zip")
    print(f"done. Next:  dvc add {out}  &&  dvc push")


if __name__ == "__main__":
    main()
