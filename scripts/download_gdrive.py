"""Download files or folders from Google Drive (e.g. the manual final-check videos).

    python scripts/download_gdrive.py --url "https://drive.google.com/drive/folders/<id>" --out data/raw/manual/videos
    python scripts/download_gdrive.py --id <file_id> --out data/raw/manual/videos/demo.mp4

The share link must be "Anyone with the link". Requires `pip install gdown`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Drive file or folder URL")
    ap.add_argument("--id", help="Drive file id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    if args.url and "/folders/" in args.url:
        out.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(url=args.url, output=str(out), quiet=False, use_cookies=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url=args.url, id=args.id, output=str(out), quiet=False, fuzzy=True)


if __name__ == "__main__":
    main()
