"""Download the third-party code this project builds on, at pinned commits, and prove it is unchanged.

    python third_party/fetch.py            # all three (needs git + internet to github.com)
    python third_party/fetch.py --check    # only verify the sha256 sums

Not committed to our repo because none of the licences allow republishing:
  ASTT       no licence file              -> model 1 (the thesis model)
  AIVAD      Yissum research licence      -> model 3 (AI-VAD)
  AlphaPose  SJTU non-commercial licence  -> the pose network of model 3 (FastPose, 4 files used)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    "ASTT": ("https://github.com/Tungufm/ASTT", "1d05305a94fb47b33897c7aa085c5351bcbbc8f9"),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(name: str) -> bool:
    root, ok = HERE / name, True
    for line in (HERE / f"{name}.sha256").read_text().splitlines():
        want, rel = line.split(maxsplit=1)
        f = root / rel.strip()
        if not f.exists() or sha256(f) != want:
            print(f"  {name}: {rel.strip()} MISSING or CHANGED")
            ok = False
    print(f"{name}: {'OK, unchanged' if ok else 'FAILED'}")
    return ok


def fetch(name: str) -> None:
    url, commit = SOURCES[name]
    dst = HERE / name
    if dst.exists() and check(name):
        return
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=dst, check=True, capture_output=True)
    print(f"{name}: fetching {url} @ {commit[:7]} ...")
    run("init", "-q")
    run("fetch", "-q", "--depth", "1", url, commit)
    run("checkout", "-q", "FETCH_HEAD")
    shutil.rmtree(dst / ".git")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        for n in SOURCES:
            fetch(n)
    ok = all([check(n) for n in SOURCES])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
