"""Deploy the app (web page + API + model 2) to a Hugging Face Space, which builds and runs it.

    python docker/deploy_space.py <hf-user>/dsp391m --with-model   # first time, from the computer with the model
    python docker/deploy_space.py <hf-user>/dsp391m                # later: code only (what CI/CD does on every version)
    python docker/deploy_space.py <hf-user>/dsp391m --dry-run      # only show what would be uploaded

Needs a Hugging Face token with write access: `huggingface-cli login` once (type it in the
terminal), or the HF_TOKEN environment variable (the GitHub secret in CI/CD).

What is uploaded: the code the app needs, docker/huggingface/README.md as the Space's page, and a
Dockerfile made from docker/Dockerfile.app (CPU variant: a free Space has no GPU) plus the light
CPU settings. --with-model adds model 2's weights and results (outputs/...); a code-only deploy
keeps the ones already in the Space.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["requirements-api.txt", "requirements-ui.txt", "src", "api", "app", "configs",
        "third_party/fetch.py", "third_party/ASTT.sha256", "third_party/README.md"]
MODEL = ["outputs/training/astt_improved/model.pth", "outputs/training/astt_improved/run.json",
         "outputs/evaluation/astt_improved/metrics.json"]
LIGHT = "VAD_MAX_SECONDS=15 VAD_DETECTOR_SIZE=480 VAD_DETECT_EVERY=3"      # 2 CPU cores, no GPU


def space_dockerfile() -> str:
    """docker/Dockerfile.app, CPU variant, + the light settings: one source for both images."""
    s = (ROOT / "docker/Dockerfile.app").read_text()
    s = s.replace("ARG BASE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime", "ARG BASE=python:3.11-slim")
    s = s.replace("ARG TORCH=gpu", "ARG TORCH=cpu")
    assert "ARG BASE=python:3.11-slim" in s and "ARG TORCH=cpu" in s, "docker/Dockerfile.app changed: update deploy_space.py"
    return ("# Made by docker/deploy_space.py from docker/Dockerfile.app (CPU variant + light settings).\n"
            + s + f"\n# a free Space: 2 CPU cores, no GPU\nENV {LIGHT}\n")


def stage(with_model: bool, out: str | None = None) -> Path:
    out = Path(out) if out else Path(tempfile.mkdtemp(prefix="space_"))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    for rel in CODE + (MODEL if with_model else []):
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel}" + (": train and evaluate model 2 first" if rel in MODEL else ""))
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    shutil.copy2(ROOT / "docker/huggingface/README.md", out / "README.md")
    (out / "Dockerfile").write_text(space_dockerfile())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space", help="<hf-user>/<space-name>, e.g. anish/dsp391m")
    ap.add_argument("--with-model", action="store_true", help="also upload model 2's weights and results")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded, upload nothing")
    ap.add_argument("--message", default="Deploy from the DSP391m repository")
    ap.add_argument("--out", help="stage the Space's files in this (new / empty) folder (CI builds it to test)")
    a = ap.parse_args()
    folder = stage(a.with_model, a.out)
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    print(f"{len(files)} files for {a.space}:", ", ".join(f for f in files if "/" not in f or f.startswith("outputs")), "...")
    if a.dry_run:
        print(f"(dry run) staged in {folder}: `docker build {folder}` builds what the Space builds")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=a.space, repo_type="space", commit_message=a.message,
                      delete_patterns=["src/**", "api/**", "app/**", "configs/**"])   # code removed in git goes too
    user, name = a.space.split("/")
    print(f"deployed: https://huggingface.co/spaces/{a.space} (building, ~10 min)")
    print(f"API: https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space/docs")
    shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
