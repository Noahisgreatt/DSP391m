"""The Hugging Face Space gets the app's code and a Dockerfile made from docker/Dockerfile.app."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "docker/deploy_space.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def test_space_bundle(tmp_path):
    folder = deploy.stage(with_model=False, out=str(tmp_path / "space"))
    docker = (folder / "Dockerfile").read_text()
    assert "ARG BASE=python:3.11-slim" in docker and "ARG TORCH=cpu" in docker          # a free Space has no GPU
    assert docker.rstrip().endswith("ENV VAD_MAX_SECONDS=15 VAD_DETECTOR_SIZE=480 VAD_DETECT_EVERY=3")
    assert "sdk: docker" in (folder / "README.md").read_text() and "app_port: 7860" in (folder / "README.md").read_text()
    for need in ("app/server.py", "api/main.py", "api/schemas.py", "src/inference.py", "configs/astt_improved.yaml",
                 "third_party/fetch.py", "requirements-api.txt", "requirements-ui.txt"):
        assert (folder / need).exists(), need
    assert not (folder / "outputs").exists()                       # code-only deploy keeps the Space's weights
    assert not list(folder.rglob("__pycache__"))
