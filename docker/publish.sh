#!/usr/bin/env bash
# Build the app image (docker/Dockerfile.app: web page + API + model 2) and publish it to
# GitHub Packages, so anyone can run it with one command:
#   bash docker/publish.sh v5.0
#   -> ghcr.io/<owner>/dsp391m-app:v5.0 and :latest (GPU), :v5.0-cpu and :cpu (smaller, CPU only)
#   then:  docker run --gpus all -p 7860:7860 ghcr.io/<owner>/dsp391m-app   -> http://localhost:7860
# Once before: docker login ghcr.io -u <your GitHub user name>
#   (password: a GitHub token with "write:packages", github.com -> Settings -> Developer settings
#    -> Personal access tokens -> Tokens (classic). Type it in the terminal, never share it.)
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=${1:?give a version, e.g. bash docker/publish.sh v5.0}
OWNER=$(git remote get-url origin | sed -E 's#.*github.com[:/]([^/]+)/.*#\1#' | tr '[:upper:]' '[:lower:]')
IMG=ghcr.io/$OWNER/dsp391m-app
for f in outputs/training/astt_improved/model.pth outputs/training/astt_improved/run.json outputs/evaluation/astt_improved/metrics.json; do
    [[ -f $f ]] || { echo "missing $f: train and evaluate model 2 first (bash run_all.sh astt_improved)"; exit 1; }
done
echo "== GPU image (also runs without a GPU): $IMG:$VERSION, :latest"
docker build -f docker/Dockerfile.app -t "$IMG:$VERSION" -t "$IMG:latest" .
docker push "$IMG:$VERSION" && docker push "$IMG:latest"
echo "== CPU image, smaller (the Hugging Face Space uses it): $IMG:$VERSION-cpu, :cpu"
docker build -f docker/Dockerfile.app --build-arg BASE=python:3.11-slim --build-arg TORCH=cpu \
    -t "$IMG:$VERSION-cpu" -t "$IMG:cpu" .
docker push "$IMG:$VERSION-cpu" && docker push "$IMG:cpu"
echo
echo "published: $IMG:$VERSION (GPU) and $IMG:$VERSION-cpu"
echo "first time only: make it public (GitHub -> your profile -> Packages -> dsp391m-app -> Package settings -> Change visibility -> Public)"
echo "anyone can now run:  docker run --gpus all -p 7860:7860 $IMG   (no NVIDIA GPU: docker run -p 7860:7860 $IMG:cpu)"
echo "the Hugging Face Space (docker/huggingface/) rebuilds from $IMG:cpu: Space -> Settings -> Factory rebuild"
