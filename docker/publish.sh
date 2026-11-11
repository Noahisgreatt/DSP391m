#!/usr/bin/env bash
# Build the app image (docker/Dockerfile.app: web page + API + model 2) and publish it to
# GitHub Packages, so anyone can run it with one command:
#   bash docker/publish.sh v5.0
#   -> ghcr.io/<owner>/dsp391m-app:v5.0 and :latest
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
docker build -f docker/Dockerfile.app -t "$IMG:$VERSION" -t "$IMG:latest" .
docker push "$IMG:$VERSION"
docker push "$IMG:latest"
echo
echo "published: $IMG:$VERSION"
echo "first time only: make it public (GitHub -> your profile -> Packages -> dsp391m-app -> Package settings -> Change visibility -> Public)"
echo "anyone can now run:  docker run --gpus all -p 7860:7860 $IMG   (no NVIDIA GPU: leave out --gpus all)"
