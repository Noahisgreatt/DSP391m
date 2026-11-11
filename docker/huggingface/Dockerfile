# Hugging Face Space (free CPU): the published app image (docker/publish.sh) in its light mode.
FROM ghcr.io/noahisgreatt/dsp391m-app:cpu
# a free Space has 2 CPU cores and no GPU: analyze at most 15 s, find people at 480 px on every 3rd frame
ENV VAD_MAX_SECONDS=15 VAD_DETECTOR_SIZE=480 VAD_DETECT_EVERY=3
