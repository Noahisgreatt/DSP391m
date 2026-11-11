---
title: DSP391m Video Anomaly Detection
emoji: 🎥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Who is loitering or moving against the crowd in a video
---

# DSP391m · Video anomaly detection

Upload a short surveillance video: every person gets a box and a direction arrow, people who
**loiter**, move **against the crowd** or **run** are flagged, and the model (our improved ASTT,
100.4M parameters, trained on ShanghaiTech Campus) scores how abnormal each frame is.

- Web page: this Space.  API: `/docs` on this Space's address (`https://<user>-<space>.hf.space/docs`).
- Free CPU server: a 10-second clip takes a few minutes; only the first 15 s are analyzed.
  For full speed run the same app on a computer with an NVIDIA GPU:
  `docker run --gpus all -p 7860:7860 ghcr.io/noahisgreatt/dsp391m-app`
- Code, data versions, results: https://github.com/Noahisgreatt/DSP391m
