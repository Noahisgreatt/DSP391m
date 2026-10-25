# Third-party code (fetched, not committed)

```bash
python third_party/fetch.py          # download at pinned commits + verify sha256
python third_party/fetch.py --check  # prove nothing was edited
```

| Folder | Source (pinned commit) | Used for | Licence |
|---|---|---|---|
| `ASTT/` | https://github.com/Tungufm/ASTT @ `1d05305` | Model 1 (thesis model) and the building blocks of model 2 | none given: coursework use with citation, do not republish |
| `AIVAD/` | https://github.com/talreiss/Accurate-Interpretable-VAD @ `88ebc02` | Model 3 (AI-VAD): object box filtering, FlowNet2 network code, scoring; FlowNetC of model 4 reuses FlowNet2's weights | Yissum research licence (research / teaching only) |
| `AlphaPose/` | https://github.com/MVIG-SJTU/AlphaPose @ `c60106d` | Model 3's pose network layers (FastPose ResNet-50, 4 files) | SJTU non-commercial research licence |

None of these licences allows republishing the code, so it is downloaded by `fetch.py` instead of
being stored in our repository. `*.sha256` lists the files we rely on; `fetch.py --check` proves they
are byte-for-byte the originals. **Our code never edits them**: where a piece needs adapting (for
example FlowNet2's CUDA-only layers, or ASTT's `timm.create_model` call, which gets our own ViT), the
replacement lives in `src/models/` and is plugged in from outside.

## Citations
- Tran, T. M., Bui, D. C., Nguyen, T. V., Nguyen, K. "Transformer-based Spatio-Temporal Unsupervised
  Traffic Anomaly Detection in Aerial Videos", IEEE TCSVT 34 (2024) 8292-8309.
- Reiss, T., Hoshen, Y. "An Attribute-based Method for Video Anomaly Detection", TMLR (2025),
  arXiv:2212.00789.
- Fang, H.-S. et al. "AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in
  Real-Time", IEEE TPAMI (2022).
