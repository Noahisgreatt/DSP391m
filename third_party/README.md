# Third-party code and weights

```bash
python third_party/fetch.py          # download the thesis code at its pinned commit + verify sha256
python third_party/fetch.py --check  # prove nothing was edited
```

| What | Source | Used for | Licence |
|---|---|---|---|
| `ASTT/` (code, fetched) | https://github.com/Tungufm/ASTT @ `1d05305` | Model 1 (thesis model) and the building blocks of model 2 | none given: coursework use with citation, do not republish |
| Hiera-B weights (`weights/`) | Meta, https://github.com/facebookresearch/hiera (`hiera_base_16x224`, Kinetics-400) | Models 3 and 4 (MULDE features); the network itself is our code in `src/models/nets.py` | CC BY-NC 4.0 (non-commercial) |

The ASTT licence does not allow republishing, so the code is downloaded by `fetch.py` instead of
being stored in our repository. `ASTT.sha256` lists the files we rely on; `fetch.py --check` proves
they are byte-for-byte the originals. **Our code never edits them**: where a piece needs adapting
(ASTT's `timm.create_model` call, which gets our own ViT), the replacement lives in `src/models/`
and is plugged in from outside.

## Citations
- Tran, T. M., Bui, D. C., Nguyen, T. V., Nguyen, K. "Transformer-based Spatio-Temporal Unsupervised
  Traffic Anomaly Detection in Aerial Videos", IEEE TCSVT 34 (2024) 8292-8309.
- Micorek, J., Possegger, H., Narnhofer, D., Bischof, H., Kozinski, M. "MULDE: Multiscale
  Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection", CVPR 2024.
- Ryali, C. et al. "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles",
  ICML 2023.
