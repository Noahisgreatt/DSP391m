# The four models

All four answer the same question (how abnormal is each frame of a test video?) and are
evaluated by the same code (`src/evaluate.py`) on the same 107 ShanghaiTech test videos.
Parameter counts were measured by building each network.

**No model API / wrapper.** Every network is written as plain PyTorch layers in
`src/models/nets.py` (ViT, Hiera video transformer, ResNet-50 + FPN + Keypoint R-CNN) or comes from
the thesis's own code in `third_party/ASTT`. Only the *weights* are downloaded
(`python -m src.models.nets download <model>`). Library calls such as `timm.create_model`,
`torch.hub.load`, `YOLO(...)`, `clip.load` or torchvision's `*_resnet50_fpn()` are not used
anywhere; torchvision is used only for basic operations (`nms`, `roi_align`). `tests/test_nets.py`
checks that each network gives the same output as the official implementation with the same
weights, and `tests/test_no_wrappers.py` builds all four models with timm / ultralytics / CLIP /
detectron2 / hiera blocked.

| Model | What | Parameters | Training budget |
|---|---|---|---|
| 1 `astt_baseline` | thesis ASTT, unchanged except 4 epochs | 150,579,876 | 4 epochs (thesis: 5), laptop; v4.2 results below are after 2 |
| 2 `astt_improved` | our ASTT | 100,379,299 | 4 epochs (planned 2), Kaggle T4 |
| 3 `mulde` | MULDE (CVPR 2024) on Hiera-B | 70,724,289 | features + 20,000 density steps (226 min), laptop |
| 4 `mulde_improved` | our MULDE | 58,133,185 | features + 20,000 density steps (188 min), laptop |

## Results (v4.2: 107 ShanghaiTech test videos, 40,791 frames, 17,326 abnormal)

| Model | Parameters | AUC micro | AUC per-video norm | AUC macro | EER | Frames/s * | GPU memory * |
|---|---|---|---|---|---|---|---|
| 1 ASTT (thesis, unchanged) | 150.6M | 47.0% | 63.9% | 67.7% | 40.2% | 35.5 | 1151 MB |
| 2 ASTT improved (ours) | 100.4M | 57.6% | 72.6% | **81.0%** | 33.0% | **84.2** | 1365 MB |
| 3 MULDE (Micorek et al.) | 70.7M | **66.7%** | **76.2%** | 75.1% | **29.9%** | 15.3 | 1846 MB |
| 4 MULDE improved (ours) | 58.1M | 66.3% | 71.4% | 73.5% | 33.5% | 7.9 | 1798 MB |
| Behavior rules (no training) | 0 | 59.9% | - | - | 40.1% | - | - |

\* `python -m src.evaluate --benchmark` (RTX 4050 Laptop, video 01_0014, everything computed from
the frames). The "frames/s" in `metrics.json` is the whole test run and is not comparable: model 4
reused the long-clip features cached by model 3. AUC micro = all frames of all videos together;
per-video norm = scores min-max normalised per video first (the usual ShanghaiTech protocol);
macro = mean of the per-video AUCs. Files: `outputs/comparison/` and `outputs/evaluation/<model>/`.

What the numbers say:
- **Model 2 vs model 1: the upgrade works.** +10.6 micro, +8.7 norm, +13.3 macro AUC, EER 40.2 ->
  33.0%, with a third fewer parameters and 2.4x the speed. Caveat: model 2 trained 4 epochs, model 1
  only 2. Model 1's micro AUC below 50% means its raw error is not comparable between cameras (a
  busy camera has a high error even when normal); it still ranks frames inside one video (67.7% macro).
- **Model 3** is the best on the whole test set (micro 66.7%, EER 29.9%). The paper reports 73.1% /
  82.7% with Hiera-B; the gap comes from the details the paper does not give (listed under model 3).
- **Model 4 vs model 3: not better.** It is 18% smaller but loses 0.4 micro and 4.8 norm AUC and is
  2x slower (two Hiera passes per frame). The per-camera mixtures are fitted on ~1/13 of the
  training frames each, which probably makes them less reliable than one mixture on all of them.
- **Behavior rules** (66 loitering, 202 counter-flow, 58 too-fast events on the test set) need no
  training and are weak frame detectors (59.9%), but they are the only part that says WHO and WHY.

## Model 1: ASTT, the thesis model (unchanged)
Tran Minh Tung's PhD thesis (Ch. 3) and IEEE TCSVT 34 (2024) 8292-8309; code
https://github.com/Tungufm/ASTT @ `1d05305` (`third_party/ASTT`, checksummed).

4 past frames -> each through ViT-B/32-384 (one 768-d token per frame) -> temporal transformer
(2 layers) -> temporal cross-attention -> decoder draws the next frame (256x256). Trained on
normal videos only, so a badly predicted frame (high MSE) is abnormal.

| Part | Parameters |
|---|---|
| STE: ViT-B/32-384 (random init, our ViT) | 87,528,192 |
| TTE + temporal CLS | 9,460,224 |
| Cross-attention | 2,360,064 |
| Decoder (Linear 768 -> 256x16x16 + convs) | 51,230,627 |
| Sphere params (unused, kept from the repo) | 769 |
| **Total** | **150,579,876** |

The repo never uploaded its final `model.py`; `ASTTThesis` in `src/models/astt.py` switches on the
cross-attention that `model_ste_tte.py` has commented out, exactly as `model_baseline_ca.py`
applies it. Verified: same weights give the same output as the previous version of our code
(difference 0.0). The repo builds its ViT with `timm.create_model`; we pass it our own ViT
(same layers, same parameter names) instead, without editing the repo file. Training = the repo's: BGR in [-1, 1], frames newest-first, SGD 1e-4
momentum 0.9, OneCycleLR (500 warm-up), batch 8, fp32. **One change: 4 epochs instead of 5** (our
time limit: one epoch takes 15-25 h on the laptop GPU). It was first trained 2 epochs (results v4.2);
model 2 trained 4 on Kaggle, so model 1 was continued to 4 for a fair comparison. Continuing
stretches the learning-rate schedule to 4 epochs from where it was (a warm restart at the point a
4-epoch schedule would be after 2 epochs). Procedure-only additions: `train_steps: auto`, checkpoints every 20 min +
automatic resume.

## Model 2: ASTT improved (ours), 100,379,299 parameters
Same task, same backbone and same building blocks (the repo's Encoder, CrossAttention and
decoder convolutions), so the comparison isolates the changes (`configs/astt_improved.yaml`):

| Change | Why |
|---|---|
| ImageNet-pretrained ViT weights (transfer learning) loaded into our ViT, backbone LR 10x lower | model 1 starts from random weights on only 330 videos |
| Keep the 12x12 patch tokens of every frame; temporal transformer per position | model 1 squeezes a frame into ONE vector and must draw 256x256 pixels from it (a 50M-parameter Linear layer); with positions kept the decoder knows WHERE things are |
| Predict the change from the last frame (residual) | capacity goes to motion, where anomalies are |
| Gradient loss (Liu et al., CVPR 2018) | sharper predictions, so blur is not mistaken for anomaly |
| RGB, oldest-to-newest order, windows never cross two videos | what the pretrained ViT expects; fixes a repo quirk |
| AdamW + cosine schedule, mixed precision (bf16; fp16 on GPUs without bf16 such as Kaggle's T4) | standard fine-tuning; ~2x faster, less GPU memory |

## Model 3: MULDE (Micorek et al., CVPR 2024), 70,724,289 parameters
"MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly
Detection". We found no official code, so `src/models/mulde.py` is written from the paper (frame-centric).

1. **Features.** Every frame -> the 16-frame clip around it (every 4th frame) -> **Hiera-B**, a video
   transformer pretrained on Kinetics-400 (MAE + fine-tuning; our code in `nets.py`, Meta's weights,
   same output as the official code) -> 768 numbers, standardised with the training statistics.
2. **Density.** A network f(x, sigma) (2 x 4096 GELU) is trained by denoising score matching on the
   normal training videos only: `sigma^2 ||grad f(x~) - (x~ - x)/sigma^2||^2 + beta f(x)^2`,
   x~ = x + noise of size sigma, sigma log-uniform in [1e-3, 1]. f is low where normal features are
   dense. It is the only part that is trained.
3. **Score.** f(x, sigma) at 16 noise levels -> negative log-likelihood under a Gaussian mixture
   fitted on the training frames -> Gaussian smoothing over time. High = abnormal.

| Part | Parameters |
|---|---|
| Hiera-B (features) | 50,784,960 |
| Density network (768 + 1 -> 4096 -> 4096 -> 1) | 19,939,329 |

Paper (supplement, Hiera-B, beta 0.1): **73.1% micro / 82.7% macro AUC** on ShanghaiTech (Hiera-L,
214M: 81.3% / 85.9%). Hiera-B is the smallest video backbone they report, the closest to our size
budget. Not given in the paper, chosen by us and marked in `configs/mulde.yaml`: how sigma enters
the network (log sigma appended to x), log spacing of the 16 levels, 3 mixture components, 20,000
training steps, smoothing width 7, a feature for every 4th training frame.

## Model 4: MULDE improved (ours), 58,133,185 parameters
Same method (`configs/mulde_improved.yaml`, changes marked `[CHANGE]`):

| Change | Why |
|---|---|
| Two clip lengths side by side: 16 x every 2nd frame (1.3 s) + 16 x every 4th (2.6 s) | short clips see fast motion (running, pushing against the flow), long ones slow behaviour (loitering); the long-clip features are shared with model 3 |
| One Gaussian mixture per camera | ShanghaiTech has 13 cameras; what is "normal" (crowd size, walking direction, distance) differs per camera |
| Density network 2 x 2048 instead of 2 x 4096 | 7.3M instead of 19.9M parameters |

Each change can be switched off in the config, so its effect can be measured (ablation).

## Behavior rules (no training)
Per tracked person (Keypoint R-CNN people boxes + our IoU tracker with Hungarian matching): LOITERING (stays within 1 body height for 10 s),
COUNTER_FLOW (direction > 120 degrees from the nearby crowd, or from the camera's normal flow),
TOO_FAST (> 2 body heights/s and > 2x the neighbours). Thresholds: `configs/behavior.yaml`.
They give the *reason* that the frame-level models can't, and are evaluated as a yes/no detector.
