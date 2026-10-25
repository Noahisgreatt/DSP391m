# The four models

All four answer the same question (how abnormal is each frame of a test video?) and are
evaluated by the same code (`src/evaluate.py`) on the same 107 ShanghaiTech test videos.
Parameter counts were measured by building each network.

**No model API / wrapper.** Every network is written as plain PyTorch layers in
`src/models/nets.py` (ViT, CLIP image encoder, ResNet-50 + FPN + Faster / Keypoint R-CNN heads,
FastPose, FlowNetC) or comes from the papers' own code in `third_party/` (ASTT blocks, FlowNet2).
Only the *weights* are downloaded (`python -m src.models.nets download <group>`). Library calls
such as `timm.create_model`, `YOLO(...)`, `clip.load`, detectron2's `DefaultPredictor` or
torchvision's `*_resnet50_fpn()` are not used anywhere; torchvision is used only for basic
operations (`nms`, `roi_align`). `tests/test_nets.py` checks that each network gives the same
output as the reference implementation with the same weights (difference 0.0), and
`tests/test_no_wrappers.py` builds all four models with timm / ultralytics / CLIP / detectron2
blocked.

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
momentum 0.9, OneCycleLR (500 warm-up), batch 8, 5 epochs, fp32. Procedure-only additions:
`train_steps: auto`, checkpoints every 20 min + automatic resume.

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
| AdamW + cosine schedule, bf16 mixed precision | standard fine-tuning; ~2x faster, less GPU memory |

## Model 3: AI-VAD (Reiss & Hoshen, TMLR 2025), 331,044,505 parameters
https://github.com/talreiss/Accurate-Interpretable-VAD @ `88ebc02` (`third_party/AIVAD`).
Nothing is trained by gradient descent: pretrained networks describe every object, and density
estimators fitted on the normal training objects say how unusual each description is.

| Step | Network | Parameters |
|---|---|---|
| Objects | R-CNN R50-FPN with COCO Mask R-CNN weights (boxes only), conf 0.8, + moving blobs | 41,755,286 |
| Velocity | FlowNet2 optical flow (1024x640) -> 8-bin histogram per box | 162,518,834 |
| Pose | AlphaPose FastPose ResNet-50 (our code, AlphaPose weights) -> 17 keypoints, box 48x112 | 40,577,745 |
| Appearance | CLIP ViT-B/16 image encoder (our code, OpenAI weights) | 86,192,640 |

Scoring (AI-VAD `evaluate.py`): velocity -> GMM (5 components); pose / appearance -> distance to
the nearest training sample; each calibrated on the training videos (leave-one-video-out), the
most unusual object per frame, velocity + pose summed (the released ShanghaiTech setting), Gaussian
smoothing sigma 7. Paper: 85.9% micro / 89.6% macro AUC on ShanghaiTech.
Two ways to run it (`model.features`): `extract` = run all networks on our frames;
`released` = the authors' attribute files, to reproduce their numbers exactly.
Code is used unchanged; things are plugged in from outside instead of editing it:
FlowNet2's CUDA-only layers (correlation, resample, channel norm) are re-written in plain
PyTorch (`src/models/nets.py`, checked value-for-value in `tests/test_aivad.py`); the 1-NN search
runs in PyTorch instead of faiss-gpu (exact, also tested); the velocity histogram is re-written so
it works with numpy 2 (same numbers, tested). One honest difference: AI-VAD's boxes come from
detectron2's Mask R-CNN 3x weights; we use torchvision's COCO Mask R-CNN weights in our own
R-CNN, because detectron2 is a model wrapper. Both are R50-FPN COCO detectors (box AP 37.9 vs 41.0).

## Model 4: AI-VAD Lite (ours), 98,312,556 parameters (Keypoint R-CNN 59,137,258 + FlowNetC 39,175,298)
Same attributes and scoring as model 3 (`configs/aivad_lite.yaml`):

| Change | Effect |
|---|---|
| Keypoint R-CNN (59.1M) finds people AND their keypoints in one pass | replaces Mask R-CNN + FastPose (82.3M, two networks, two passes) |
| FlowNetC (39.2M, FlowNet2's first network with the same weights) at 512x320 | replaces FlowNet2 (162.5M, 5 networks) at 1024x640, the slowest step |
| No CLIP | AI-VAD's ShanghaiTech score doesn't use appearance anyway |
| Per-camera density models | ShanghaiTech has 13 cameras; "normal speed" in pixels differs per camera |
| Crowd-relative velocity attribute | a person's motion minus the motion of the others in the frame: counter-flow and running stand out (the project's target behaviours) |
| Track smoothing (our IoU tracker, `src/behavior.py`) | a person's score averaged over their last 5 frames: fewer one-frame false alarms |

Each improvement can be switched off in the config, so its effect can be measured (ablation).
Model 4 is the one served by the API (model 3 is too heavy to deploy); TensorRT comes last.

## Behavior rules (no training)
Per tracked person (Keypoint R-CNN people boxes + our IoU tracker with Hungarian matching): LOITERING (stays within 1 body height for 10 s),
COUNTER_FLOW (direction > 120 degrees from the nearby crowd, or from the camera's normal flow),
TOO_FAST (> 2 body heights/s and > 2x the neighbours). Thresholds: `configs/behavior.yaml`.
They give the *reason* that the frame-level models can't, and are evaluated as a yes/no detector.
