# The four models

All four answer the same question (how abnormal is each frame of a test video?) and are
evaluated by the same code (`src/evaluation/metrics.py`) on the same 107 ShanghaiTech test videos.
Parameter counts were measured by building each network.

## Model 1: ASTT, the thesis model (unchanged)
Tran Minh Tung's PhD thesis (Ch. 3) and IEEE TCSVT 34 (2024) 8292-8309; code
https://github.com/Tungufm/ASTT @ `1d05305` (`third_party/ASTT`, checksummed).

4 past frames -> each through ViT-B/32-384 (one 768-d token per frame) -> temporal transformer
(2 layers) -> temporal cross-attention -> decoder draws the next frame (256x256). Trained on
normal videos only, so a badly predicted frame (high MSE) is abnormal.

| Part | Parameters |
|---|---|
| STE: ViT-B/32-384 (random init) | 87,528,192 |
| TTE + temporal CLS | 9,460,224 |
| Cross-attention | 2,360,064 |
| Decoder (Linear 768 -> 256x16x16 + convs) | 51,230,627 |
| Sphere params (unused, kept from the repo) | 769 |
| **Total** | **150,579,876** |

The repo never uploaded its final `model.py`; `src/models/astt_baseline.py` switches on the
cross-attention that `model_ste_tte.py` has commented out, exactly as `model_baseline_ca.py`
applies it. Verified: same weights give the same output as the previous version of our code
(difference 0.0). Training = the repo's: BGR in [-1, 1], frames newest-first, SGD 1e-4
momentum 0.9, OneCycleLR (500 warm-up), batch 8, 5 epochs, fp32. Procedure-only additions:
`train_steps: auto`, checkpoints every 20 min + automatic resume.

## Model 2: ASTT improved (ours), 100,379,299 parameters
Same task, same backbone and same building blocks (the repo's Encoder, CrossAttention and
decoder convolutions), so the comparison isolates the changes (`configs/astt_improved.yaml`):

| Change | Why |
|---|---|
| ImageNet-pretrained ViT (transfer learning), backbone LR 10x lower | model 1 starts from random weights on only 330 videos |
| Keep the 12x12 patch tokens of every frame; temporal transformer per position | model 1 squeezes a frame into ONE vector and must draw 256x256 pixels from it (a 50M-parameter Linear layer); with positions kept the decoder knows WHERE things are |
| Predict the change from the last frame (residual) | capacity goes to motion, where anomalies are |
| Gradient loss (Liu et al., CVPR 2018) | sharper predictions, so blur is not mistaken for anomaly |
| RGB, oldest-to-newest order, windows never cross two videos | what the pretrained ViT expects; fixes a repo quirk |
| AdamW + cosine schedule, bf16 mixed precision | standard fine-tuning; ~2x faster, less GPU memory |

## Model 3: AI-VAD (Reiss & Hoshen, TMLR 2025), 333,632,435 parameters
https://github.com/talreiss/Accurate-Interpretable-VAD @ `88ebc02` (`third_party/AIVAD`).
Nothing is trained by gradient descent: pretrained networks describe every object, and density
estimators fitted on the normal training objects say how unusual each description is.

| Step | Network | Parameters |
|---|---|---|
| Objects | Mask R-CNN R50-FPN 3x (detectron2), conf 0.8, + moving blobs | 44,343,216 |
| Velocity | FlowNet2 optical flow (1024x640) -> 8-bin histogram per box | 162,518,834 |
| Pose | AlphaPose FastPose ResNet-50 -> 17 keypoints per person | 40,577,745 |
| Appearance | CLIP ViT-B/16 image embedding | 86,192,640 |

Scoring (AI-VAD `evaluate.py`): velocity -> GMM (5 components); pose / appearance -> distance to
the nearest training sample; each calibrated on the training videos (leave-one-video-out), the
most unusual object per frame, velocity + pose summed (the released ShanghaiTech setting), Gaussian
smoothing sigma 7. Paper: 85.9% micro / 89.6% macro AUC on ShanghaiTech.
Two ways to run it (`model.features`): `extract` = run all networks on our frames;
`released` = the authors' attribute files, to reproduce their numbers exactly.
Code is used unchanged; two things are plugged in from outside instead of editing it:
FlowNet2's CUDA-only layers (correlation, resample, channel norm) are re-written in plain
PyTorch (`src/models/aivad/flownet2_ops.py`, checked value-for-value against the CUDA kernel's
definition in `tests/test_aivad.py`), and the 1-NN search runs in PyTorch instead of faiss-gpu
(exact, also tested).

## Model 4: AI-VAD Lite (ours), 64,147,417 parameters (YOLO11x-pose 58,889,881 + RAFT 5,257,536)
Same attributes and scoring as model 3 (`configs/aivad_lite.yaml`):

| Change | Effect |
|---|---|
| YOLO11x-pose (58.9M) finds people AND their keypoints in one pass | replaces Mask R-CNN + AlphaPose (84.9M, two networks) |
| RAFT optical flow (5.3M) at 512x288 | replaces FlowNet2 (162.5M) at 1024x640, the slowest step |
| No CLIP | AI-VAD's ShanghaiTech score doesn't use appearance anyway |
| Per-camera density models | ShanghaiTech has 13 cameras; "normal speed" in pixels differs per camera |
| Crowd-relative velocity attribute | a person's motion minus the motion of the others in the frame: counter-flow and running stand out (the project's target behaviours) |
| Track smoothing (ByteTrack ids) | a person's score averaged over their last 5 frames: fewer one-frame false alarms |

Each improvement can be switched off in the config, so its effect can be measured (ablation).
Model 4 is the one served by the API (model 3 is too heavy to deploy); TensorRT comes last.

## Behavior rules (no training)
Per tracked person (YOLO11s + ByteTrack): LOITERING (stays within 1 body height for 10 s),
COUNTER_FLOW (direction > 120 degrees from the nearby crowd, or from the camera's normal flow),
TOO_FAST (> 2 body heights/s and > 2x the neighbours). Thresholds: `configs/behavior.yaml`.
They give the *reason* that the frame-level models can't, and are evaluated as a yes/no detector.
