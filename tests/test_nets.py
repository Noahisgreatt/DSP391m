"""Our own networks give the SAME output as the reference implementations when both have the
same weights (references are only used here, never by the project)."""
import math

import pytest
import torch
import torch.nn as nn

from src.models import nets

torch.manual_seed(0)


def test_vit_equals_timm():
    timm = pytest.importorskip("timm")
    ref = timm.create_model("vit_base_patch32_384", num_classes=0, pretrained=False).eval()
    mine = nets.ViT().eval()
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
    assert nets.n_params(mine) == 87_528_192


def test_clip_visual_equals_openai():
    cm = pytest.importorskip("clip.model")
    ref = cm.VisionTransformer(224, 16, 768, 12, 12, 512).eval()
    mine = nets.CLIPVisual().eval()
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(mine(x), ref(x), atol=1e-5)


def _frozen(m):
    from torchvision.ops import FrozenBatchNorm2d
    for name, c in m.named_children():
        if isinstance(c, nn.BatchNorm2d):
            f = FrozenBatchNorm2d(c.num_features)
            for k in ("weight", "bias", "running_mean", "running_var"):
                getattr(f, k).copy_(getattr(c, k).detach())
            setattr(m, name, f)
        else:
            _frozen(c)


@pytest.mark.parametrize("kind", ["keypoint", "box"])
def test_rcnn_equals_torchvision(kind):
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
    if kind == "keypoint":
        ref = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=320, box_score_thresh=0.0)
        mine = nets.RCNN(2, 17, min_size=320, score_thresh=0.0)
    else:
        ref = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=320, box_score_thresh=0.02)
        mine = nets.RCNN(91, min_size=320, score_thresh=0.02)
    _frozen(ref)
    ref.eval(); mine.eval()
    mine.load_state_dict({k: v for k, v in ref.state_dict().items() if k in mine.state_dict()})
    imgs = [torch.rand(3, 240, 428)]
    with torch.no_grad():
        a, b = ref(imgs)[0], mine(imgs)[0]
    assert len(a["boxes"]) == len(b["boxes"]) > 0
    assert torch.allclose(a["boxes"], b["boxes"], atol=1e-3) and torch.equal(a["labels"], b["labels"])
    if kind == "keypoint":
        assert torch.allclose(a["keypoints"], b["keypoints"], atol=1e-3)


def test_flownet_correlation_matches_cuda_kernel_definition():
    a, b = torch.randn(1, 3, 7, 9), torch.randn(1, 3, 7, 9)
    out = torch.empty(0)
    nets._correlation_forward(a, b, None, None, out, 4, 1, 4, 1, 2, 1)
    r, ref = 2, torch.zeros(1, 25, 7, 9)
    for y in range(7):
        for x in range(9):
            for tj in range(-r, r + 1):
                for ti in range(-r, r + 1):
                    yy, xx = y + tj * 2, x + ti * 2
                    if 0 <= yy < 7 and 0 <= xx < 9:
                        ref[0, (tj + r) * 5 + ti + r, y, x] = (a[0, :, y, x] * b[0, :, yy, xx]).sum() / 3
    assert torch.allclose(out, ref, atol=1e-6)


def test_flownet_resample_matches_cuda_kernel_definition():
    img, flow = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8) * 3
    out = torch.zeros_like(img)
    nets._resample2d_forward(img, flow, out, 1, True)
    cl = lambda v, m: max(min(v, m - 1), 0)
    for y in range(6):
        for x in range(8):
            xf, yf = x + float(flow[0, 0, y, x]), y + float(flow[0, 1, y, x])
            al, be = xf - math.floor(xf), yf - math.floor(yf)
            xl, xr, yt, yb = cl(math.floor(xf), 8), cl(math.floor(xf) + 1, 8), cl(math.floor(yf), 6), cl(math.floor(yf) + 1, 6)
            ref = ((1 - al) * (1 - be) * img[0, :, yt, xl] + al * (1 - be) * img[0, :, yt, xr]
                   + (1 - al) * be * img[0, :, yb, xl] + al * be * img[0, :, yb, xr])
            assert torch.allclose(out[0, :, y, x], ref, atol=1e-5)


def test_parameter_counts():
    assert nets.n_params(nets.RCNN(2, 17)) == 59_137_258          # Keypoint R-CNN
    assert nets.n_params(nets.RCNN(91)) == 41_755_286             # Faster R-CNN part of Mask R-CNN
    assert nets.n_params(nets.FastPose()) == 40_577_745
    assert nets.n_params(nets.CLIPVisual()) == 86_192_640
    assert nets.n_params(nets.FlowNetC(weights=False)) == 39_175_298
