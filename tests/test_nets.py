"""Our own networks give the SAME output as the reference implementations when both have the
same weights (references are only used here, never by the project)."""
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


def test_hiera_equals_official():
    hiera = pytest.importorskip("hiera")                     # pip install hiera-transformer (test only)
    ref = hiera.hiera_base_16x224(pretrained=False).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.05)
    mine = nets.Hiera().eval()
    mine.load_state_dict({k: v for k, v in ref.state_dict().items() if not k.startswith("head.")})
    x, feat = torch.randn(1, 3, 16, 224, 224), {}
    ref.head.register_forward_pre_hook(lambda m, i: feat.setdefault("f", i[0]))   # the classifier's input
    with torch.no_grad():
        ref(x)
        assert torch.allclose(mine(x), feat["f"], atol=1e-5)


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
def test_rcnn_equals_torchvision(kind, tmp_path):
    """Same output as torchvision with the same weights, loaded from a file that uses the old
    (2019) layer names, like the real COCO weight files."""
    import re
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
    if kind == "keypoint":
        ref = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=320, box_score_thresh=0.0)
        mine = nets.RCNN(2, 17, min_size=320, score_thresh=0.0)
    else:
        ref = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=320, box_score_thresh=0.02)
        mine = nets.RCNN(91, min_size=320, score_thresh=0.02)
    _frozen(ref)
    ref.eval(); mine.eval()
    old = {re.sub(r"rpn\.head\.conv\.0\.0\.", "rpn.head.conv.", re.sub(r"(_blocks\.\d)\.0\.", r"\1.", k)): v
           for k, v in ref.state_dict().items()}
    assert "backbone.fpn.inner_blocks.0.weight" in old and "rpn.head.conv.weight" in old
    torch.save(old, tmp_path / "w.pth")
    mine.load_pretrained("x", path=tmp_path / "w.pth")
    imgs = [torch.rand(3, 240, 428)]
    with torch.no_grad():
        a, b = ref(imgs)[0], mine(imgs)[0]
    assert len(a["boxes"]) == len(b["boxes"]) > 0
    assert torch.allclose(a["boxes"], b["boxes"], atol=1e-3) and torch.equal(a["labels"], b["labels"])
    if kind == "keypoint":
        assert torch.allclose(a["keypoints"], b["keypoints"], atol=1e-3)


def test_parameter_counts():
    assert nets.n_params(nets.RCNN(2, 17)) == 59_137_258          # Keypoint R-CNN
    assert nets.n_params(nets.RCNN(91)) == 41_755_286             # Faster R-CNN part of Mask R-CNN
    assert nets.n_params(nets.Hiera()) == 50_784_960             # Hiera-B video, without classifier
