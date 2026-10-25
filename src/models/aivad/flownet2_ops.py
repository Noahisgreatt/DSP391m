"""Pure-PyTorch versions of FlowNet2's three CUDA extensions, so AI-VAD's FlowNet2 code runs
UNCHANGED on a current PyTorch (no compiling, works on CPU too, exportable to ONNX/TensorRT).

AI-VAD's files do `import correlation_cuda`, `import resample2d_cuda`, `import channelnorm_cuda`
(modules normally compiled from .cu files). `install()` registers these three modules; each
`forward` fills the output tensor exactly like the CUDA kernel does
(third_party/AIVAD/pre_processing/flownet_networks/*_package/*_kernel.cu). Inference only.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


def correlation_forward(input1, input2, rbot1, rbot2, output, pad_size, kernel_size,
                        max_displacement, stride1, stride2, corr_multiply):
    """correlation_cuda_kernel.cu: out[b, (tj+r)*(2r+1) + (ti+r), y, x] =
    sum_c in1[c, y, x] * in2[c, y + tj*stride2, x + ti*stride2] / (k*k*C), zero padding."""
    assert kernel_size == 1 and stride1 == 1 and pad_size == max_displacement, "FlowNetC settings"
    b, c, h, w = input1.shape
    r = max_displacement // stride2
    p2 = F.pad(input2, (pad_size, pad_size, pad_size, pad_size))
    out = input1.new_empty(b, (2 * r + 1) ** 2, h, w)
    k = 0
    for tj in range(-r, r + 1):
        for ti in range(-r, r + 1):
            dy, dx = pad_size + tj * stride2, pad_size + ti * stride2
            out[:, k] = (input1 * p2[:, :, dy:dy + h, dx:dx + w]).sum(1) / c
            k += 1
    output.resize_(out.shape).copy_(out)
    return 1


def resample2d_forward(input1, input2, output, kernel_size, bilinear):
    """resample2d_kernel.cu: warp input1 by flow input2 with bilinear weights; the four
    neighbour indices are clamped to the image (alpha / beta from the unclamped position)."""
    assert kernel_size == 1 and bilinear, "FlowNet2 uses Resample2d(kernel_size=1, bilinear)"
    b, c, h, w = input1.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=input1.device, dtype=input1.dtype),
                            torch.arange(w, device=input1.device, dtype=input1.dtype), indexing="ij")
    xf, yf = xs + input2[:, 0], ys + input2[:, 1]                       # (b, h, w)
    x0, y0 = torch.floor(xf), torch.floor(yf)
    alpha, beta = (xf - x0).unsqueeze(1), (yf - y0).unsqueeze(1)
    xl, xr = x0.clamp(0, w - 1).long(), (x0 + 1).clamp(0, w - 1).long()
    yt, yb = y0.clamp(0, h - 1).long(), (y0 + 1).clamp(0, h - 1).long()
    flat = input1.reshape(b, c, h * w)
    g = lambda yy, xx: flat.gather(2, (yy * w + xx).reshape(b, 1, h * w).expand(b, c, h * w)).reshape(b, c, h, w)
    out = ((1 - alpha) * (1 - beta) * g(yt, xl) + alpha * (1 - beta) * g(yt, xr)
           + (1 - alpha) * beta * g(yb, xl) + alpha * beta * g(yb, xr))
    output.copy_(out)
    return 1


def channelnorm_forward(input1, output, norm_deg):
    """channelnorm_kernel.cu: per-pixel L2 norm over channels."""
    output.copy_(input1.pow(norm_deg).sum(1, keepdim=True).pow(1.0 / norm_deg))
    return 1


def _no_backward(*_):
    raise NotImplementedError("inference only")


def install() -> None:
    """Register the three modules before AI-VAD's FlowNet2 code is imported."""
    for name, fwd in (("correlation_cuda", correlation_forward),
                      ("resample2d_cuda", resample2d_forward),
                      ("channelnorm_cuda", channelnorm_forward)):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.forward, m.backward = fwd, _no_backward
            sys.modules[name] = m
