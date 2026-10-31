"""Models 3 and 4: MULDE (Micorek et al., "Multiscale Log-Density Estimation via Denoising Score
Matching for Video Anomaly Detection", CVPR 2024), frame-centric.

1 FEATURES  each frame t -> the clip of 16 frames around it (every `step`-th frame) -> Hiera-B, a
            video transformer pretrained on Kinetics-400 (our code: src/models/nets.py) -> 768 numbers.
            Standardised with the training statistics.
2 DENSITY   a small network f(x, sigma) learns "how normal is x" at every noise level sigma by
            denoising score matching on the NORMAL training videos only (the only thing trained):
              loss = sigma^2 * || grad_x f(x~, sigma) - (x~ - x) / sigma^2 ||^2  +  beta * f(x, sigma)^2
            x~ = x + noise of size sigma, sigma log-uniform in [sigma_low, sigma_high]. So f is low
            where normal features are dense and grows away from them.
3 SCORE     the 16 values f(x, sigma_1 ... sigma_16) of a frame -> negative log-likelihood under a
            Gaussian mixture fitted on the training frames -> Gaussian smoothing over time.

                model 3 mulde (paper, Hiera-B)          model 4 mulde_improved (ours)
    clips       16 x every 4th frame                     + 16 x every 2nd frame (fast motion), joined
    density     2 x 4096 hidden units                    2 x 2048 (lighter)
    mixture     one for all cameras                      one per camera (13 cameras look different)

    python -m src.train    --config configs/mulde.yaml   (features of the training videos + fit)
    python -m src.evaluate --config configs/mulde.yaml
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

from src.models import nets
from src.utils import config_hash, frames_dir, git_commit, list_frames, out_dir

MEAN, STD = 0.45, 0.225                    # Kinetics normalisation of Hiera's video models


# ====================================================================== 1. features
class Features:
    """Hiera-B feature of every frame, for each clip length in `frame_steps`, cached per video
    (data/features/hiera_b16/<split>/<video>_s<step>_e<stride>.npy, shared by models 3 and 4)."""

    def __init__(self, cfg, device, weights=True):
        self.m, self.device = cfg["model"], device
        self.net = nets.Hiera()
        if weights:
            self.net.load_pretrained()
        self.net = self.net.to(device).eval()
        self.root = Path(cfg["paths"]["features"]) / "hiera_b16"

    @torch.no_grad()
    def clips(self, frames, step, centres):
        """frames (n, 224, 224, 3) uint8 RGB -> one feature per centre frame (len(centres), 768)."""
        offsets = step * (np.arange(self.m["clip_frames"]) - self.m["clip_frames"] // 2)
        out = []
        for s in range(0, len(centres), self.m["batch_clips"]):
            idx = np.clip(centres[s:s + self.m["batch_clips"], None] + offsets, 0, len(frames) - 1)
            x = torch.from_numpy(frames[idx]).to(self.device).permute(0, 4, 1, 2, 3).float().div_(255).sub_(MEAN).div_(STD)
            with nets.autocast(self.device):
                out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out)

    def video(self, frame_dir, vid, split=None, stride=1, max_frames=None):
        """(frames, features) of one video: features for frames 0, stride, 2*stride, ...;
        steps joined side by side (768 per step). split=None: no cache (demo / API)."""
        paths = list_frames(frame_dir)[:max_frames]
        centres = np.arange(0, len(paths), stride)
        files = [self.root / split / f"{vid}_s{s}_e{stride}_n{len(paths)}.npy" if split else None
                 for s in self.m["frame_steps"]]
        cached = [load_cache(f, len(centres)) for f in files]
        if all(c is not None for c in cached):
            return centres, np.concatenate(cached, 1)
        frames = np.stack([cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (224, 224), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB) for p in paths])
        feats = []
        for s, f, c in zip(self.m["frame_steps"], files, cached):
            feats.append(c if c is not None else self.clips(frames, s, centres))
            if f is not None and c is None:                  # write whole or not at all (power cut)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f.with_suffix(".tmp.npy"), feats[-1].astype(np.float16))
                f.with_suffix(".tmp.npy").replace(f)
        return centres, np.concatenate(feats, 1)


def load_cache(f, n):
    """A cached feature file, or None if it is missing or broken (then it is computed again)."""
    if f is None or not f.exists():
        return None
    try:
        x = np.load(f).astype(np.float32)
        if len(x) == n:
            return x
    except Exception:
        pass
    print(f"broken cache file, computing it again: {f.name}")
    f.unlink()
    return None


# ====================================================================== 2. density network
class DensityNet(nn.Module):
    """f(x, sigma): x and log(sigma) -> 2 hidden GELU layers -> one number (twice differentiable)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, sigma):
        return self.net(torch.cat([x, sigma.log()[:, None]], 1)).squeeze(1)


def train_density(X, d, hidden, device, iterations, log=None):
    """Denoising score matching over noise levels (MULDE Eq. 6)."""
    net = DensityNet(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=d["lr"], betas=tuple(d["adam_betas"]))
    X = torch.as_tensor(X, device=device)
    lo, hi = np.log(d["sigma_low"]), np.log(d["sigma_high"])
    for it in range(1, iterations + 1):
        x = X[torch.randint(len(X), (d["batch_size"],), device=device)]
        sigma = torch.exp(torch.empty(len(x), device=device).uniform_(lo, hi))
        noise = torch.randn_like(x) * sigma[:, None]
        xt = (x + noise).requires_grad_(True)
        grad = torch.autograd.grad(net(xt, sigma).sum(), xt, create_graph=True)[0]
        loss = (sigma[:, None] * grad - noise / sigma[:, None]).pow(2).sum(1).mean() + d["beta"] * net(x, sigma).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log is not None and (it % 200 == 0 or it == iterations):
            log.append(f"{it},{loss.item():.6f}")
            print(f"density step {it}/{iterations} loss {loss.item():.4f}")
    return net.eval()


@torch.no_grad()
def levels(net, X, sigmas, device):
    """(n, d) standardised features -> (n, L) values f(x, sigma_l)."""
    out = []
    for s in range(0, len(X), 8192):
        x = torch.as_tensor(X[s:s + 8192], device=device)
        out.append(torch.stack([net(x, torch.full((len(x),), float(sg), device=device)) for sg in sigmas], 1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fit_mixture(F, k):
    return GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(F)


def mixture_state(g):
    return {a: getattr(g, a) for a in ("weights_", "means_", "covariances_", "precisions_cholesky_")}


def mixture(state):
    g = GaussianMixture(n_components=len(state["weights_"]), covariance_type="full")
    for a, v in state.items():
        setattr(g, a, v)
    return g


# ====================================================================== 3. "training", scoring, size
def parameters(cfg):
    dim, h = 768 * len(cfg["model"]["frame_steps"]), cfg["model"]["hidden"]
    density = (dim + 1) * h + h + h * h + h + h + 1
    return {"Hiera-B (features)": nets.n_params(nets.Hiera()), "density network": density,
            "Total": nets.n_params(nets.Hiera()) + density}


def fit(cfg, smoke=False, fresh=False, max_hours=None):
    """Features of the normal training videos -> density network -> mixture(s) -> outputs/training/<model>/."""
    from src.data import load_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, d, s = cfg["model"], cfg["density"], cfg["scoring"]
    out = out_dir(cfg, "training", smoke)
    if fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    chash = config_hash({k: cfg[k] for k in ("model", "density", "scoring")})
    done = [json.loads(f.read_text()) if f.exists() else {} for f in (out / "metrics.json", out / "run.json")]
    if (out / "model.pth").exists() and done[0].get("finished") and done[1].get("config_hash") == chash:
        return print(f"{out}/model.pth is already trained with these settings (e.g. on another computer): "
                     "nothing to do (--fresh to train again)")
    vids, t0 = load_split(cfg)["train"][: 2 if smoke else None], time.time()
    ex = Features(cfg, device)
    feats = {}
    for k, v in enumerate(vids, 1):                          # cached per video: a stopped run continues
        if max_hours and time.time() - t0 > 3600 * max_hours:
            return print(f"stopped after {max_hours} h at video {k}/{len(vids)}: run again to continue")
        feats[v] = ex.video(frames_dir(cfg, "shanghaitech") / v, v, "train", m["train_stride"], 30 if smoke else None)[1]
        print(f"features [{k}/{len(vids)}] {v}: {len(feats[v])} clips ({time.time() - t0:.0f}s)")
    X = np.concatenate([feats[v] for v in vids])
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xs = ((X - mean) / std).astype(np.float32)
    log = ["step,loss"]
    net = train_density(Xs, d, m["hidden"], device, 50 if smoke else d["iterations"], log)
    sigmas = np.geomspace(d["sigma_low"], d["sigma_high"], d["levels"])
    F = levels(net, Xs, sigmas, device)
    k = 1 if smoke else s["gmm_components"]
    gmms = {"all": mixture_state(fit_mixture(F, k))}
    if s["per_scene"]:                                       # model 4: one mixture per camera
        scene = np.concatenate([np.full(len(feats[v]), v[:2]) for v in vids])
        gmms.update({c: mixture_state(fit_mixture(F[scene == c], k)) for c in np.unique(scene) if (scene == c).sum() >= 50 * k})
    torch.save({"net": net.state_dict(), "mean": mean, "std": std, "sigmas": sigmas, "gmms": gmms}, out / "model.pth")
    params = parameters(cfg)
    (out / "log.csv").write_text("\n".join(log) + "\n")
    (out / "run.json").write_text(json.dumps({
        "model": m["name"], "config_hash": chash,
        "git_commit": git_commit(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device), "parameters": params, "train_videos": len(vids), "config": cfg}, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps({
        "finished": True, "train_videos": len(vids), "train_clips": len(X), "mixtures": len(gmms),
        "final_loss": float(log[-1].split(",")[1]), "minutes": round((time.time() - t0) / 60, 1),
        "parameters_total": params["Total"]}, indent=2))
    print(f"done: {out}/model.pth")


class Scorer:
    """Score one video with a fitted model 3 / 4 (used by evaluate, demo, API)."""
    kind = "score"

    def __init__(self, cfg, checkpoint, device, smoke=False):
        self.cfg, self.device, self.smoke = cfg, device, smoke
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)     # our own file
        dim = 768 * len(cfg["model"]["frame_steps"])
        self.net = DensityNet(dim, cfg["model"]["hidden"]).to(device).eval()
        self.net.load_state_dict(ck["net"])
        self.mean, self.std, self.sigmas = ck["mean"], ck["std"], ck["sigmas"]
        self.gmms = {c: mixture(g) for c, g in ck["gmms"].items()}
        self.features = None

    def __call__(self, video_dir, vid, details=False, cache=True, **_):
        if self.features is None:
            self.features = Features(self.cfg, self.device)
        n = len(list_frames(video_dir)[: 30 if self.smoke else None])
        _, X = self.features.video(video_dir, vid, "test" if cache else None, 1, n)
        F = levels(self.net, ((X - self.mean) / self.std).astype(np.float32), self.sigmas, self.device)
        score = -self.gmms.get(vid[:2], self.gmms["all"]).score_samples(F)
        sigma = self.cfg["scoring"]["smoothing_sigma"]
        score = (gaussian_filter1d(score, sigma) if sigma else score).astype(np.float32)
        return (score, {"levels": F}) if details else score
