"""AI-VAD's scoring (evaluate.py + score_calibration.py), on the per-video attribute arrays.

fit (training videos = normal behaviour):
  velocity    Gaussian mixture, 5 components; unusualness = -log p(x)
  pose        bank of all training poses; unusualness = squared distance to the nearest one
  appearance  bank of all training CLIP embeddings; same 1-NN distance
  calibration velocity: min and 99.9th percentile of the training scores;
              pose / appearance: min and 99.9th pct / max of LEAVE-ONE-VIDEO-OUT 1-NN distances
              (each training video scored against all the OTHER training videos)
score (one video):
  per frame, per attribute: the most unusual object -> calibrated -> sum of the attributes in
  `attributes` -> Gaussian smoothing over time (sigma 7 on ShanghaiTech)
Model 4 options (all off for model 3): per_scene banks, crowd-relative velocity, track smoothing.
The 1-NN search runs in PyTorch (GPU if available) instead of faiss-gpu; it is exact.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

ATTRIBUTES = ("velocity", "pose", "appearance", "relative_velocity")


# ---------------------------------------------------------------- nearest neighbours
@torch.no_grad()
def nn_distance(query: np.ndarray, bank: np.ndarray, q_group=None, b_group=None,
                device="cpu", q_chunk=2048, b_chunk=65536) -> np.ndarray:
    """Squared L2 distance of every query row to its nearest bank row (= faiss IndexFlatL2, k=1).
    With groups: bank rows of the query's own group are ignored (leave-one-video-out).
    Tiled over queries and bank, so memory stays small (q_chunk x b_chunk floats)."""
    if len(query) == 0:
        return np.zeros(0, np.float32)
    b = torch.as_tensor(bank, dtype=torch.float32, device=device)
    b2 = (b * b).sum(1)
    bg = None if b_group is None else torch.as_tensor(b_group, device=device)
    out = np.empty(len(query), np.float32)
    for s in range(0, len(query), q_chunk):
        q = torch.as_tensor(query[s:s + q_chunk], dtype=torch.float32, device=device)
        q2 = (q * q).sum(1, keepdim=True)
        qg = None if bg is None else torch.as_tensor(q_group[s:s + q_chunk], device=device)
        best = torch.full((len(q),), float("inf"), device=device)
        for t in range(0, len(b), b_chunk):
            d = q2 + b2[None, t:t + b_chunk] - 2 * q @ b[t:t + b_chunk].T
            if qg is not None:
                d[qg[:, None] == bg[None, t:t + b_chunk]] = float("inf")
            best = torch.minimum(best, d.min(1).values)
        out[s:s + q_chunk] = best.clamp_min(0).cpu().numpy()
    return out


def gmm_state(g: GaussianMixture) -> dict:
    return {k: torch.as_tensor(getattr(g, k)) for k in ("weights_", "means_", "covariances_", "precisions_cholesky_")}


def gmm_from_state(s: dict) -> GaussianMixture:
    g = GaussianMixture(n_components=len(s["weights_"]), covariance_type="full")
    for k, v in s.items():
        setattr(g, k, v.numpy())
    g.converged_, g.n_iter_, g.lower_bound_ = True, 0, 0.0
    return g


# ---------------------------------------------------------------- per-object attribute values
def relative_velocity(feat: dict) -> np.ndarray:
    """Model 4: each person's motion minus the median motion of the OTHER objects in the same frame
    (in body heights per frame). Walking against / much faster than the crowd -> large."""
    mf, fr = feat["mean_flow"], feat["frame"]
    out = mf.copy()
    for t in np.unique(fr):
        rows = np.flatnonzero(fr == t)
        if len(rows) > 1:
            for r in rows:
                others = rows[rows != r]
                out[r] = mf[r] - np.median(mf[others], 0)
    return out


def scene_of(video: str) -> str:
    return video.split("_")[0]


class AttributeScorer:
    def __init__(self, cfg: dict):
        s = cfg["scoring"]
        self.attributes = list(s["attributes"])
        self.extra = [a for a in s.get("also_calibrate", []) if a not in self.attributes]
        self.k_gmm, self.pct, self.sigma = s["gmm_components"], s["calibration_percentile"], s["smoothing_sigma"]
        self.per_scene, self.track_smooth = s.get("per_scene", False), s.get("track_smoothing", 0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state: dict = {}

    def needed(self):
        return self.attributes + self.extra

    # ------------------------------------------------------------ fit
    def _fit_group(self, feats: dict[str, dict]) -> dict:
        vids = sorted(feats)
        st = {}
        for att in ("velocity", "relative_velocity"):
            if att not in self.needed():
                continue
            X = np.concatenate([feats[v]["velocity"] if att == "velocity" else relative_velocity(feats[v])
                                for v in vids]).astype(np.float64)
            g = GaussianMixture(n_components=self.k_gmm, random_state=0).fit(X)
            tr = -g.score_samples(X)
            st[att] = {"gmm": gmm_state(g), "lo": float(tr.min()), "hi": float(np.percentile(tr, self.pct))}
        for att, key in (("pose", "pose"), ("appearance", "deep")):
            if att not in self.needed():
                continue
            arrs = [feats[v][key] for v in vids if key in feats[v] and len(feats[v][key])]
            if len(arrs) < 2:
                print(f"warning: {att} found in fewer than 2 training videos; it scores 0")
                st[att] = None
                continue
            groups = np.concatenate([np.full(len(feats[v][key]), i) for i, v in enumerate(vids)
                                     if key in feats[v] and len(feats[v][key])])
            bank = np.concatenate(arrs).astype(np.float32)
            d = nn_distance(bank, bank, groups, groups, self.device)    # leave-one-video-out
            d = d[np.isfinite(d)]
            hi = float(d.max()) if att == "appearance" else float(np.percentile(d, self.pct))
            st[att] = {"bank": torch.as_tensor(bank, dtype=torch.float16 if att == "appearance" else torch.float32),
                       "lo": float(d.min()), "hi": hi}
        return st

    def fit(self, feats: dict[str, dict]) -> dict:
        self.state = {"global": self._fit_group(feats)}
        if self.per_scene:
            for sc in sorted({scene_of(v) for v in feats}):
                sub = {v: f for v, f in feats.items() if scene_of(v) == sc}
                if len(sub) >= 2:                                       # LOVO needs 2+ videos
                    self.state[sc] = self._fit_group(sub)
        return {g: {a: (None if s is None else {"lo": s["lo"], "hi": s["hi"]}) for a, s in st.items()}
                for g, st in self.state.items()}

    # ------------------------------------------------------------ score
    def _object_scores(self, st: dict, feat: dict) -> dict:
        out = {}
        for att in st:
            s = st[att]
            if s is None:                                     # attribute never seen in training
                n = len(feat["pose"]) if att == "pose" else len(feat["frame"])
                out[att] = np.zeros(n)
                continue
            if att in ("velocity", "relative_velocity"):
                x = feat["velocity"] if att == "velocity" else relative_velocity(feat)
                raw = -gmm_from_state(s["gmm"]).score_samples(x.astype(np.float64))
            else:
                key = "pose" if att == "pose" else "deep"
                q = feat.get(key, np.zeros((0, 1)))
                raw = nn_distance(q.astype(np.float32), s["bank"].float().numpy(), device=self.device)
            out[att] = (raw - s["lo"]) / max(s["hi"] - s["lo"], 1e-12)
        return out

    def _smooth_tracks(self, feat, rows_score, rows_frame, rows_obj):
        """Average each tracked person's score over their last k frames (model 4)."""
        ids = feat["ids"][rows_obj]
        out = rows_score.copy()
        for tid in np.unique(ids[ids >= 0]):
            r = np.flatnonzero(ids == tid)
            r = r[np.argsort(rows_frame[r])]
            c = np.cumsum(np.insert(rows_score[r], 0, 0))
            k = np.arange(1, len(r) + 1)
            lo = np.maximum(0, k - self.track_smooth)
            out[r] = (c[k] - c[lo]) / (k - lo)
        return out

    def score(self, video: str, feat: dict) -> dict:
        st = self.state.get(scene_of(video), self.state["global"]) if self.per_scene else self.state["global"]
        n = int(feat["n_frames"])
        obj = self._object_scores(st, feat)
        per_att = {}
        for att, s in obj.items():
            if att == "pose":
                fr, objrow = feat["pose_frame"], feat["pose_object"]
            else:
                fr, objrow = feat["frame"], np.arange(len(feat["frame"]))
            if self.track_smooth and len(s) and "ids" in feat:
                s = self._smooth_tracks(feat, s, fr, objrow)
                obj[att] = s
            f = np.full(n, -np.inf)
            np.maximum.at(f, fr, s)
            per_att[att] = np.where(np.isfinite(f), f, 0.0)          # no pose in a frame -> 0
        total = sum(per_att[a] for a in self.attributes)
        smooth = gaussian_filter1d(total, self.sigma) if self.sigma else total   # = AI-VAD gaussian_video
        return {"score": smooth.astype(np.float32), "unsmoothed": total.astype(np.float32),
                "per_attribute": per_att, "objects": obj}

    # ------------------------------------------------------------ save / load
    def state_dict(self) -> dict:
        return {"attributes": self.attributes, "extra": self.extra, "state": self.state,
                "k_gmm": self.k_gmm, "pct": self.pct, "sigma": self.sigma,
                "per_scene": self.per_scene, "track_smooth": self.track_smooth}

    @classmethod
    def from_state(cls, sd: dict, cfg: dict) -> "AttributeScorer":
        s = cls(cfg)
        s.state = sd["state"]
        return s
