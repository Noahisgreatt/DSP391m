"""Step 2, MODELS.   astt.py = models 1, 2, 5, 6   mulde.py = models 3, 4   nets.py = the networks.

MODELS is the ONE list of models: evaluation, comparison, versions, verify, API and web page read it.
A new model = a configs/<name>.yaml + a line here (+ its train / evaluate entries in dvc.yaml)."""
from pathlib import Path

MODELS = {
    "astt_baseline": "Model 1: ASTT (thesis, unchanged)",
    "astt_improved": "Model 2: ASTT improved (ours)",
    "mulde": "Model 3: MULDE (Micorek et al.)",
    "mulde_improved": "Model 4: MULDE improved (ours)",
    "astt_lite": "Model 5: ASTT lite (ours, 7-block ViT)",
    "astt_lite_smooth": "Model 6: ASTT lite + Gaussian smoothing (ours)",
}


def weights_of(name: str) -> str:
    """Which model's trained weights a model uses: its own, or another's (model.weights_from: model 6
    is model 5's network scored differently, so it has nothing to train)."""
    import yaml
    cfg = Path(f"configs/{name}.yaml")
    m = (yaml.safe_load(cfg.read_text()) or {}).get("model", {}) if cfg.exists() else {}
    return m.get("weights_from") or name


def trained_dir(name: str, outputs: str = "outputs") -> Path:
    """outputs/training/<the model whose weights it uses>/ (model.pth, run.json)."""
    return Path(outputs) / "training" / weights_of(name)


def scorer(cfg, checkpoint, device, smoke=False):
    """Scores one video with any model (the config decides which)."""
    if cfg["model"].get("family") == "mulde":
        from src.models.mulde import Scorer
        return Scorer(cfg, checkpoint, device, smoke=smoke)
    from src.models.astt import Scorer
    return Scorer(cfg, checkpoint, device)
