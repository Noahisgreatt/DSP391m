"""Step 2, MODELS.   astt.py = models 1, 2   aivad.py = models 3, 4   nets.py = the networks."""


def scorer(cfg, checkpoint, device, smoke=False):
    """Scores one video with any of the 4 models (the config decides which)."""
    if cfg["model"].get("family") == "aivad":
        from src.models.aivad import Scorer
        return Scorer(cfg, checkpoint, device, smoke=smoke)
    from src.models.astt import Scorer
    return Scorer(cfg, checkpoint, device)
