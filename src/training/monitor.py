"""Live training metrics for Prometheus / Grafana.

train.py serves http://localhost:<port>/metrics (configs/api.yaml monitoring.training_metrics_port).
Prometheus (docker compose up) scrapes it, and the Grafana dashboard "Training" row shows loss,
learning rate, speed, progress and GPU memory while the model trains.
"""
from __future__ import annotations

import torch

try:
    from prometheus_client import Gauge, start_http_server
except ImportError:                                          # monitoring is optional
    Gauge = start_http_server = None


class TrainingMonitor:
    def __init__(self, model_name: str, port: int):
        self.on = bool(port) and Gauge is not None
        if not self.on:
            return
        try:
            start_http_server(port)
        except OSError as e:
            print(f"monitoring off: port {port} busy ({e})")
            self.on = False
            return
        print(f"monitoring: http://localhost:{port}/metrics")
        lab = ["model"]
        g = lambda n, h: Gauge(n, h, lab).labels(model_name)
        self.loss = g("vad_train_loss", "mean training loss of the current epoch")
        self.lr = g("vad_train_learning_rate", "learning rate (new layers)")
        self.speed = g("vad_train_iterations_per_second", "training batches per second")
        self.epoch = g("vad_train_epoch", "current epoch")
        self.progress = g("vad_train_progress_ratio", "share of all training steps done (0..1)")
        self.eta = g("vad_train_eta_seconds", "estimated seconds left in the run")
        self.val_auc = g("vad_train_val_auc", "validation frame AUC (raw) of the last epoch")
        self.gpu = g("vad_train_gpu_memory_bytes", "GPU memory allocated by PyTorch")

    def update(self, *, loss, lr, speed, epoch, step, total_steps):
        if not self.on:
            return
        self.loss.set(loss); self.lr.set(lr); self.speed.set(speed); self.epoch.set(epoch)
        self.progress.set(step / max(total_steps, 1))
        self.eta.set((total_steps - step) / max(speed, 1e-9))
        if torch.cuda.is_available():
            self.gpu.set(torch.cuda.memory_allocated())

    def validation(self, auc):
        if self.on and auc == auc:                           # skip NaN
            self.val_auc.set(auc)
