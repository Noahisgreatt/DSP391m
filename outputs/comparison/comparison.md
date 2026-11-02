# Comparison on 107 test videos, 40,791 frames (17,326 abnormal)

| Method | Parameters | AUC micro | AUC per-video norm | AUC macro | EER | Frames/s |
|---|---|---|---|---|---|---|
| Model 1: ASTT (thesis, unchanged) | 150.6M | 47.0% | 63.9% | 67.7% | 40.2% | 29.2 |
| Model 2: ASTT improved (ours) | 100.4M | 57.6% | 72.6% | 81.0% | 33.0% | 66.3 |
| Model 3: MULDE (Micorek et al.) | 70.7M | 66.7% | 76.2% | 75.1% | 29.9% | 4.6 |
| Model 4: MULDE improved (ours) | 58.1M | 66.3% | 71.4% | 73.5% | 33.5% | 15.3 |
| Behavior rules (no training) | 0M | 59.9% | 59.9% | - | 40.1% | - |
