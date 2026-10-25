"""Prometheus metrics exposed by the API at /metrics (scraped by Prometheus, shown in Grafana).

Service health:   requests, errors, latency, requests in progress
Model behaviour:  time per pipeline step, frame anomaly scores, share of abnormal frames,
                  behavior events by type, people per video. A shift in these distributions over
                  time (e.g. far more abnormal frames than usual) is the sign of data drift.
"""
from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS = Counter("vad_api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("vad_api_request_seconds", "HTTP request duration", ["endpoint"],
                    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600))
IN_PROGRESS = Gauge("vad_api_requests_in_progress", "Requests being processed")

STEP_SECONDS = Histogram("vad_pipeline_step_seconds", "Time per pipeline step", ["step"],
                         buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300))
VIDEOS = Counter("vad_videos_analyzed_total", "Videos analyzed", ["model"])
FRAMES = Counter("vad_frames_processed_total", "Frames processed", ["model"])
FRAME_SCORE = Histogram("vad_frame_anomaly_score", "Frame anomaly score (models 1-2: MSE, 3-4: AI-VAD score)",
                        ["model"], buckets=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10))
ABNORMAL_RATIO = Histogram("vad_abnormal_frame_ratio", "Share of frames above the threshold per video",
                           ["model"], buckets=(0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1))
EVENTS = Counter("vad_behavior_events_total", "Behavior events detected", ["event"])
PEOPLE = Histogram("vad_people_per_video", "People tracked per video",
                   buckets=(0, 1, 2, 5, 10, 20, 50, 100, 200))
MODEL = Info("vad_model", "Model being served")
THRESHOLD = Gauge("vad_model_threshold", "Frame threshold (EER) in use, in the model's score units")
UNUSUAL = Counter("vad_unusual_objects_total", "Objects models 3/4 found unusual, by attribute", ["attribute"])
