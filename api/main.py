"""REST API: upload a video, get back who is loitering / moving against the crowd, which frames
the model finds abnormal, and a result video. Prometheus metrics at /metrics.

    uvicorn api.main:app --host 0.0.0.0 --port 8000        (or: docker compose up)
    open http://localhost:8000/docs                           (try it in the browser)

Endpoints
    GET  /health                 is the service up, which model is the default
    GET  /models                 every model offered: ready or not, parameters, test AUC / EER
    POST /analyze                form field `file` = video (.mp4/.avi/.mov), ?model=...&render=true
    GET  /results/{job_id}/video the result video (H.264: boxes, directions, tags, score curve)
    GET  /metrics                Prometheus metrics
Default model: configs/api.yaml (api.model), or env VAD_MODEL=astt_baseline | astt_improved | mulde |
mulde_improved | none. ?model= picks another one per request (loaded on first use, at most
api.max_models_loaded kept in GPU memory). The web page (app/gradio_app.py) is a client of this API.
Env VAD_MODELS=astt_improved,... offers only those models (the Docker app: model 2 only);
env VAD_MAX_SECONDS=30 analyzes at most the first 30 s of a video (0 / unset: no limit).
"""
from __future__ import annotations

import os
import shutil
import threading
import json
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import AnalyzeResult, ErrorResponse, Health, ModelList

from src.utils import load_config

from src.models import MODELS as TITLES, trained_dir   # the one list of models

MODELS = tuple(m for m in TITLES if m in os.environ.get("VAD_MODELS", ",".join(TITLES)).split(","))
MAX_SECONDS = float(os.environ.get("VAD_MAX_SECONDS") or 0) or None
NAMES = {**TITLES, "none": "Behavior rules only"}


class M:
    """Prometheus metrics (scraped at /metrics, shown in Grafana). Service health: requests,
    errors, latency. Model behaviour: time per step, score distribution, share of abnormal frames,
    events: if these drift, incoming videos no longer look like the training data."""
    REQUESTS = Counter("vad_api_requests_total", "HTTP requests", ["endpoint", "status"])
    LATENCY = Histogram("vad_api_request_seconds", "HTTP request duration", ["endpoint"],
                        buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600))

    STEP_SECONDS = Histogram("vad_pipeline_step_seconds", "Time per pipeline step", ["step"],
                             buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300))
    VIDEOS = Counter("vad_videos_analyzed_total", "Videos analyzed", ["model"])
    FRAMES = Counter("vad_frames_processed_total", "Frames processed", ["model"])
    FRAME_SCORE = Histogram("vad_frame_anomaly_score", "Frame anomaly score (models 1-2: MSE, 3-4: MULDE score)",
                            ["model"], buckets=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10))
    ABNORMAL_RATIO = Histogram("vad_abnormal_frame_ratio", "Share of frames above the threshold per video",
                               ["model"], buckets=(0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1))
    EVENTS = Counter("vad_behavior_events_total", "Behavior events detected", ["event"])
    MODEL = Info("vad_model", "Model being served")
    THRESHOLD = Gauge("vad_model_threshold", "Frame threshold (EER) in use, in the model's score units")


CFG = load_config("configs/api.yaml")["api"]
WORK = Path(CFG["work_dir"])
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
state: dict = {"analyzers": OrderedDict()}    # model name -> Analyzer (most recently used last)
lock = threading.Lock()                      # one video at a time on the GPU


def analyzer(name: str):
    """The Analyzer of one model, loaded on first use; the least recently used is dropped beyond
    api.max_models_loaded. A model that is not trained / evaluated serves the rules only."""
    from src.inference import Analyzer
    pool = state["analyzers"]
    if name not in pool:
        pool[name] = Analyzer(name, allow_missing=True)
        while len(pool) > max(CFG.get("max_models_loaded", 2), 1):
            pool.popitem(last=False)
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pool.move_to_end(name)
    return pool[name]


def model_card(name: str) -> dict:
    """Is this model ready to serve, and how good is it on the 107 ShanghaiTech test videos."""
    if name == "none":
        return {"name": name, "title": NAMES[name], "ready": True}
    ev, tr = Path(f"outputs/evaluation/{name}/metrics.json"), trained_dir(name) / "run.json"
    m = json.loads(ev.read_text()) if ev.exists() else {}
    params = (json.loads(tr.read_text()).get("parameters") or {}).get("Total") if tr.exists() else None
    return {"name": name, "title": NAMES[name], "ready": (trained_dir(name) / "model.pth").exists() and bool(m),
            "parameters": params, **{k: m.get(k) for k in ("auc_raw", "auc_norm", "auc_macro", "eer")}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["default"] = os.environ.get("VAD_MODEL") or CFG["model"] or "none"
    info = analyzer(state["default"]).info()
    M.MODEL.info({k: str(v) for k, v in info.items() if k != "warning"})
    if info["threshold"] is not None:
        M.THRESHOLD.set(info["threshold"])
    WORK.mkdir(parents=True, exist_ok=True)
    yield


DESCRIPTION = """
Upload a video and get back **who** behaves abnormally (loitering, moving against the crowd,
running), **every person's walking direction**, **which frames** the model finds abnormal, and an
**annotated result video**.

1. `GET /models`: the models this server offers, with their test accuracy.
2. `POST /analyze` with the video as form field `file` (optionally `?model=`): the analysis (JSON).
3. `GET /results/{job_id}/video`: the annotated video of that analysis.

Errors always look like `{"error": "<code>", "detail": "<what went wrong>"}` with a matching HTTP
status (400, 404, 413, 415, 422, 500).
"""
TAGS = [{"name": "analysis", "description": "Analyze a video and fetch the result video"},
        {"name": "service", "description": "Health, available models, Prometheus metrics"}]
app = FastAPI(title="DSP391m video anomaly detection", version="3.1", description=DESCRIPTION,
              openapi_tags=TAGS, lifespan=lifespan)


class ApiError(Exception):
    """An error with a stable code; returned as ErrorResponse with the given HTTP status."""

    def __init__(self, status: int, error: str, detail: str):
        self.status, self.error, self.detail = status, error, detail


def error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


@app.exception_handler(ApiError)
async def api_error(request: Request, e: ApiError):
    return error(e.status, e.error, e.detail)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, e: RequestValidationError):
    where = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
    return error(422, "invalid_request", where or "invalid request")


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, e: StarletteHTTPException):
    return error(e.status_code, {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error"), str(e.detail))


@app.exception_handler(Exception)
async def unexpected(request: Request, e: Exception):
    traceback.print_exc()
    return error(500, "internal_error", f"{type(e).__name__}: {e}")


ERRORS = {400: {"model": ErrorResponse, "description": "Unknown model or empty file"},
          413: {"model": ErrorResponse, "description": "File too large"},
          415: {"model": ErrorResponse, "description": "Not a video file"},
          422: {"model": ErrorResponse, "description": "Missing file, or a video that can't be read / analyzed"},
          500: {"model": ErrorResponse, "description": "Unexpected server error"}}


@app.middleware("http")
async def count_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    t0, status = time.time(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        ep = route.path if route else "other"
        M.REQUESTS.labels(ep, str(status)).inc()
        M.LATENCY.labels(ep).observe(time.time() - t0)


@app.get("/health", response_model=Health, tags=["service"], summary="Is the service up, and how is it set up")
def health():
    return {"status": "ok", **analyzer(state["default"]).info(), "max_seconds": MAX_SECONDS,
            "light_tracking": bool(os.environ.get("VAD_DETECTOR_SIZE") or int(os.environ.get("VAD_DETECT_EVERY") or 1) > 1)}


@app.get("/models", response_model=ModelList, tags=["service"], summary="The models this server offers")
def models():
    return {"default": state["default"], "models": [model_card(n) for n in (*MODELS, "none")]}


def clean_old_jobs():
    """Keep the newest api.keep_jobs result folders, so the disk does not fill up."""
    jobs = sorted((d for d in WORK.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    for d in jobs[:max(len(jobs) - CFG.get("keep_jobs", 50), 0)]:
        shutil.rmtree(d, ignore_errors=True)


@app.post("/analyze", response_model=AnalyzeResult, responses=ERRORS, tags=["analysis"],
          summary="Analyze one video: abnormal people, directions, abnormal frames, result video")
def analyze(file: UploadFile = File(..., description="The video: .mp4, .avi, .mov, .mkv or .m4v"),
            model: str | None = Query(None, description="Model name from GET /models (default: the server's default)",
                                      examples=["astt_improved"]),
            render: bool = Query(True, description="Also draw the annotated result video (false: JSON only, faster)")):
    """Runs people tracking + the behavior rules (LOITERING, COUNTER_FLOW, TOO_FAST) and the chosen
    model on the uploaded video. One video at a time: a second request waits for the first."""
    model = model or state["default"]
    if model not in (*MODELS, "none"):
        raise ApiError(400, "unknown_model", f"unknown model {model!r}; one of {[*MODELS, 'none']}")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise ApiError(415, "unsupported_file_type", f"unsupported file type {ext!r}; send one of {sorted(VIDEO_EXTS)}")
    WORK.mkdir(parents=True, exist_ok=True)
    clean_old_jobs()
    job = uuid.uuid4().hex[:12]
    work = WORK / job
    work.mkdir(parents=True)
    src = work / f"input{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if src.stat().st_size == 0:
        shutil.rmtree(work)
        raise ApiError(400, "empty_file", "the uploaded file is empty")
    if src.stat().st_size > CFG["max_upload_mb"] * 1024 * 1024:
        shutil.rmtree(work)
        raise ApiError(413, "file_too_large", f"file larger than {CFG['max_upload_mb']} MB")
    try:
        with lock:
            r = analyzer(model).run(src, work, draw=render, max_seconds=MAX_SECONDS)
    except Exception as e:
        traceback.print_exc()                                         # the full error goes to the API log
        raise ApiError(422, "cannot_analyze", f"could not analyze the video: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(work / "frames", ignore_errors=True)            # keep only the results
        src.unlink(missing_ok=True)

    model = r["model"]
    for step, sec in r["timings"].items():
        if step != "total_s":
            M.STEP_SECONDS.labels(step[:-2]).observe(sec)
    M.VIDEOS.labels(model).inc()
    M.FRAMES.labels(model).inc(r["frames"])
    for s in r["frame_scores"] or []:
        if s is not None:
            M.FRAME_SCORE.labels(model).observe(s)
    if r["frame_scores"] is not None:
        M.ABNORMAL_RATIO.labels(model).observe(r["abnormal_ratio"])
    for name, n in r["event_counts"].items():
        M.EVENTS.labels(name).inc(n)
    r["job_id"] = job
    r["result_video"] = f"/results/{job}/video" if render else None
    r["video"] = file.filename
    return r


@app.get("/results/{job_id}/video", tags=["analysis"], summary="The annotated video of an analysis",
         response_class=FileResponse, responses={200: {"content": {"video/mp4": {}}, "description": "H.264 video"},
                                                  404: {"model": ErrorResponse, "description": "No such job"}})
def result_video(job_id: str):
    f = WORK / Path(job_id).name / "result.mp4"
    if not f.exists():
        raise ApiError(404, "not_found", f"no result video for job {job_id!r} (old results are deleted)")
    return FileResponse(f, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/metrics", tags=["service"], summary="Prometheus metrics", response_class=Response)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
