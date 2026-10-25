"""REST API: upload a video, get back who is loitering / moving against the crowd, which frames
the model finds abnormal, and a result video. Prometheus metrics at /metrics.

    uvicorn api.main:app --host 0.0.0.0 --port 8000        (or: docker compose up)
    open http://localhost:8000/docs                           (try it in the browser)

Endpoints
    GET  /health                 is the service up, which model is loaded
    POST /analyze                form field `file` = video (.mp4/.avi/.mov), ?render=true
    GET  /results/{job_id}       the JSON result again
    GET  /results/{job_id}/video the result video (boxes, tags, heat-map, score curve)
    GET  /metrics                Prometheus metrics
Which model is served: configs/api.yaml (api.model), or env
VAD_MODEL=astt_baseline | astt_improved | aivad | aivad_lite | none.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics as M
from src.config import load_config

CFG = load_config("configs/api.yaml")["api"]
WORK = Path(CFG["work_dir"])
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
state: dict = {}
lock = threading.Lock()                      # one video at a time on the GPU


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.inference.analyze import Analyzer
    state["analyzer"] = Analyzer(os.environ.get("VAD_MODEL") or CFG["model"])
    info = state["analyzer"].info()
    M.MODEL.info({k: str(v) for k, v in info.items()})
    if info["threshold"] is not None:
        M.THRESHOLD.set(info["threshold"])
    WORK.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="DSP391m video anomaly detection", version="2.0", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    t0 = time.time()
    M.IN_PROGRESS.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        M.IN_PROGRESS.dec()
        route = request.scope.get("route")
        ep = route.path if route else "other"
        M.REQUESTS.labels(ep, str(status)).inc()
        M.LATENCY.labels(ep).observe(time.time() - t0)


@app.get("/health")
def health():
    return {"status": "ok", **state["analyzer"].info()}


@app.post("/analyze")
def analyze(file: UploadFile = File(...), render: bool = True):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported file type {ext!r}; send one of {sorted(VIDEO_EXTS)}")
    job = uuid.uuid4().hex[:12]
    work = WORK / job
    work.mkdir(parents=True)
    src = work / f"input{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if src.stat().st_size > CFG["max_upload_mb"] * 1024 * 1024:
        shutil.rmtree(work)
        raise HTTPException(413, f"file larger than {CFG['max_upload_mb']} MB")
    try:
        with lock:
            r = state["analyzer"].run(src, work, render=render)
    except Exception as e:
        raise HTTPException(422, f"could not analyze the video: {e}")
    finally:
        shutil.rmtree(work / "frames", ignore_errors=True)            # keep only the results

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
    for o in r["unusual_objects"]:
        M.UNUSUAL.labels(o["attribute"]).inc()
    for name, n in r["event_counts"].items():
        M.EVENTS.labels(name).inc(n)
    M.PEOPLE.observe(r["people_tracked"])
    r["job_id"] = job
    r["result_video"] = f"/results/{job}/video" if render else None
    r["video"] = file.filename
    return r


@app.get("/results/{job_id}")
def result(job_id: str):
    f = WORK / Path(job_id).name / "result.json"
    if not f.exists():
        raise HTTPException(404, "unknown job")
    return Response(f.read_text(), media_type="application/json")


@app.get("/results/{job_id}/video")
def result_video(job_id: str):
    f = WORK / Path(job_id).name / "result.mp4"
    if not f.exists():
        raise HTTPException(404, "no result video for this job")
    return FileResponse(f, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
