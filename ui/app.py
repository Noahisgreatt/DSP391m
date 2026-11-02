"""Web page for everyone: drop in a video, get back the annotated video, WHO behaves abnormally
(loitering, counter-flow, too fast), WHERE each person walks, and WHEN the model finds the scene
abnormal. It only talks to the FastAPI service (api/main.py) over HTTP.

    docker compose up -d        -> http://localhost:7860
    python -m ui.app            -> http://localhost:7860   (API_URL, default http://localhost:8000)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 3600                                           # a long video can take minutes
OUT = Path(tempfile.gettempdir()) / "dsp391m_ui"
EVENTS = {"LOITERING": "Loitering", "COUNTER_FLOW": "Counter-flow", "TOO_FAST": "Too fast"}
LEGEND = """**How to read the result video**
- Every person has a **box** and an **arrow** = walking direction (colour legend top-left); a dot = standing still.
- **Red box** `COUNTER-FLOW` = walks against the crowd, `TOO FAST` = runs; **yellow box** `LOITERING` = stays in one place.
- Bottom: the model's **anomaly score** over time; yellow line = threshold. Above it: `ABNORMAL FRAME`
  (models 1-2 also paint the abnormal **region** red)."""


def models():
    """Dropdown entries from GET /models: name, test AUC, or 'not trained yet'."""
    try:
        j = requests.get(f"{API}/models", timeout=10).json()
    except (requests.RequestException, ValueError):
        return [("Behavior rules only (API not reachable yet)", "none")], "none"
    out = []
    for m in j["models"]:
        label = m["title"]
        if m["name"] != "none":
            label += (f"  ·  {m['parameters'] / 1e6:.1f}M params  ·  test AUC {m['auc_macro'] * 100:.1f}%"
                      if m["ready"] and m.get("auc_macro") is not None else "  ·  not trained yet")
        out.append((label, m["name"]))
    return out, j["default"]


def summary(r: dict) -> str:
    dur = r["frames"] / (r["fps"] or 25)
    bad_people = sorted({e["track_id"] for e in r["events"]})
    lines = [f"### {len(bad_people)} of {r['people_tracked']} people behave abnormally",
             f"Video: **{r['video']}** · {r['frames']} frames · {dur:.1f} s · model: **{r['model']}** · "
             f"analyzed in {r['timings']['total_s']:.0f} s"]
    for tid in bad_people:
        ev = [e for e in r["events"] if e["track_id"] == tid]
        lines.append(f"- **Person #{tid}**: " + "; ".join(f"{EVENTS.get(e['event'], e['event'])} "
                                                         f"{e['start_s']:.1f}-{e['end_s']:.1f} s" for e in ev))
    if r["frame_scores"] is not None:
        lines.append(f"\nThe model marks **{r['abnormal_ratio'] * 100:.1f}%** of the frames abnormal "
                     f"({len(r['abnormal_frames'])} frames).")
    if r.get("warning"):
        lines.append(f"\n> ⚠️ {r['warning']}")
    return "\n".join(lines)


def tables(r: dict):
    events = pd.DataFrame([{"Person": f"#{e['track_id']}", "Behavior": EVENTS.get(e["event"], e["event"]),
                            "From (s)": e["start_s"], "To (s)": e["end_s"], "Why": e["reason"]} for e in r["events"]],
                          columns=["Person", "Behavior", "From (s)", "To (s)", "Why"])
    people = pd.DataFrame([{"Person": f"#{p['track_id']}", "Seen (s)": f"{p['from_s']:.1f}-{p['to_s']:.1f}",
                            "Main direction": p["direction"], "Moving": f"{p['moving_share'] * 100:.0f}%",
                            "Abnormal": ", ".join(EVENTS.get(e, e) for e in p["events"]) or "-"} for p in r.get("people", [])],
                          columns=["Person", "Seen (s)", "Main direction", "Moving", "Abnormal"])
    fps, scores = r["fps"] or 25, r["frame_scores"] or []
    curve = pd.DataFrame([{"time (s)": i / fps, "value": s, "line": "anomaly score"} for i, s in enumerate(scores) if s is not None]
                         + ([{"time (s)": i / fps, "value": r["threshold"], "line": "threshold"} for i in (0, len(scores) - 1)]
                            if scores and r.get("threshold") is not None else []),
                         columns=["time (s)", "value", "line"])
    return events, people, curve


def analyze(video, model, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a video first.")
    progress(0.05, desc="Uploading and analyzing (people tracking, behavior rules, model)...")
    try:
        with open(video, "rb") as f:
            res = requests.post(f"{API}/analyze", params={"model": model}, timeout=TIMEOUT,
                                files={"file": (Path(video).name, f, "application/octet-stream")})
    except requests.RequestException as e:
        raise gr.Error(f"The API at {API} is not reachable: {e}")
    if res.status_code != 200:
        try:
            msg = res.json().get("detail", res.text)
        except ValueError:
            msg = res.text
        raise gr.Error(f"Analysis failed ({res.status_code}): {msg}")
    r = res.json()
    progress(0.9, desc="Downloading the result video...")
    job = OUT / r["job_id"]
    job.mkdir(parents=True, exist_ok=True)
    out_video = None
    if r.get("result_video"):
        out_video = job / "result.mp4"
        out_video.write_bytes(requests.get(API + r["result_video"], timeout=TIMEOUT).content)
    (job / "result.json").write_text(json.dumps(r, indent=2))
    return (str(out_video) if out_video else None, summary(r), *tables(r), str(job / "result.json"))


def build() -> gr.Blocks:
    choices, default = models()
    with gr.Blocks(title="DSP391m - Video anomaly detection") as page:
        gr.Markdown("# DSP391m · Video anomaly detection\nUpload a surveillance video: see where every person "
                    "walks, who is **loitering**, moving **against the crowd** or **running**, and when the scene is abnormal.")
        with gr.Row():
            with gr.Column(scale=2):
                video_in = gr.Video(label="Your video (.mp4 / .avi / .mov)", sources=["upload"], format="mp4")  # .avi -> .mp4 to preview
                model = gr.Dropdown(choices, value=default, label="Model")
                go = gr.Button("Analyze", variant="primary")
                gr.Markdown(LEGEND)
            with gr.Column(scale=3):
                video_out = gr.Video(label="Result", interactive=False)
                info = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("Abnormal people"):
                events = gr.Dataframe(interactive=False, wrap=True)
            with gr.Tab("Every person and direction"):
                people = gr.Dataframe(interactive=False, wrap=True)
            with gr.Tab("Anomaly score over time"):
                curve = gr.LinePlot(x="time (s)", y="value", color="line", height=300)
            with gr.Tab("Raw result (JSON)"):
                raw = gr.File(label="result.json")
        go.click(analyze, [video_in, model], [video_out, info, events, people, curve, raw])
        page.load(lambda: gr.Dropdown(choices=models()[0]), None, model)   # models trained after start-up show up
    return page


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", 7860)),
                                                     allowed_paths=[str(OUT)])
