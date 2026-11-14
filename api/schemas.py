"""What the API sends back: one Pydantic model per response, shown with field descriptions and
examples on /docs (and in /openapi.json, so clients can be generated from it)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

EventName = Literal["LOITERING", "COUNTER_FLOW", "TOO_FAST"]


class ErrorResponse(BaseModel):
    """Every error has the same shape: a stable code for programs, a sentence for people."""
    error: str = Field(description="Stable error code", examples=["unsupported_file_type"])
    detail: str = Field(description="What went wrong, in words",
                        examples=["unsupported file type '.txt'; send one of ['.avi', '.m4v', '.mkv', '.mov', '.mp4']"])


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: str = Field(description="Default model (`none` = behavior rules only)", examples=["astt_improved"])
    device: str = Field(description="Where the models run", examples=["cuda", "cpu"])
    threshold: Optional[float] = Field(None, description="Frame score above which a frame is abnormal (test EER point)")
    score_type: Optional[str] = Field(None, description="`mse` (ASTT models) or `score` (MULDE models)")
    warning: Optional[str] = Field(None, description="Why the model can't be used, if it can't")
    max_seconds: Optional[float] = Field(None, description="Only this many seconds of a video are analyzed (null: all)")
    light_tracking: bool = Field(False, description="People found at a lower resolution / not on every frame (CPU servers)")


class ModelCard(BaseModel):
    name: str = Field(examples=["astt_improved"])
    title: str = Field(examples=["Model 2: ASTT improved (ours)"])
    ready: bool = Field(description="Trained and evaluated, so it can be used")
    parameters: Optional[int] = Field(None, examples=[100379299])
    auc_raw: Optional[float] = Field(None, description="Frame-level AUC micro on the 107 ShanghaiTech test videos")
    auc_norm: Optional[float] = Field(None, description="AUC with per-video normalised scores")
    auc_macro: Optional[float] = Field(None, description="Mean of the per-video AUCs", examples=[0.8104])
    eer: Optional[float] = Field(None, description="Equal error rate")


class ModelList(BaseModel):
    default: str = Field(description="Used when /analyze gets no `model`", examples=["astt_improved"])
    models: list[ModelCard]


class Event(BaseModel):
    """One abnormal behavior of one person."""
    track_id: int = Field(description="The person's number (same as in `people` and in the result video)", examples=[3])
    event: EventName
    reason: str = Field(examples=["moving against the crowd"])
    start_frame: int
    end_frame: int
    start_s: float = Field(examples=[1.2])
    end_s: float = Field(examples=[4.8])


class Person(BaseModel):
    """One tracked person."""
    track_id: int
    from_s: float = Field(description="First seen (seconds)")
    to_s: float = Field(description="Last seen (seconds)")
    direction: str = Field(description="Main walking direction in the image, or `standing`",
                           examples=["right", "down-left", "standing"])
    moving_share: float = Field(description="Share of the time this person moves (0-1)")
    events: list[EventName] = Field(description="Abnormal behaviors found for this person")


class AnalyzeResult(BaseModel):
    job_id: str = Field(description="Id of this analysis (for /results/{job_id}/video)", examples=["3f9c2a1b7d40"])
    video: str = Field(description="Name of the uploaded file", examples=["clip.mp4"])
    frames: int = Field(description="Frames analyzed")
    fps: float
    model: str = Field(description="Model used (`none` = behavior rules only)")
    device: str
    threshold: Optional[float] = None
    score_type: Optional[str] = None
    warning: Optional[str] = Field(None, description="Limits that applied (e.g. only the first seconds analyzed, a camera unlike the training data)")
    people_tracked: int = Field(description="Number of different people seen")
    people: list[Person]
    events: list[Event] = Field(description="Who behaved abnormally, how and when")
    event_counts: dict[str, int] = Field(examples=[{"LOITERING": 1, "COUNTER_FLOW": 2, "TOO_FAST": 0}])
    frame_scores: Optional[list[Optional[float]]] = Field(
        None, description="The model's anomaly score of every frame (null for the first frames, which have no prediction)")
    abnormal_frames: list[int] = Field(description="Frames whose score is above the threshold")
    abnormal_ratio: float = Field(description="Share of the frames that are abnormal (0-1)")
    result_video: Optional[str] = Field(None, description="Path of the annotated video (GET it on this server)",
                                        examples=["/results/3f9c2a1b7d40/video"])
    timings: dict[str, float] = Field(description="Seconds per step", examples=[{"frames_s": 1.2, "tracking_s": 8.5,
                                                                                    "model_s": 3.1, "render_s": 2.0, "total_s": 14.8}])
