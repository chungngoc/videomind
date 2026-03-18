from pydantic import BaseModel
from app.schemas.summary import VideoSummary


class HealthResponse(BaseModel):
    status: str
    version: str


class SummarizeResponse(BaseModel):
    video_id: str
    filename: str
    duration_seconds: float
    summary: VideoSummary
    processing_time_seconds: float
    frames_analyzed: int
    segments_transcribed: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
