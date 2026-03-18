import cv2
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from pathlib import Path

from app.main import app
from app.schemas.summary import VideoSummary, KeyMoment
from app.pipeline.audio import TranscriptionResult, TranscriptSegment
from app.pipeline.visual import VisualAnalysisResult, FrameAnalysis
from app.pipeline.preprocessing import VideoMetadata

### Helper


def make_test_video_bytes(duration: int = 2, fps: int = 10) -> bytes:
    """Generate a real minimal mp4 in memory."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (320, 240))
    for i in range(duration * fps):
        frame = np.full((240, 320, 3), (i * 4 % 255, 100, 150), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


### Mock factories


def mock_metadata(tmp_path: Path) -> VideoMetadata:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()  # create empty file so path is not None
    return VideoMetadata(
        video_id="test123",
        filename="test_video.mp4",
        duration_seconds=2.0,
        fps=10.0,
        width=320,
        height=240,
        total_frames=20,
        audio_path=audio_path,
        frames_dir=frames_dir,
        sampled_frame_paths=[],
    )


def mock_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        full_text="Hello world.",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="Hello world.")],
        language="en",
        duration_seconds=2.0,
        processing_time_seconds=0.1,
    )


def mock_visual() -> VisualAnalysisResult:
    return VisualAnalysisResult(
        frames_analyzed=2,
        frame_analyses=[],
        key_frames=[
            FrameAnalysis(
                frame_path=Path("frame_0.jpg"),
                timestamp_seconds=0.0,
                caption="a person speaking",
                clip_scores={},
            )
        ],
        processing_time_seconds=0.1,
        model_used="CLIP",
    )


def mock_summary() -> VideoSummary:
    return VideoSummary(
        title="Test Video",
        overview="A short test video.",
        key_moments=[KeyMoment(timestamp_seconds=0.0, description="Start")],
        topics=["testing"],
        language="en",
        sentiment="neutral",
        transcript_summary="Someone says hello.",
        visual_summary="A person speaking.",
    )


### Tests
@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_summarize_invalid_file_type():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "api/v1/summarize",
            files={"file": ("test.txt", b"not a video", "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_summarize_response_structure(tmp_path):
    video_bytes = make_test_video_bytes()

    with (
        patch("app.api.routes.VideoPreprocessor") as mock_pre,
        patch("app.api.routes.AudioPipeline") as mock_audio,
        patch("app.api.routes.VisualPipeline") as mock_visual_cls,
        patch("app.api.routes.FusionPipeline") as mock_fusion,
    ):
        mock_pre.return_value.process.return_value = mock_metadata(tmp_path)
        mock_audio.return_value.transcribe.return_value = mock_transcription()
        mock_visual_cls.return_value.analyze.return_value = mock_visual()
        mock_fusion.return_value.summarize.return_value = mock_summary()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/summarize",
                files={"file": ("test.mp4", video_bytes, "video/mp4")},
            )

    data = response.json()

    assert "video_id" in data
    assert "duration_seconds" in data
    assert "processing_time_seconds" in data
    assert "summary" in data
    assert "key_moments" in data["summary"]
    assert "topics" in data["summary"]


@pytest.mark.asyncio
async def test_summarize_success(tmp_path):
    video_bytes = make_test_video_bytes()

    with (
        patch("app.api.routes.VideoPreprocessor") as mock_pre,
        patch("app.api.routes.AudioPipeline") as mock_audio,
        patch("app.api.routes.VisualPipeline") as mock_visual_cls,
        patch("app.api.routes.FusionPipeline") as mock_fusion,
    ):
        mock_pre.return_value.process.return_value = mock_metadata(tmp_path)
        mock_audio.return_value.transcribe.return_value = mock_transcription()
        mock_visual_cls.return_value.analyze.return_value = mock_visual()
        mock_fusion.return_value.summarize.return_value = mock_summary()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/summarize",
                files={"file": ("test.mp4", video_bytes, "video/mp4")},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_video.mp4"
    assert data["summary"]["title"] == "Test Video"
    assert data["frames_analyzed"] == 2
    assert data["segments_transcribed"] == 1
