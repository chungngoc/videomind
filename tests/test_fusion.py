import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.pipeline.fusion import FusionPipeline
from app.pipeline.audio import TranscriptionResult, TranscriptSegment
from app.pipeline.visual import VisualAnalysisResult, FrameAnalysis
from app.schemas.summary import VideoSummary


### Fixtures
@pytest.fixture
def mock_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        full_text="Hello world this is a test video.",
        segments=[
            TranscriptSegment(start=0, end=1.5, text="Hello world"),
            TranscriptSegment(start=1.5, end=3.0, text="this is a test video"),
        ],
        language="en",
        duration_seconds=3.0,
        processing_time_seconds=0.5,
    )


@pytest.fixture
def mock_visual() -> VisualAnalysisResult:
    return VisualAnalysisResult(
        frames_analyzed=3,
        frame_analyses=[
            FrameAnalysis(
                frame_path=Path("frame_0.jpg"),
                timestamp_seconds=0.0,
                caption=" a person speaking",
                clip_scores={"a person speaking": 0.9},
            )
        ],
        key_frames=[
            FrameAnalysis(
                frame_path=Path("frame_0.jpg"),
                timestamp_seconds=0.0,
                caption="a person speaking",
                clip_scores={"a person speaking": 0.9},
            )
        ],
        processing_time_seconds=0.3,
        model_used="CLIP",
    )


@pytest.fixture
def mock_llm_response() -> str:
    return """
        {
        "title": "Test Video",
        "overview": "A short test video with a person speaking.",
        "key_moments": [
            {"timestamp_seconds": 0.0, "description": "Person starts speaking"}
        ],
        "topics": ["testing", "demo"],
        "language": "en",
        "sentiment": "neutral",
        "transcript_summary": "Someone says hello world.",
        "visual_summary": "A person speaking to the camera."
        }
        """.strip()


@pytest.fixture
def pipeline() -> FusionPipeline:
    return FusionPipeline()


def mock_completion(content: str) -> MagicMock:
    """
    Helper to build a mock OpenAI completion response.
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


### Test
def test_returns_video(pipeline, mock_transcription, mock_visual, mock_llm_response):
    with patch.object(pipeline, "_call_llm", return_value=mock_llm_response):
        result = pipeline.summarize(mock_transcription, mock_visual)

    assert isinstance(result, VideoSummary)
    assert result.title == "Test Video"
    assert "testing" in result.topics
    assert len(result.key_moments) == 1
    assert result.key_moments[0].timestamp_seconds == 0.0
    assert result.sentiment == "neutral"


def test_transcript_context_build(pipeline, mock_transcription):
    context = pipeline._build_transcript_context(mock_transcription)
    assert "[1.5s - 3.0s]" in context
    assert "Hello world" in context


def test_visual_context_build(pipeline, mock_visual):
    context = pipeline._build_visual_context(mock_visual)
    assert "[0.0s]" in context
    assert "a person speaking" in context


def test_malformed_json_fallback(pipeline, mock_transcription, mock_visual):
    with patch.object(pipeline, "_call_llm", return_value="not valid json {{"):
        result = pipeline.summarize(mock_transcription, mock_visual)
    assert isinstance(result, VideoSummary)
    assert result.title == "Untitled"


def test_markdown_fences_stripped(
    pipeline, mock_transcription, mock_visual, mock_llm_response
):
    fenced = f"```json\n{mock_llm_response}\n```"
    with patch.object(pipeline, "_call_llm", return_value=fenced):
        result = pipeline.summarize(mock_transcription, mock_visual)
    assert result.title == "Test Video"


def test_empty_transcript_handled(pipeline, mock_visual, mock_llm_response):
    empty_transcription = TranscriptionResult(
        full_text="",
        segments=[],
        language="unknown",
        duration_seconds=0.0,
        processing_time_seconds=0.0,
    )
    with patch.object(pipeline, "_call_llm", return_value=mock_llm_response):
        result = pipeline.summarize(empty_transcription, mock_visual)
    assert isinstance(result, VideoSummary)


def test_unknown_provider_raises(mock_transcription, mock_visual):
    pipeline = FusionPipeline()
    pipeline.provider = "unknown_provider"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        pipeline._get_client()
