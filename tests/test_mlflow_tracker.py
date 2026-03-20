import pytest
from pathlib import Path
import mlflow

from mlops.mlflow.mlflow_tracker import MLflowTracker
from app.pipeline.audio import TranscriptionResult, TranscriptSegment
from app.pipeline.visual import VisualAnalysisResult, FrameAnalysis
from app.pipeline.preprocessing import VideoMetadata
from app.schemas.summary import VideoSummary, KeyMoment


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    if mlflow.active_run():
        mlflow.end_run()
    yield
    if mlflow.active_run():
        mlflow.end_run()


### Mocking
@pytest.fixture
def tracker(tmp_path):
    """Tracker using a temp directory as MLflow store."""
    return MLflowTracker(tracking_uri=str(tmp_path / "mlruns"))


@pytest.fixture
def mock_metadata(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    return VideoMetadata(
        video_id="test123",
        filename="test.mp4",
        duration_seconds=60.0,
        fps=10.0,
        width=320,
        height=240,
        total_frames=600,
        audio_path=None,
        frames_dir=frames_dir,
        sampled_frame_paths=[Path("f1.jpg"), Path("f2.jpg")],
    )


@pytest.fixture
def mock_transcription():
    return TranscriptionResult(
        full_text="Hello world this is a test.",
        segments=[TranscriptSegment(start=0.0, end=2.0, text="Hello world")],
        language="en",
        duration_seconds=60.0,
        processing_time_seconds=2.5,
    )


@pytest.fixture
def mock_visual():
    return VisualAnalysisResult(
        frames_analyzed=10,
        frame_analyses=[],
        key_frames=[
            FrameAnalysis(
                frame_path=Path("f1.jpg"),
                timestamp_seconds=0.0,
                caption="a person speaking",
                clip_scores={},
            )
        ],
        processing_time_seconds=1.2,
        model_used="CLIP",
    )


@pytest.fixture
def mock_summary():
    return VideoSummary(
        title="Test Video",
        overview="A test video overview.",
        key_moments=[KeyMoment(timestamp_seconds=0.0, description="Start")],
        topics=["testing", "demo"],
        language="en",
        sentiment="neutral",
        transcript_summary="Someone says hello.",
        visual_summary="A person speaking.",
    )


### Test


def test_tracker_initializes(tracker):
    assert tracker is not None


def test_start_run_context_manager(tracker):
    with tracker.start_run(video_filename="test.mp4") as run:
        assert run is not None
        assert run.info.run_id is not None


def test_log_video_params(tracker, mock_metadata):
    with tracker.start_run(video_filename="test.mp4"):
        tracker.log_video_params(mock_metadata)
        run = mlflow.active_run()
        assert run is not None


def test_log_transcription_metrics(tracker, mock_transcription):
    with tracker.start_run(video_filename="test.mp4"):
        tracker.log_transcription_metrics(mock_transcription)
        run = mlflow.active_run()
        assert run is not None


def test_log_visual_metrics(tracker, mock_visual):
    with tracker.start_run(video_filename="test.mp4"):
        tracker.log_visual_metrics(mock_visual)
        run = mlflow.active_run()
        assert run is not None


def test_log_summary_metrics(tracker, mock_summary):
    with tracker.start_run(video_filename="test.mp4"):
        tracker.log_summary_metrics(mock_summary, total_processing_s=5.0)
        run = mlflow.active_run()
        assert run is not None


def test_log_summary_artifact(tracker, mock_summary):
    with tracker.start_run(video_filename="test.mp4"):
        tracker.log_summary_artifact(mock_summary, video_id="test123")
        run = mlflow.active_run()
        assert run is not None


def test_multiple_runs_tracked(tracker, mock_summary):
    with tracker.start_run(video_filename="video1.mp4"):
        tracker.log_summary_metrics(mock_summary, total_processing_s=3.0)
    with tracker.start_run(video_filename="video2.mp4"):
        tracker.log_summary_metrics(mock_summary, total_processing_s=5.0)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("video-summarization")
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) == 2
