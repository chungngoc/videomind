import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from app.pipeline.visual import VisualPipeline, FrameAnalysis, VisualAnalysisResult

### Helpers functions for tests


def make_test_frames(tmp_path: Path, count: int = 3) -> list[Path]:
    """
    Generate minimal RGB frames for testing.
    """
    paths = []
    for i in range(count):
        img = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        path = tmp_path / f"frame_{i:03d}.jpg"
        img.save(path)
        paths.append(path)
    return paths


def make_mock_clip_outputs(labels: list[str]) -> MagicMock:
    """Return a mock CLIP output with uniform probability distribution."""
    logits = torch.zeros(1, len(labels))  # Uniform logits
    mock_output = MagicMock()
    mock_output.logits_per_image = logits
    return mock_output


@pytest.fixture
def test_frames(tmp_path: Path) -> list[Path]:
    return make_test_frames(tmp_path, count=4)


@pytest.fixture
def mock_clip():
    """Mock CLIP model and processor."""
    with (
        patch("app.pipeline.visual.CLIPModel.from_pretrained") as mock_model_cls,
        patch(
            "app.pipeline.visual.CLIPProcessor.from_pretrained"
        ) as mock_processor_cls,
    ):
        mock_processor = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_processor_cls.return_value = mock_processor

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.return_value = make_mock_clip_outputs(VisualPipeline.DEFAULT_LABELS)
        mock_model_cls.return_value = mock_model

        yield mock_model


@pytest.fixture
def pipeline(mock_clip) -> VisualPipeline:
    p = VisualPipeline(use_blip=False)
    p._clip_model = mock_clip
    p._clip_processor = MagicMock()
    p._clip_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }

    # Make processor behave like a callable returning a mock with .to()
    inputs_mock = MagicMock()
    inputs_mock.to.return_value = inputs_mock
    p._clip_processor.return_value = inputs_mock
    p._clip_processor.side_effect = None  # Ensure it doesn't interfere with the test
    p._clip_processor = MagicMock(return_value=inputs_mock)
    return p


#### Test cases


def test_no_frames_raises(pipeline: VisualPipeline):
    with pytest.raises(ValueError, match="No frames provided"):
        pipeline.analyze([])


def test_return_visual_analysis_result(pipeline, test_frames):
    result = pipeline.analyze(test_frames)
    assert isinstance(result, VisualAnalysisResult)
    assert result.frames_analyzed == len(test_frames)
    assert len(result.frame_analyses) == len(test_frames)
    assert all(
        isinstance(analysis, FrameAnalysis) for analysis in result.frame_analyses
    )


def test_timestamps_assigned(pipeline, test_frames):
    result = pipeline.analyze(test_frames, fps=1.0)
    timestamps = [a.timestamp_seconds for a in result.frame_analyses]
    assert timestamps == [0.0, 1.0, 2.0, 3.0]


def test_key_frames_selected(pipeline, test_frames):
    result = pipeline.analyze(test_frames, top_k_frames=2)
    assert len(result.key_frames) <= 2
    assert all(isinstance(kf, FrameAnalysis) for kf in result.key_frames)


def test_key_frames_dont_exceed_total(pipeline, test_frames):
    result = pipeline.analyze(test_frames, top_k_frames=100)
    assert len(result.key_frames) <= len(test_frames)


def test_model_used_clip_only(pipeline, test_frames):
    result = pipeline.analyze(test_frames)
    assert result.model_used == "CLIP"


def test_processing_time_recorded(pipeline, test_frames):
    result = pipeline.analyze(test_frames)
    assert result.processing_time_seconds >= 0


def test_frames_analysis_has_caption(pipeline, test_frames):
    result = pipeline.analyze(test_frames)
    for analysis in result.frame_analyses:
        assert isinstance(analysis.caption, str)
        assert len(analysis.caption) > 0
