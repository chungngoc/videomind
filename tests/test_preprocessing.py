# We'll use a real tiny video generated on the fly so no test assets needed
import pytest
import cv2
import numpy as np
from pathlib import Path
from app.pipeline.preprocessing import VideoPreprocessor, VideoMetadata

def make_test_video(path: Path, duration_seconds: int = 3, fps: int = 10) ->Path:
    """
    Generate a minimal video file for testing.
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    total_frames = duration_seconds * fps
    for i in range(total_frames):
        # Gradient frame so each frame is slightly different
        frame = np.full((height, width, 3), (i * 4 % 255, 100, 150), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path

@pytest.fixture
def test_video(tmp_path) ->Path:
    return make_test_video(tmp_path / "test_video.mp4")

@pytest.fixture
def preprocessor() ->VideoPreprocessor:
    return VideoPreprocessor()

def test_returns_metadata(test_video: Path, preprocessor: VideoPreprocessor):
    metadata = preprocessor.process(test_video)
    assert isinstance(metadata, VideoMetadata)

def test_metadata_fields(test_video: Path, preprocessor: VideoPreprocessor):
    metadata = preprocessor.process(test_video)
    assert metadata.filename == "test_video.mp4"
    assert metadata.duration_seconds > 0
    assert metadata.fps > 0
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.total_frames > 0
    assert metadata.video_id is not None

def test_frame_extraction(test_video: Path, preprocessor: VideoPreprocessor):
    metadata = preprocessor.process(test_video)
    assert len(metadata.sampled_frame_paths) > 0
    assert all(path.exists() for path in metadata.sampled_frame_paths)
    assert all(path.suffix == ".jpg" for path in metadata.sampled_frame_paths)

def test_frames_dir_exists(test_video: Path, preprocessor: VideoPreprocessor):
    metadata = preprocessor.process(test_video)
    assert metadata.frames_dir.exists()

def test_invalid_path_raises(preprocessor, tmp_path):
    with pytest.raises(ValueError, match="Could not open video file"):
        preprocessor.process(tmp_path / "nonexistent.mp4")

def test_video_too_long_raises(preprocessor, tmp_path):
    # Create a video that exceeds the max duration limit
    long_video = make_test_video(tmp_path / "long_video.mp4", duration_seconds=2)
    preprocessor.max_duration = 1  # Set max duration to 1 second for this test
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        preprocessor.process(long_video)