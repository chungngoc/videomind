### Whisper needs a real audio file with actual speech to transcribe.
### We will generate a silent wav for structured testing and mock Whisper for the transcription logic.

import wave
import struct
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.pipeline.audio import AudioPipeline, TranscriptionResult, TranscriptSegment


def make_silent_wav(
    path: Path, duration_seconds: int = 2, sample_rate: int = 16000
) -> Path:
    """
    Create a silent WAV file for testing.
    """
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        num_frames = duration_seconds * sample_rate
        frames = struct.pack(
            f"<{num_frames}h",
            *[0] * num_frames,  # Silent audio (all samples are zero)
        )
        wf.writeframes(frames)
    return path


def make_tone_wav(
    path: Path, duration_seconds: int = 2, sample_rate: int = 16000
) -> Path:
    """
    Generate a simple sine wave wav for testing.
    """
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = duration_seconds * sample_rate
        frames = struct.pack(
            f"<{num_frames}h",
            *[
                int(32767 * math.sin(2 * math.pi * 440 * t / sample_rate))
                for t in range(num_frames)
            ],
        )
        wf.writeframes(frames)
    return path


@pytest.fixture
def silent_audio(tmp_path) -> Path:
    return make_silent_wav(tmp_path / "silent.wav")


@pytest.fixture
def tone_audio(tmp_path) -> Path:
    return make_tone_wav(tmp_path / "tone.wav")


@pytest.fixture
def mock_whisper_result():
    return {
        "text": "Hello world this is a test transcription.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Hello world"},
            {"start": 1.5, "end": 3.0, "text": "this is a test transcription."},
        ],
    }


@pytest.fixture
def mock_model(mock_whisper_result):
    """
    Mock whisper.load_model returning a fake model
    """
    model = MagicMock()
    model.transcribe.return_value = mock_whisper_result
    with patch("whisper.load_model", return_value=model):
        yield model


@pytest.fixture
def pipeline(mock_model) -> AudioPipeline:
    """Pipeline with Whisper mocked out."""
    return AudioPipeline()


@pytest.fixture
def pipeline_no_mock() -> AudioPipeline:
    """Pipeline without any mock — for error path tests only."""
    return AudioPipeline()


def test_file_not_found_raises(pipeline_no_mock, tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        pipeline_no_mock.transcribe(tmp_path / "missing.wav")


def test_returns_transcription_result(pipeline, tone_audio, mock_whisper_result):
    result = pipeline.transcribe(tone_audio)
    assert isinstance(result, TranscriptionResult)


def test_full_text(pipeline, tone_audio, mock_whisper_result):
    result = pipeline.transcribe(tone_audio)
    assert result.full_text == "Hello world this is a test transcription."


def test_segments_parsed(pipeline, tone_audio, mock_whisper_result):
    result = pipeline.transcribe(tone_audio)
    assert len(result.segments) == 2
    assert all(isinstance(seg, TranscriptSegment) for seg in result.segments)


def test_segment_fields(pipeline, tone_audio, mock_whisper_result):
    result = pipeline.transcribe(tone_audio)
    first = result.segments[0]
    assert first.start == 0.0
    assert first.end == 1.5
    assert first.text == "Hello world"


def test_language_detected(pipeline, tone_audio, mock_whisper_result):
    result = pipeline.transcribe(tone_audio)
    assert result.language == "en"


def test_processing_time_recorded(pipeline, tone_audio, mock_whisper_result):
    result = pipeline.transcribe(tone_audio)
    assert result.processing_time_seconds >= 0


def test_empty_segments(pipeline_no_mock, tone_audio):
    empty_result = {
        "text": "",
        "language": "unknown",
        "segments": [],
    }
    with patch(
        "whisper.load_model",
        return_value=MagicMock(transcribe=MagicMock(return_value=empty_result)),
    ):
        result = pipeline_no_mock.transcribe(tone_audio)
    assert result.full_text == ""
    assert result.segments == []
    assert result.duration_seconds == 0
