import logging
import time
from dataclasses import dataclass
from pathlib import Path

import whisper

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    full_text: str
    segments: list[TranscriptSegment]
    language: str
    duration_seconds: float
    processing_time_seconds: float


class AudioPipeline:
    """
    Wraps OpenAI Whisper to transcribe audio files extracted from videos.
    Returns full transcripts and timestamped segments.
    """

    def __init__(self):
        self.model_size = settings.whisper_model_size
        self._model = None  # Lazy load

    @property
    def model(self):
        """
        Load model on first access to avoid long startup time if not needed.
        """
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")

        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe an audio file and return structured results.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file: {audio_path.name}")
        start_time = time.time()

        # Transcribe using Whisper
        result = self.model.transcribe(
            str(audio_path),
            verbose=False,
            word_timestamps=False,
        )

        elapsed_time = round(time.time() - start_time, 2)

        segments = [
            TranscriptSegment(
                start=round(segment["start"], 2),
                end=round(segment["end"], 2),
                text=segment["text"].strip(),
            )
            for segment in result.get("segments", [])
        ]

        transcription = TranscriptionResult(
            full_text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "unknown"),
            duration_seconds=round(segments[-1].end if segments else 0.0, 2),
            processing_time_seconds=elapsed_time,
        )

        logger.info(
            f"Transcription completed in {elapsed_time} seconds, language: {transcription.language}, duration: {transcription.duration_seconds} seconds"
        )
        return transcription
