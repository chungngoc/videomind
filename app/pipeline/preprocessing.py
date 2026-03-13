import cv2
import ffmpeg
import logging
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    video_id: str
    filename: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    total_frames: int
    audio_path: Path | None
    frames_dir: Path
    sampled_frame_paths: list[Path] = field(default_factory=list)

class VideoPreprocessor:
    """
    Handles video ingestion:
        - Validates video file
        - Extracts metadata (duration, fps, resolution)
        - Extracts audio as wav
        - Samples frames at a fixed rate (e.g., 1 frame per second)
    """

    def __init__(self):
        self.frame_sample_rate = settings.frame_sample_rate
        self.max_duration = settings.max_video_duration
        self.upload_dir = settings.upload_dir
        self.output_dir = settings.output_dir

    def process(self, video_path: Path) -> VideoMetadata:
        """
        Full preprocessing pipeline for a single video file.
        Returns metadata
        """
        logger.info(f"Processing video: {video_path.name}")

        video_id = str(uuid.uuid4())[:8]  # Short unique ID for this video
        work_dir = self.output_dir / video_id
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._extract_metadata(video_path, video_id, frames_dir)
        self._validate(metadata)

        metadata.sampled_frame_paths = self._extract_frames(video_path, frames_dir)
        metadata.audio_path = self._extract_audio(video_path, work_dir)

        logger.info(
            f"Preprocessing complete for video {video_path.name}: "
            f"{len(metadata.sampled_frame_paths)} frames extracted, "
            f"audio: {metadata.audio_path is not None}"
        )

        return metadata

    def _extract_metadata(self, video_path: Path, video_id: str, frames_dir: Path) -> VideoMetadata:
        """
        Extracts video metadata using OpenCV. Validates duration and other properties.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Default to 25 if FPS is not available
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
    
        cap.release()

        return VideoMetadata(
            video_id=video_id,
            filename=video_path.name,
            duration_seconds=round(duration, 2),
            fps=round(fps, 2),
            width=width,
            height=height,
            total_frames=total_frames,
            audio_path=None,  # To be filled in after audio extraction
            frames_dir=frames_dir
        )
    
    def _validate(self, metadata: VideoMetadata) ->None:
        """Validates video metadata against constraints (e.g., max duration)."""
        if metadata.duration_seconds > self.max_duration:
            raise ValueError(f"Video duration {metadata.duration_seconds}s exceeds maximum allowed {self.max_duration}s")
        
        if metadata.fps ==0 or metadata.total_frames == 0:
            raise ValueError("Video has invalid FPS or total frames, cannot process.")
    
    def _extract_frames(self, video_path: Path, frames_dir: Path) -> list[Path]:
        """
        Sample one frame every N second using OpenCV and save to frames_dir.
        Returns list of saved frame paths.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = int(fps * self.frame_sample_rate)
        interval = max(frame_interval, 1)  # Ensure at least every frame is sampled if frame rate is very low

        saved, frame_idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved.append(out_path)
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(saved)} frames from video to {frames_dir}.")
        return saved
    
    def _extract_audio(self, video_path: Path, output_dir: Path) -> Path | None:
        """
        Extract audio track as 16kHz mono wav (Whisper-ready format).
        Returns path to extracted audio or None if no audio track is present.
        """
        audio_path = output_dir / "audio.wav"
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16000",
                    loglevel="error"
                )
                .overwrite_output()
                .run()
            )
            logger.info(f"Extracted audio to {audio_path}.")
            return audio_path
        except ffmpeg.Error as e:
            logger.warning(f"Audio extraction failed (video may be silent): {e}")
            return None
