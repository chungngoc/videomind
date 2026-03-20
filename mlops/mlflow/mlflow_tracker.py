import os
import json
import tempfile
import logging
from pathlib import Path
from contextlib import contextmanager

import mlflow
from app.core.config import settings
from app.pipeline.audio import TranscriptionResult
from app.pipeline.visual import VisualAnalysisResult
from app.pipeline.preprocessing import VideoMetadata
from app.schemas.summary import VideoSummary

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "video-summarization"

class MLflowTracker:
    """
    Tracks VideoMind pipeline runs with MLflow
    Records params, metrics, and artifacts for every summarization.
    """

    def __init__(self, tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    @contextmanager
    def start_run(self, video_filename: str):
        """Context manager - wraps a full pipeline run."""
        with mlflow.start_run(run_name=video_filename, nested=True) as run:
            logger.info(f"MLflow run started: {run.info.run_id}")
            yield run
            logger.info(f"MLflow run finished: {run.info.run_id}")
    
    def log_video_params(self, metadata: VideoMetadata) -> None:
        """Log video metadata as MLflow params"""
        mlflow.log_params({
            "video_filename": metadata.filename,
            "video_duration_s": metadata.duration_seconds,
            "video_fps": metadata.fps,
            "video_resolution": f"{metadata.width}x{metadata.height}",
            "video_total_frames": metadata.total_frames,
            "frames_sampled": len(metadata.sampled_frame_paths),
        })
    
    def log_model_params(
            self,
            whisper_model: str,
            clip_model: str,
            llm_provider: str,
            llm_model: str,
            use_blip: bool,
    ) -> None:
        """Log model configuration as MLflow params."""
        mlflow.log_params({
            "whisper_model": whisper_model,
            "clip_model": clip_model,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "use_blip": use_blip,
        })
    
    def log_transcription_metrics(self, transcription: TranscriptionResult) -> None:
        """Log transcription quality metrics."""
        mlflow.log_metrics({
            "transcription_segments": len(transcription.segments),
            "transcription_duration_s": transcription.duration_seconds,
            "transcription_processing_s": transcription.processing_time_seconds,
            "transcript_word_count": len(transcription.full_text.split()),
        })
    
    def log_visual_metrics(self, visual: VisualAnalysisResult) -> None:
        """Log visual analysis metrics"""
        mlflow.log_metrics({
            "frames_analyzed": visual.frames_analyzed,
            "key_frames_selected": len(visual.key_frames),
            "visual_processing_s": visual.processing_time_seconds,
        })
    
    def log_summary_metrics(
            self,
            summary: VideoSummary,
            total_processing_s: float,
    ) -> None:
        """Log summary quality metrics."""
        mlflow.log_metrics({
            "key_moment_count": len(summary.key_moments),
            "topics_count": len(summary.topics),
            "overview_word_count": len(summary.overview.split()),
            "total_processing_s": total_processing_s,
        })
        mlflow.log_params({
            "detected_language": summary.language,
            "detected_sentiment": summary.sentiment,
        })
    
    def log_summary_artifact(self, summary: VideoSummary, video_id: str) -> None:
        """Save the full summary as a JSON artifact."""
        artifact = {
            "title": summary.title,
            "overview": summary.overview,
            "topics": summary.topics,
            "language": summary.language,
            "sentiment": summary.sentiment,
            "transcript_summary": summary.transcript_summary,
            "visual_summary": summary.visual_summary,
            "key_moments": [
                {
                    "timestamp_seconds": m.timestamp_seconds,
                    "description": m.description,
                }
                for m in summary.key_moments
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"summary_{video_id}_",
            delete=False,
        ) as f:
            json.dump(artifact, f,indent=2)
            tmp_path = f.name
        
        mlflow.log_artifact(tmp_path, artifact_path="summaries")
        os.unlink(tmp_path)
        logger.info(f"Summary artifact logged for video_id: {video_id}")