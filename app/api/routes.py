import logging
import time
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.core.config import settings
from app.pipeline.preprocessing import VideoPreprocessor
from app.pipeline.audio import AudioPipeline, TranscriptionResult
from app.pipeline.visual import VisualPipeline
from app.pipeline.fusion import FusionPipeline
from app.schemas.responses import SummarizeResponse, ErrorResponse
from mlops.mlflow.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)
router = APIRouter()
tracker = MLflowTracker()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Summarize a video file",
    description="Upload a video and get a structured multimodal summary.",
)
async def summarize_video(
    file: UploadFile = File(..., description="Video file to summarize"),
    use_blip: bool = Form(default=False),
    llm_provider: str = Form(default="ollama"),
    llm_model: str = Form(default="llama3.2:1b"),
    top_k_frames: int = Form(default=5),
    frame_sample_rate: int = Form(default=1),
):
    start = time.time()

    # Validate file type
    ext = get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save upload to disk
    upload_path = settings.upload_dir / file.filename
    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved upload: {upload_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        with tracker.start_run(video_filename=file.filename):
            # Params
            tracker.log_model_params(
                whisper_model=settings.whisper_model_size,
                clip_model=settings.clip_model_name,
                llm_provider=llm_provider,
                llm_model=llm_model,
                use_blip=use_blip,
            )

        # Preprocessing
        preprocessor = VideoPreprocessor()
        preprocessor.frame_sample_rate = frame_sample_rate
        metadata = preprocessor.process(upload_path)
        tracker.log_video_params(metadata)

        # Audio transcription
        audio_pipeline = AudioPipeline()
        transcription = (
            audio_pipeline.transcribe(metadata.audio_path)
            if metadata.audio_path
            else _empty_transcription()
        )
        tracker.log_transcription_metrics(transcription)

        # Visual analysis
        visual_pipeline = VisualPipeline(use_blip=use_blip)
        visual = visual_pipeline.analyze(
            frame_paths=metadata.sampled_frame_paths,
            fps=metadata.fps,
            top_k_frames=top_k_frames,
        )
        tracker.log_visual_metrics(visual)

        # LLM fusion
        fusion = FusionPipeline()
        fusion.provider = llm_provider
        fusion.model = llm_model
        summary = fusion.summarize(transcription, visual)

        elapsed = round(time.time() - start, 2)
        tracker.log_summary_metrics(summary=summary, total_processing_s=elapsed)
        tracker.log_summary_artifact(summary=summary, video_id=metadata.video_id)

        return SummarizeResponse(
            video_id=metadata.video_id,
            filename=metadata.filename,
            duration_seconds=metadata.duration_seconds,
            summary=summary,
            processing_time_seconds=elapsed,
            frames_analyzed=visual.frames_analyzed,
            segments_transcribed=len(transcription.segments),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during summarization")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup upload
        if upload_path.exists():
            upload_path.unlink()
            logger.info(f"Cleaned up upload: {upload_path}")


def _empty_transcription():
    return TranscriptionResult(
        full_text="",
        segments=[],
        language="unknown",
        duration_seconds=0.0,
        processing_time_seconds=0.0,
    )
