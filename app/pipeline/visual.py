import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    frame_path: Path
    timestamp_seconds: float
    caption: str
    clip_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class VisualAnalysisResult:
    frames_analyzed: int
    frame_analyses: list[FrameAnalysis]
    key_frames: list[FrameAnalysis]
    processing_time_seconds: float
    model_used: str


class VisualPipeline:
    """
    Analyzes sampled video frames using CLIP and optionally BLIP.
        - CLIP: scores frames against candidate labels (fast, CPU-friendly)
        - BLIP: generates natural language captions per frame (heavier)
    """

    # Default scene labels for CLIP scoring
    DEFAULT_LABELS = [
        "a person speaking",
        "a group of people",
        "outdoor scene",
        "indoor scene",
        "text or slides",
        "a product or object",
        "an animal",
        "a vehicle",
        "a natural landscape",
    ]

    def __init__(self, use_blip: bool = False):
        self.use_blip = use_blip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_processor = None
        self._blip_model = None
        self._blip_processor = None
        logger.info(
            f"VisualPipeline initialized with use_blip={self.use_blip} on device {self.device}"
        )

    # Lazy loaders
    def _load_clip(self):
        if self._clip_model is None:
            logger.info(f"Loading CLIP model: {settings.clip_model_name}")
            self._clip_processor = CLIPProcessor.from_pretrained(
                settings.clip_model_name
            )
            self._clip_model = CLIPModel.from_pretrained(settings.clip_model_name).to(
                self.device
            )
            logger.info("CLIP model loaded successfully")

    def _load_blip(self):
        if self._blip_model is None:
            logger.info("Loading BLIP (this may take a moment)...")
            self._blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16
                if self.device.type == "cuda"
                else torch.float32,
            ).to(self.device)
            logger.info("BLIP model loaded successfully")

    # Public API
    def analyze(
        self,
        frame_paths: list[Path],
        fps: float = 1.0,
        labels: list[str] | None = None,
        top_k_frames: int = 5,
    ) -> VisualAnalysisResult:
        """Analyzes a list of frame paths and returns a structured result."""
        if not frame_paths:
            raise ValueError("No frames provided for analysis.")

        labels = labels or self.DEFAULT_LABELS
        start = time.time()

        self._load_clip()
        if self.use_blip:
            self._load_blip()

        analyses = []
        for i, frame_path in enumerate(frame_paths):
            timestamp = round(i / fps, 2)
            analysis = self._analyze_frame(frame_path, timestamp, labels)
            analyses.append(analysis)

        key_frames = self._select_key_frames(analyses, top_k_frames)
        elapsed = round(time.time() - start, 2)

        logger.info(
            f"Visual analysis done in {elapsed}s -"
            f" {len(analyses)} frames analyzed, {len(key_frames)} key frames selected."
        )

        return VisualAnalysisResult(
            frames_analyzed=len(analyses),
            frame_analyses=analyses,
            key_frames=key_frames,
            processing_time_seconds=elapsed,
            model_used="BLIP + CLIP" if self.use_blip else "CLIP",
        )

    # Private helpers
    def _analyze_frame(
        self, frame_path: Path, timestamp: float, labels: list[str]
    ) -> FrameAnalysis:
        """Analyzes a single frame and returns a FrameAnalysis dataclass."""
        image = Image.open(frame_path).convert("RGB")

        # CLIP scoring
        clip_scores = self._score_with_clip(image, labels)
        caption = (
            self._caption_with_blip(image)
            if self.use_blip
            else max(
                clip_scores, key=clip_scores.get
            )  # Fallback to top CLIP label as caption
        )

        return FrameAnalysis(
            frame_path=frame_path,
            timestamp_seconds=timestamp,
            caption=caption,
            clip_scores=clip_scores,
        )

    def _score_with_clip(
        self, image: Image.Image, labels: list[str]
    ) -> dict[str, float]:
        """
        Return a confidence score per label for the given image
        """
        inputs = self._clip_processor(
            text=labels, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        return {label: round(float(prob), 4) for label, prob in zip(labels, probs)}

    def _caption_with_blip(self, image: Image.Image) -> str:
        """Generate a caption for the image using BLIP."""
        inputs = self._blip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._blip_model.generate(**inputs, max_new_tokens=50)
            caption = self._blip_processor.decode(outputs[0], skip_special_tokens=True)

        return caption.strip()

    def _select_key_frames(
        self, analyses: list[FrameAnalysis], top_k: int
    ) -> list[FrameAnalysis]:
        """Select top-k frames by their highest CLIP score."""
        scored = [
            (
                max(analysis.clip_scores.values()) if analysis.clip_scores else 0,
                analysis,
            )
            for analysis in analyses
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [analysis for _, analysis in scored[:top_k]]
