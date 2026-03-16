import json
import logging
import time

import openai

from app.core.config import settings
from app.pipeline.visual import VisualAnalysisResult
from app.pipeline.audio import TranscriptionResult
from app.schemas.summary import VideoSummary, KeyMoment

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """
You are an AI assistant that summarizes videos by analyzing both speech and visual content.

You will be given:
1. A transcript of the spoken audio with timestamps
2. Descriptions of key visual frames with timestamps

Your task is to produce a structured JSON summary.

--- TRANSCRIPT ---
{transcript}

--- VISUAL FRAMES ---
{visual_context}

--- INSTRUCTIONS ---
Return ONLY a valid JSON object with this exact structure:
{{
    "title": "inferred title of the video",
    "overview": "2-3 sentence overview of the full video",
    "key_moments": [
        {{"timestamp_seconds": 0.0, "description": "what happens at this moment"}}
    ],
    "topics": ["topic1", "topic2"],
    "language": "detected language",
    "sentiment": "positive | neutral | negative",
    "transcript_summary": "summary of what was said",
    "visual_summary": "summary of what was seen"
}}

Return only the JSON - na markdown, no explanation, no code blocks.
""".strip()


class FusionPipeline:
    """
    Fuse transcript + visual analysis into a structured summary using an LLM.
    Support Ollama (local) and OpenAPI (API).
    """

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self._client = None

    ### LLM Clients

    def _get_ollama_client(self):
        if self._client is None:
            self._client = openai.OpenAI(
                base_url="http:/localhost:11434/v1", api_key="ollama"
            )
        return self._client

    def _get_openai_client(self):
        if self._client is None:
            self._client = openai.OpenAI(api_key=settings.openai_api_key)

    def _get_client(self):
        if self.provider == "ollama":
            return self._get_ollama_client()
        elif self.provider == "openai":
            return self._get_openai_client()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    ### Context builders
    def _build_transcript_context(self, transcription: TranscriptionResult) -> str:
        if not transcription.segments:
            return transcription.full_text or "No speech detected."
        lines = [f"[{s.start}s - {s.end}s] {s.text}" for s in transcription.segments]

        return "\n".join(lines)

    def _build_visual_context(self, visual: VisualAnalysisResult) -> str:
        if not visual.key_frames:
            return "No visual frames available"

        lines = [f"[{f.timestamp_seconds}s] {f.caption}" for f in visual.key_frames]
        return "\n".join(lines)

    ### LLM call

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    ### JSON parsing
    def _parse_response(self, raw: str) -> VideoSummary:
        try:
            # Strip markdown fences if model adds them despite instructions
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]

            data = json.loads(clean.strip())
            key_moments = [KeyMoment(**m) for m in data.get("key_moments", [])]

            return VideoSummary(
                title=data.get("title", "Untitled"),
                overview=data.get("overview", ""),
                key_moments=key_moments,
                topics=data.get("topics", []),
                language=data.get("language", "unknown"),
                sentiment=data.get("sentiment", "neutral"),
                transcript_summary=data.get("transcript_summary", ""),
                visual_summary=data.get("visual_summary", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return VideoSummary(
                title="Untitled",
                overview=raw,
                key_moments=[],
                topics=[],
                language="unknown",
                sentiment="neutral",
                transcript_summary=raw,
                visual_summary="",
            )

    # Public API
    def summarize(
        self,
        transcription: TranscriptionResult,
        visual: VisualAnalysisResult,
    ) -> VideoSummary:
        """Fuse transcript _ visual analysis into a structured summary."""
        logger.info(f"Starting fusion - provider: {self.provider}, model: {self.model}")
        start = time.time()

        transcript_context = self._build_transcript_context(transcription)
        visual_context = self._build_visual_context(visual)

        prompt = SUMMARY_PROMPT.format(
            transcript=transcript_context,
            visual_context=visual_context,
        )

        raw = self._call_llm(prompt)
        summary = self._parse_response(raw)

        elapsed = round(time.time() - start, 2)
        logger.info(f"Fusion done in {elapsed} seconds")

        return summary
