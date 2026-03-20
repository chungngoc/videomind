import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import traceback

from app.pipeline.preprocessing import VideoPreprocessor
from app.pipeline.audio import AudioPipeline, TranscriptionResult
from app.pipeline.visual import VisualPipeline
from app.pipeline.fusion import FusionPipeline
from mlops.mlflow.mlflow_tracker import MLflowTracker

tracker = MLflowTracker()


# Pipeline runners
def _error_outputs(msg: str) -> tuple:
    return ("", "", "", "", "", _status(msg, success=False))


def _status(msg: str, success: bool) -> str:
    icon = "✅" if success else "❌"
    return f"{icon} {msg}"


def run_summarize(
    video_path: str,
    llm_provider: str,
    llm_model: str,
    use_blip: bool,
    top_k_frames: int,
    frame_sample_rate: int,
    progress=gr.Progress(),
) -> tuple:
    """
    Full pipeline run triggered by Gradio UI.
    """
    if not video_path:
        return _error_outputs("Please upload a video first.")

    try:
        video_path = Path(video_path)

        with tracker.start_run(video_filename=video_path.name):
            # Preprocessing
            progress(0.1, desc="Extracting frames and audio...")
            preprocessor = VideoPreprocessor()
            preprocessor.frame_sample_rate = frame_sample_rate
            metadata = preprocessor.process(video_path=video_path)
            tracker.log_video_params(metadata)

            meta_text = (
                f"**Filename:** {metadata.filename}\n"
                f"**Duration:** {metadata.duration_seconds}s\n"
                f"**FPS:** {metadata.fps}\n"
                f"**Resolution:** {metadata.width}x{metadata.height}\n"
                f"**Frames sampled:** {len(metadata.sampled_frame_paths)}"
            )

            # Transcription
            progress(0.3, desc="Trancribing audio with Whisper...")
            if metadata.audio_path:
                audio_pipeline = AudioPipeline()
                transcription = audio_pipeline.transcribe(metadata.audio_path)
                transcript_text = transcription.full_text or "No speech detected"
            else:
                transcription = TranscriptionResult(
                    full_text="",
                    segments=[],
                    language="unknown",
                    duration_seconds=0.0,
                    processing_time_seconds=0.0,
                )
                transcript_text = "No audio track found."
            tracker.log_transcription_metrics(transcription)

            # Visual analysis
            progress(0.6, desc="Analyzing frames with CLIP...")
            visual_pipeline = VisualPipeline(use_blip=use_blip)
            visual = visual_pipeline.analyze(
                frame_paths=metadata.sampled_frame_paths,
                fps=metadata.fps,
                top_k_frames=top_k_frames,
            )
            visual_text = "\n".join(
                [f"**[{f.timestamp_seconds}s]**" for f in visual.key_frames]
            )
            tracker.log_visual_metrics(visual)

            # LLM Fusion
            progress(0.85, desc="Generating summary with LLM...")
            fusion = FusionPipeline()
            fusion.provider = llm_provider
            fusion.model = llm_model
            summary = fusion.summarize(transcription=transcription, visual=visual)

            tracker.log_summary_metrics(summary, total_processing_s=0.0)
            tracker.log_summary_artifact(summary, video_id=metadata.video_id)

            # Format outputs
            progress(1.0, desc="Done!")

        summary_text = summary_text = f"""
        ## {summary.title}

        #### Overview
        {summary.overview}

        **Language:** {summary.language} | **Sentiment:** {summary.sentiment}

        #### Topics
        {", ".join(summary.topics)}

        #### Transcript Summary
        {summary.transcript_summary}

        #### Visual Summary
        {summary.visual_summary}
        """.strip()

        moments_text = (
            "\n".join(
                [
                    f"- **{m.timestamp_seconds}s** — {m.description}"
                    for m in summary.key_moments
                ]
            )
            or "No key moments detected."
        )

        return (
            meta_text,
            transcript_text,
            visual_text,
            summary_text,
            moments_text,
            _status("✅ Done", success=True),
        )

    except Exception as e:
        traceback.print_exc()
        return _error_outputs(f"Error: {str(e)}")


#### UI Layout
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Multimodal Video Summarization") as demo:
        gr.Markdown("""
# 🎬 Multimodal Video Summarization
### Multimodal Video Summarizer
Upload a video and get a structured summary powered by **Whisper** (speech),
**CLIP** (vision), and an **LLM** (fusion).
---
""")
        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload")
                video_input = gr.Video(label="Video file")

                gr.Markdown("### ⚙️ Settings")
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=["ollama", "openai"],
                        value="ollama",
                        label="LLM Provider",
                    )
                    llm_model = gr.Textbox(
                        value="llama3.2:1b",
                        label="Model name",
                    )
                    use_blip = gr.Checkbox(
                        value=False,
                        label="Use BLIP-2 captions (slower, richer)",
                    )
                    top_k_frames = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="key frames to analyze",
                    )
                    frame_sample_rate = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=1,
                        step=1,
                        label="Sample 1 frame every N seconds",
                    )

                run_btn = gr.Button("🚀 Summarize", variant="primary", size="lg")
                status_out = gr.Markdown("")

            # Right column: outputs
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Results")

                with gr.Tabs():
                    with gr.Tab("📝 Summary"):
                        summary_out = gr.Markdown("")
                    with gr.Tab("⏱️ Key Moments"):
                        moments_out = gr.Markdown("")
                    with gr.Tab("🎙️ Transcript"):
                        transcript_out = gr.Textbox(
                            label="Full transcript",
                            lines=10,
                            interactive=False,
                        )
                    with gr.Tab("🖼️ Visual Analysis"):
                        visual_out = gr.Markdown("")
                    with gr.Tab("📊 Video Info"):
                        meta_out = gr.Markdown("")

        # Wire up
        run_btn.click(
            fn=run_summarize,
            inputs=[
                video_input,
                llm_provider,
                llm_model,
                use_blip,
                top_k_frames,
                frame_sample_rate,
            ],
            outputs=[
                meta_out,
                transcript_out,
                visual_out,
                summary_out,
                moments_out,
                status_out,
            ],
        )

    return demo


# Entry point
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
