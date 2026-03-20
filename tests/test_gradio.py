from gradio_demo.app import build_ui, run_summarize


def test_ui_build_without_error():
    """Gradio UI should build without raising any exceptions."""
    demo = build_ui()
    assert demo is not None


def test_no_video_returns_error():
    """Calling run_summarize with no video should return error state."""
    outputs = run_summarize(
        video_path=None,
        llm_provider="ollama",
        llm_model="llama3.2:1b",
        use_blip=False,
        top_k_frames=5,
        frame_sample_rate=1,
    )
    # Last output is the status message
    status = outputs[-1]
    assert "❌" in status
