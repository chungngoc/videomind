from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    use_blip: bool = Field(
        default=False, description="Enable BLIP-2 for richer captions (slower)"
    )
    llm_provider: str = Field(
        default="ollama", description="LLM provider: ollama or openai"
    )
    llm_model: str = Field(
        default="llama3.2:1b", description="Model name to use for summarization"
    )
    top_k_frames: int = Field(default=5, description="NUmber of key frames to analyze")
    frame_sample_rate: int = Field(
        default=1, description="Sample one frame every N seconds"
    )
