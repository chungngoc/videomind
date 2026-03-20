from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = "VideoMind"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    upload_dir: Path = Path("/tmp/videomind/uploads")
    output_dir: Path = Path("/tmp/videomind/outputs")

    # Models
    whisper_model_size: str = "base"  # Options: tiny, base, small, medium, large
    clip_model_name: str = "openai/clip-vit-base-patch32"
    blip_model_name: str = "Salesforce/blip2-opt-2.7b"
    llm_provider: str = "ollama"  # Options: openai, ollama
    llm_model: str = "llama3.2:1b"  # For Ollama: llama3.2:1b, llama3.2:7b, llama3.2:70b

    # OpenAI (optional fallback)
    openai_api_key: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"

    # Processing
    frame_sample_rate: int = 1  # Extract 1 frame per second
    max_video_duration: int = 600  # Max video duration in seconds (10 minutes)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Create dirs on import
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
