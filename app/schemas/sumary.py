from pydantic import BaseModel, Field

# Define the structured output schema
class KeyMoment(SaseModel):
    timestamp_seconds: float
    description: str

class VideoSummary(BaseModel):
    title: str = Field(description="Inferred title of the video")
    overview: str = Field(description="2-3 sentences overview of the full video")
    key_moments: list[KeyMoment] = Field(description="Most important moments in the video")
    topics: list[str] = Field(description="Main topics covered")
    language: str = Field(description="Detected language of the video")
    sentiment: str = Field(description="Overall tone: positive, neutral, negative")
    transcript_summary: str = Field(description="Summary of the spoken content")
    visual_summary: str = Field(description="Summary of the visual content")