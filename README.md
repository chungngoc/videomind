# videomind
Multimodal video summarizer that fuses **speech**, **visuals**, and **LLM reasoning**
into a single structured summary.

Upload any video → get key moments, topics, and a full transcript.

---

## Architecture
```
Video → Preprocessing → ┌─ Whisper (audio)
                         └─ CLIP / BLIP-2 (frames)
                                    ↓
                           Multimodal fusion (LLM)
                                    ↓
                         Structured summary output
                                    ↓
                    ┌─ FastAPI (REST service)
                    ├─ Gradio (demo UI)
                    └─ MLflow + Docker (MLOps)
```

## Tech Stack

| Layer | Tool |
|---|---|
| Frame analysis | CLIP / BLIP-2 |
| Transcription | OpenAI Whisper |
| Summarization | LLaMA 3 / GPT-4o |
| API | FastAPI |
| Demo | Gradio |
| Experiment tracking | MLflow |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## Quickstart

### Requirements
- Python 3.10+
- ffmpeg installed on your system

### Setuo
```bash
git clone https://github.com/chungngoc/videomind.git
cd videomind
make setup
source venv/bin/activate
```
### Test
```bash
make test # Run all tests
make check # Format + lint + test
```

### Run
```bash
make run        # FastAPI with hot reload → http://localhost:8000
make gradio     # Gradio demo UI
```

## Project Structure
```
videomind/
├── app/
│   ├── api/            # Route handlers
│   ├── core/           # Config and settings
│   ├── pipeline/      # Audio, visual, fusion pipelines
│   └── schemas/        # Pydantic models
├── gradio_demo/        # Gradio interface
├── models/             # Model wrappers
├── mlops/              # Docker, MLflow configs
├── tests/              # Pytest suite
└── .github/workflows/  # CI/CD
```