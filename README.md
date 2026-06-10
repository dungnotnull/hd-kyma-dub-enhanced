# kyma-dub-enhanced — Neural Dubbing & Voice Synthesis Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous neural dubbing platform that transforms any video into a professionally dubbed version in 20+ languages. Using state-of-the-art AI models (Whisper, SeamlessM4T, XTTS-v2, Wav2Lip, UTMOS22), the platform delivers broadcast-quality dubbed videos with zero human intervention.

## Features

- **Zero-Shot Voice Cloning** — Clone any speaker's voice from a 6-second audio clip
- **20 Language Support** — English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Polish, Dutch, Swedish, Vietnamese, Indonesian, Thai, Persian
- **Lip Synchronization** — Wav2Lip integration for realistic lip-synced output
- **Automated Quality Control** — UTMOS22 neural MOS + LLM script review with 3-attempt retry
- **Self-Improving** — Daily ArXiv research crawler keeps knowledge base current
- **Production-Ready** — CLI, REST API, Docker deployment, Prometheus metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/kyma-dub-enhanced.git
cd kyma-dub-enhanced

# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install -y ffmpeg libsndfile1
# macOS:
brew install ffmpeg
```

### Configuration

```bash
# Copy environment template
cp config/.env.example config/.env

# Edit config/.env with your API keys
# Required: ANTHROPIC_API_KEY or OPENAI_API_KEY
# Optional: HF_TOKEN for gated HuggingFace models
```

### Usage

```bash
# Dub a video into Spanish
python -m agent.main dub video.mp4 --languages es

# Dub into multiple languages
python -m agent.main dub video.mp4 --languages es --languages fr --languages ja

# With custom reference audio for voice cloning
python -m agent.main dub video.mp4 --languages es --reference-audio voice.wav

# Transcribe only
python -m agent.main transcribe video.mp4 --output ./transcripts

# Start the REST API server
python -m agent.main serve --port 7821

# Update knowledge base
python -m agent.main update-knowledge

# View cost report
python -m agent.main cost-report --days 30
```

## REST API

### Start Server

```bash
python -m agent.main serve --host 0.0.0.0 --port 7821
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/dub` | POST | Submit dubbing job |
| `/api/v1/transcribe` | POST | Transcribe audio |
| `/api/v1/job/{job_id}` | GET | Get job status |
| `/api/v1/languages` | GET | List supported languages |
| `/api/v1/knowledge/update` | POST | Trigger knowledge update |
| `/api/v1/cost` | GET | Get cost summary |
| `/metrics` | GET | Prometheus metrics |

### Example Request

```bash
curl -X POST http://localhost:7821/api/v1/dub \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "target_languages": ["es", "fr"],
    "voice_style": "casual",
    "output_dir": "./output"
  }'
```

## Docker Deployment

### CPU Mode

```bash
docker-compose up -d
```

### GPU Mode

```bash
docker-compose --profile gpu up -d
```

### Check Logs

```bash
docker-compose logs -f kyma-dub-agent
```

## Architecture

```
Source Video
      ↓
Step 1: ASR Transcription (Whisper-large-v3)
        → Transcript with word timestamps
      ↓
Step 2: Script Translation (SeamlessM4T-v2 + Claude)
        → Idiomatic translation with duration fitting
      ↓
Step 3: TTS Synthesis (XTTS-v2 zero-shot voice cloning)
        → Dubbed WAV per segment
      ↓
Step 4: Lip-Sync (Wav2Lip)
        → Lip-synced video frames
      ↓
Step 5: MOS Quality Gate (UTMOS22 + LLM review)
        → Pass: deliver | Fail: retry TTS
      ↓
Final Dubbed MP4
```

## Modules

| Module | Description | Model |
|--------|-------------|-------|
| `agent/modules/asr_transcriber.py` | Speech recognition with word timestamps | Whisper-large-v3 |
| `agent/modules/script_translator.py` | Two-stage translation (base + idiomatic) | SeamlessM4T-v2 + Claude |
| `agent/modules/tts_synthesizer.py` | Zero-shot voice cloning | XTTS-v2 / Bark |
| `agent/modules/lipsync_engine.py` | Lip synchronization | Wav2Lip |
| `agent/modules/mos_evaluator.py` | Quality evaluation | UTMOS22 + LLM |
| `tools/llm_client.py` | Unified LLM client | Claude / OpenAI / Ollama |
| `tools/hf_model_manager.py` | Model lazy loading | HuggingFace |
| `tools/knowledge_updater.py` | Research crawler | ArXiv + Semantic Scholar |

## Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Output MOS | ≥ 4.0 | UTMOS22 neural predictor |
| Translation BLEU | ≥ 38 | FLORES-200 test set |
| Lip-sync coverage | ≥ 80% | Wav2Lip sync_confidence |
| E2E latency (10-min) | ≤ 15 min | GPU (T4) benchmark |

## Supported Languages

- English (en), Spanish (es), French (fr), German (de)
- Italian (it), Portuguese (pt), Russian (ru), Chinese (zh)
- Japanese (ja), Korean (ko), Arabic (ar), Hindi (hi)
- Turkish (tr), Polish (pl), Dutch (nl), Swedish (sv)
- Vietnamese (vi), Indonesian (id), Thai (th), Persian (fa)

## Configuration

Edit `config/agent_config.yaml` for advanced settings:

```yaml
asr:
  model_size: large-v3
  beam_size: 5

translation:
  default_voice_style: casual
  timing_tolerance_pct: 0.15

tts:
  primary_model: xtts-v2
  fallback_model: bark

lipsync:
  checkpoint_path: Wav2Lip/checkpoints/wav2lip_gan.pth
  sync_confidence_threshold: 0.85

quality:
  mos_threshold: 3.5
  naturalness_threshold: 4.0
  max_retries: 3
```

## Testing

```bash
# Run all tests
pytest tests/test_agent.py -v

# Run specific test class
pytest tests/test_agent.py::TestASRTranscriber -v

# Run with coverage
pytest tests/test_agent.py --cov=agent --cov=tools
```

## Documentation

- [Phase Tracking](PROJECT-DEVELOPMENT-PHASE-TRACKING.md) — Development progress
- [Technical Spec](PROJECT-detail.md) — Full architecture documentation
- [Knowledge Base](SECOND-KNOWLEDGE-BRAIN.md) — Research papers & state-of-the-art
- [Integration Guide](ai_layer/patches/kyma_dub_ai_integration.md) — API reference
- [Test Scenarios](tests/test-scenarios.md) — 8 end-to-end scenarios

## Dependencies

See [requirements.txt](requirements.txt) for full list.

Key dependencies:
- `openai-whisper` — ASR transcription
- `transformers` — HuggingFace models (SeamlessM4T, UTMOS22)
- `TTS` — XTTS-v2 voice cloning
- `opencv-python` — Face detection for lip-sync
- `fastapi` / `uvicorn` — REST API server
- `anthropic` / `openai` — LLM clients

## License

MIT License — See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Upstream: [kyma-dub](https://github.com/sonpiaz/kyma-dub)
- Whisper: OpenAI
- XTTS-v2: Coqui AI
- Wav2Lip: Rudrabha/Wav2Lip
- SeamlessM4T: Meta AI
- UTMOS22: UTokyo-SaruLab

