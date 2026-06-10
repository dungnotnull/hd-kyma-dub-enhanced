# kyma-dub-enhanced AI Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   kyma-dub-enhanced                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Python AI Layer (agent/ + tools/)                      │    │
│  │                                                         │    │
│  │  agent/main.py          ← CLI + FastAPI server          │    │
│  │  agent/orchestrator.py  ← E2E pipeline driver          │    │
│  │  agent/modules/         ← 5 AI pipeline modules        │    │
│  │    asr_transcriber.py   ← Whisper-large-v3             │    │
│  │    script_translator.py ← SeamlessM4T + Claude         │    │
│  │    tts_synthesizer.py   ← XTTS-v2 / Bark              │    │
│  │    lipsync_engine.py    ← Wav2Lip                      │    │
│  │    mos_evaluator.py     ← UTMOS22 + LLM review         │    │
│  │  tools/                                                 │    │
│  │    llm_client.py        ← Claude / GPT / Ollama        │    │
│  │    hf_model_manager.py  ← HuggingFace lazy loader      │    │
│  │    knowledge_updater.py ← ArXiv daily crawl            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  upstream/kyma-dub  (unmodified, reference only)        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/sonpiaz/kyma-dub upstream/kyma-dub
pip install -r requirements.txt

# 2. Install system dependencies
apt-get install -y ffmpeg libsndfile1  # Ubuntu
# brew install ffmpeg                  # macOS

# 3. Install Wav2Lip (optional, for lip-sync)
git clone https://github.com/Rudrabha/Wav2Lip
pip install -r Wav2Lip/requirements.txt
# Download checkpoint: https://github.com/Rudrabha/Wav2Lip#getting-the-weights

# 4. Configure
cp config/.env.example config/.env
# Edit config/.env with your API keys

# 5. Dub a video
python -m agent.main dub my_video.mp4 --languages es --languages fr

# 6. Or start the server
python -m agent.main serve --port 7821
```

## REST API Quick Reference

### POST /api/v1/dub
```bash
curl -X POST http://localhost:7821/api/v1/dub \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/sample.mp4",
    "target_languages": ["es", "fr", "ja"],
    "voice_style": "casual",
    "output_dir": "/output"
  }'
# Response: {"job_id": "a1b2c3d4", "status": "queued", "message": "..."}
```

### GET /api/v1/job/{job_id}
```bash
curl http://localhost:7821/api/v1/job/a1b2c3d4
# Response: {"job_id": "...", "status": "completed", "details": {...}}
```

### POST /api/v1/transcribe
```bash
curl -X POST http://localhost:7821/api/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/data/audio.wav", "language": "en"}'
```

### GET /api/v1/languages
```bash
curl http://localhost:7821/api/v1/languages
```

### POST /api/v1/knowledge/update
```bash
curl -X POST http://localhost:7821/api/v1/knowledge/update
# Response: {"new_entries": 8, "status": "ok"}
```

### GET /api/v1/cost
```bash
curl http://localhost:7821/api/v1/cost
```

### GET /metrics (Prometheus)
```bash
curl http://localhost:7821/metrics
```

## CLI Quick Reference

```bash
# Dub a video
python -m agent.main dub video.mp4 --languages es --languages fr --voice-style formal

# With reference audio for better voice cloning
python -m agent.main dub video.mp4 --languages es --reference-audio reference.wav

# Transcribe only
python -m agent.main transcribe audio.wav --output ./transcripts

# Check job status
python -m agent.main status abc12345

# Update knowledge base
python -m agent.main update-knowledge

# Cost report
python -m agent.main cost-report --days 30
```

## Cross-Agent Integration

### With life-chronicle-agent (Folder 1)
```python
# life-chronicle-agent can send diary narration audio to kyma-dub-enhanced
# for multilingual short film dubbing
import aiohttp

async def request_dubbing(video_path: str, languages: list[str]) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://kyma-dub-agent:7821/api/v1/dub",
            json={
                "video_path": video_path,
                "target_languages": languages,
                "voice_style": "documentary",
            }
        ) as resp:
            return await resp.json()
```

### With ai-benchmark-agent (Folder 22)
```python
# ai-benchmark-agent can intercept LLM calls from kyma-dub-enhanced
# to measure translation quality, latency, and cost per language pair
```

### With academic-research-enhanced (Folder 18)
```python
# academic-research-enhanced can feed new TTS/lip-sync papers directly
# into kyma-dub-enhanced SECOND-KNOWLEDGE-BRAIN.md via knowledge update API
```

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `kymadub_jobs_total` | Counter | Total dubbing jobs processed |
| `kymadub_quality_passed_total` | Counter | Jobs passing MOS ≥ 3.5 quality gate |
| `kymadub_quality_failed_total` | Counter | Jobs failing quality gate |
| `kymadub_knowledge_papers_total` | Gauge | Total papers in knowledge base |
| `kymadub_llm_cost_usd_30d` | Gauge | LLM API spend last 30 days (USD) |

## Adding a New Target Language

1. Add language code to `SUPPORTED_LANGUAGES` in `agent/modules/script_translator.py`
2. Add SeamlessM4T language mapping in `_to_seamless_lang()` function
3. Add Bark voice preset in `BARK_LANG_PREFIXES` dict in `tts_synthesizer.py`
4. Verify XTTS-v2 supports the language (check `XTTS_SUPPORTED_LANGUAGES` set)
5. Test: `python -m agent.main dub test.mp4 --languages <new_lang>`

## Production Hardening Checklist

- [ ] Set `ANTHROPIC_API_KEY` (required for best translation quality)
- [ ] Set `HF_TOKEN` (required for pyannote gated models)
- [ ] Download Wav2Lip checkpoint (required for lip-sync)
- [ ] Mount output/models/data volumes (Docker)
- [ ] Set `LOG_LEVEL=WARNING` in production
- [ ] Enable GPU profile for 3–5× faster processing
- [ ] Monitor `/metrics` in Grafana (connect to Prometheus)
- [ ] Set up Ollama for offline/privacy fallback: `docker compose --profile gpu up ollama`
- [ ] Run `python -m agent.main update-knowledge` on first start to seed SECOND-KNOWLEDGE-BRAIN.md
