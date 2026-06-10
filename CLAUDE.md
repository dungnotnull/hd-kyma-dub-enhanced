# kyma-dub-enhanced — Neural Dubbing & Voice Synthesis Platform

**Build Phase:** Phase 0 → Phase 7 (Full Production Build)
**Cluster:** A — Autonomous Creative Media Agents
**Upstream:** github.com/sonpiaz/kyma-dub (fork, pinned at latest stable commit)

## Problem Statement

Dubbing video content into multiple languages is prohibitively expensive: a 60-minute film requires hours of studio recording, professional voice actors, lip-sync technicians, and post-production work. kyma-dub-enhanced solves this autonomously — given a source video and target language(s), the agent transcribes the original speech with word-level timestamps, translates the script with idiomatic adaptation via LLM, clones the original speaker voice using XTTS-v2 zero-shot synthesis from a 6-second reference clip, aligns lip movements via Wav2Lip, runs automated MOS quality evaluation, and delivers a broadcast-quality dubbed video in 20+ languages with no human intervention required.

## Agent Architecture

```
Source Video
      ↓
Step 1: ASR Transcription     (modules/asr_transcriber.py)
        Whisper-large-v3 → aligned transcript JSON with word-level timestamps
      ↓
Step 2: Script Translation    (modules/script_translator.py)
        SeamlessM4T-v2 base translation → Claude idiomatic adaptation
        → translated + duration-fitted script per target language
      ↓
Step 3: TTS Synthesis         (modules/tts_synthesizer.py)
        XTTS-v2 zero-shot voice cloning (6s reference) → dubbed WAV per segment
        Bark fallback if XTTS-v2 unavailable
      ↓
Step 4: Lip-Sync              (modules/lipsync_engine.py)
        Wav2Lip face detection → lip-synced video frames → ffmpeg MP4 assembly
      ↓
Step 5: MOS Quality Gate      (modules/mos_evaluator.py)
        UTMOS22 neural MOS ≥ 3.5 AND LLM script quality ≥ 4.0/5.0
        → pass: deliver final MP4 | fail: retry TTS with adjusted parameters
      ↓
Final Dubbed MP4 per target language
```

## Module List

| File | Responsibility |
|------|----------------|
| `agent/modules/asr_transcriber.py` | Whisper-large-v3 transcription with word-level timestamps; faster-whisper fallback for speed |
| `agent/modules/script_translator.py` | SeamlessM4T-v2 base translation + Claude idiomatic rewrite; duration-fitting; 20+ language support |
| `agent/modules/tts_synthesizer.py` | XTTS-v2 zero-shot voice cloning; Bark fallback; segment-wise synthesis with prosody control |
| `agent/modules/lipsync_engine.py` | Wav2Lip face detection + lip-sync generation; ffmpeg frame assembly; face-crop preprocessing |
| `agent/modules/mos_evaluator.py` | UTMOS22 neural MOS prediction; LLM naturalness + accuracy scoring; 3-attempt retry gate |

## Tools Used

| File | Responsibility |
|------|----------------|
| `agent/tools/video_processor.py` | ffmpeg wrapper: extract audio, merge video+audio, resize frames, encode MP4 |
| `tools/knowledge_updater.py` | Crawl ArXiv cs.SD/cs.CV/cs.CL + Semantic Scholar + Interspeech/ICASSP → SECOND-KNOWLEDGE-BRAIN.md |
| `tools/llm_client.py` | Unified Claude/GPT-4o/Ollama client with streaming and exponential-backoff retry |
| `tools/hf_model_manager.py` | Lazy-load XTTS-v2/Wav2Lip/Whisper/SeamlessM4T/UTMOS22; CUDA auto-detect; 600s idle unload |

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|------------|
| `openai/whisper-large-v3` | ASR + word-level timestamps | SOTA WER on multilingual ASR; supports 99 languages; word timestamps for alignment |
| `coqui/XTTS-v2` | Zero-shot TTS voice cloning | 6s reference → cloned voice in 17 languages; beats Bark on naturalness (MOS 4.2 vs 3.8) |
| `facebook/seamless-m4t-v2-large` | Speech/text translation | 100+ language pairs in one model; speech-to-text + text-to-text in single pass |
| `Rudrabha/Wav2Lip` | Lip-sync video generation | Best open-source lip-sync; trained on LRS2/LRS3 benchmarks; real-time capable |
| `microsoft/UTMOS22` | Neural MOS evaluation | Pearson r=0.945 with human ratings on UTMOS challenge leaderboard |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity | Translation quality dedup and script segment matching |

## LLM API Integration

| Provider | Model | Use Cases |
|----------|-------|-----------|
| Claude (primary) | `claude-opus-4-8` | Idiomatic script adaptation, translation quality review, research synthesis, timing adjustment |
| OpenAI (fallback) | `gpt-4o` | Same use cases; multimodal frame quality review if needed |
| Ollama (offline) | `llama3` | Privacy mode: local translation correction and script rewriting |

**Key prompt templates:**
1. `IDIOMATIC_ADAPT_PROMPT` — rewrite literal translation into natural target-language idioms
2. `QUALITY_REVIEW_PROMPT` — assess translated script for naturalness (0–5) and semantic accuracy (0–5)
3. `TIMING_ADJUST_PROMPT` — shorten/expand translated text to match source segment duration ±10%
4. `RESEARCH_SYNTHESIS_PROMPT` — synthesize TTS/lip-sync papers into improvement recommendations

## Knowledge Crawl Sources

| Source | Categories / Queries | Frequency |
|--------|----------------------|-----------|
| ArXiv API | cs.SD, cs.CV, cs.CL (TTS, neural dubbing, lip sync, voice cloning) | Daily |
| Semantic Scholar | "neural TTS voice cloning", "lip sync video synthesis", "speech translation dubbing", "talking head" | Daily |
| Papers with Code | TTS leaderboard, lip sync leaderboard, ASR leaderboard | Weekly |
| Interspeech proceedings | TTS, ASR, voice conversion, multilingual speech | Weekly |
| ICASSP proceedings | Speech synthesis, neural codec, voice cloning | Weekly |

## Supporting Tools

- `tools/knowledge_updater.py` — ArXiv + Semantic Scholar + PwC crawler → SECOND-KNOWLEDGE-BRAIN.md (daily at 06:00)
- `tools/llm_client.py` — Claude/GPT-4o/Ollama unified client with streaming, retry (1s/2s/4s backoff), cost tracking
- `tools/hf_model_manager.py` — Lazy-load registry for 6 HuggingFace models; CUDA auto-detect; 600s idle unload

## Active Development Tasks

- [ ] Phase 0: Fork kyma-dub; document upstream capabilities; establish baseline WER/MOS metrics
- [ ] Phase 1: ASR transcriber (Whisper-large-v3 with word timestamps) + video_processor ffmpeg wrapper
- [ ] Phase 2: Script translator (SeamlessM4T-v2 + Claude idiomatic adaptation + duration fitting)
- [ ] Phase 3: TTS synthesizer (XTTS-v2 zero-shot voice cloning + Bark fallback)
- [ ] Phase 4: Lip-sync engine (Wav2Lip face crop + lip-sync + ffmpeg assembly)
- [ ] Phase 5: MOS evaluator (UTMOS22 + LLM quality gate + 3-attempt retry loop)
- [ ] Phase 6: SECOND-KNOWLEDGE-BRAIN daily pipeline (ArXiv cs.SD + cs.CV + Semantic Scholar)
- [ ] Phase 7: Docker containerization + full test suite execution
