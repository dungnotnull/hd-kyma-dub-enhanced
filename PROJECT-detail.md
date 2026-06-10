# kyma-dub-enhanced — Full Technical Specification

## Executive Summary

kyma-dub-enhanced is an autonomous neural dubbing platform that transforms any video into a professionally dubbed version in 20+ languages. It forks the kyma-dub open-source project and adds a full AI pipeline: Whisper-large-v3 ASR, SeamlessM4T-v2 + LLM translation, XTTS-v2 zero-shot voice cloning, Wav2Lip lip-sync, and UTMOS22 automated quality gating. The agent runs end-to-end with zero human intervention and a quality gate that ensures no output is delivered below MOS 3.5.

## Problem Statement

Manual video dubbing costs $15–$50 per minute of content. For a 90-minute film with 4 target languages that is $5,400–$18,000 plus weeks of production time. Existing automated tools either produce robotic-sounding speech (eSpeak, Festival), ignore lip synchronization entirely, or require pre-recorded voice datasets. kyma-dub-enhanced closes all three gaps simultaneously using SOTA pretrained models.

## Target Users & Use Cases

| User | Trigger | Agent Does |
|------|---------|------------|
| Content creator | Upload YouTube video + select "Spanish, French" | Full dubbing pipeline → 2 MP4 files delivered |
| Film distributor | Upload film + select 10 languages | Batch parallel dubbing → 10 language versions |
| E-learning platform | Upload course video | Auto-translate + dub all modules with instructor voice clone |
| News broadcaster | Live stream clip | Near-real-time dubbing via streaming ASR + TTS pipeline |

## Agent Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Input: source_video.mp4, target_languages: ["es", "fr", "ja"]           │
│  Optional: reference_audio.wav (for voice cloning, 6s minimum)           │
└─────────────────────┬────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  KymaDubOrchestrator  (agent/orchestrator.py)                           │
│                                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────┐  │
│  │ ASR         │→ │ Script       │→ │ TTS           │→ │ Lip-Sync  │  │
│  │ Transcriber │  │ Translator   │  │ Synthesizer   │  │ Engine    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  └───────────┘  │
│         ↓                ↓                  ↓                  ↓        │
│  Word timestamps  Idiomatic script   Dubbed WAV segments  Synced frames  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ MOS Evaluator (quality gate: MOS ≥ 3.5 AND script score ≥ 4.0)  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓ pass                    ↓ fail (max 3 retries)                 │
│  Final MP4 assembled          Retry TTS with adjusted parameters         │
└─────────────────────────────────────────────────────────────────────────┘
         ↓                ↓                   ↓
  HuggingFace        LLM API           ffmpeg/OpenCV
  (Whisper/XTTS/    (Claude primary,   (video assembly,
  SeamlessM4T/      GPT-4o fallback,   frame extraction,
  Wav2Lip/UTMOS22)  Ollama offline)    audio merging)
         ↓
  Output: dubbed_{lang}.mp4 per target language
```

## Full Module Catalog

### Module 1: `agent/modules/asr_transcriber.py` — ASR Transcription

**Responsibility:** Transcribe source video audio with word-level timestamps for downstream alignment.

**Inputs:**
- `video_path: str` — path to source video
- `language: str | None` — source language code (None = auto-detect)
- `beam_size: int` — Whisper beam size (default 5)

**Outputs:**
- `TranscriptResult` dataclass:
  - `segments: list[Segment]` — each with `start, end, text, words: list[Word]`
  - `language: str` — detected language
  - `duration_seconds: float`

**Tools called:** `HFModelManager.load_whisper()`, `VideoProcessor.extract_audio()`

**Quality gate:** Word count ≥ 5 AND `no_speech_prob < 0.8` for each segment

---

### Module 2: `agent/modules/script_translator.py` — Script Translation

**Responsibility:** Translate transcript into target language with idiomatic adaptation and duration fitting.

**Inputs:**
- `transcript: TranscriptResult`
- `target_language: str` — ISO 639-1 code (es, fr, ja, zh, de, ko, …)
- `voice_style: str` — formal/casual/dramatic/documentary

**Outputs:**
- `TranslationResult` dataclass:
  - `segments: list[TranslatedSegment]` — with `original_text, translated_text, adapted_text, duration_fitted_text`
  - `quality_scores: dict` — naturalness and accuracy per segment
  - `target_language: str`

**Tools called:** `HFModelManager.load_seamless_m4t()` for base translation, `LLMClient.complete()` for idiomatic adaptation

**LLM prompt:** `IDIOMATIC_ADAPT_PROMPT` — rewrites literal machine translation into natural speech idioms

**Quality gate:** Semantic similarity (MiniLM-L6-v2) between source and back-translated text ≥ 0.75

---

### Module 3: `agent/modules/tts_synthesizer.py` — TTS Voice Cloning

**Responsibility:** Synthesize dubbed audio segments using XTTS-v2 zero-shot voice cloning.

**Inputs:**
- `translation: TranslationResult`
- `reference_audio_path: str` — 6–30 second WAV clip of target voice
- `target_language: str`
- `speed_factor: float` — 0.8–1.2 to fit segment duration

**Outputs:**
- `SynthesisResult` dataclass:
  - `segments: list[AudioSegment]` — WAV file path per segment
  - `full_audio_path: str` — concatenated dubbed audio
  - `model_used: str` — "xtts-v2" or "bark"

**Tools called:** `HFModelManager.load_xtts()`, `HFModelManager.load_bark()` (fallback)

**Quality gate:** Audio duration within ±15% of target segment duration

---

### Module 4: `agent/modules/lipsync_engine.py` — Lip-Sync Generation

**Responsibility:** Generate lip-synced video by aligning dubbed audio with speaker face movements.

**Inputs:**
- `video_path: str` — source video
- `dubbed_audio_path: str` — full dubbed audio WAV
- `face_crop: bool` — True for portrait/closeup; False for full frame

**Outputs:**
- `LipSyncResult` dataclass:
  - `output_video_path: str` — lip-synced video (no audio embedded yet)
  - `sync_confidence: float` — Wav2Lip confidence score (0–1)
  - `frames_processed: int`

**Tools called:** `HFModelManager.load_wav2lip()`, `VideoProcessor.extract_frames()`, `VideoProcessor.merge_audio_video()`

**Quality gate:** `sync_confidence ≥ 0.85` (Wav2Lip face detection confidence)

---

### Module 5: `agent/modules/mos_evaluator.py` — MOS Quality Evaluation

**Responsibility:** Evaluate dubbed audio quality using UTMOS22 neural MOS predictor and LLM script review.

**Inputs:**
- `audio_path: str` — dubbed audio WAV
- `original_script: str` — source language text
- `dubbed_script: str` — target language text
- `target_language: str`

**Outputs:**
- `QualityResult` dataclass:
  - `mos_score: float` — UTMOS22 predicted MOS (1.0–5.0)
  - `script_naturalness: float` — LLM score (0–5)
  - `script_accuracy: float` — LLM score (0–5)
  - `passed: bool` — MOS ≥ 3.5 AND naturalness ≥ 4.0 AND accuracy ≥ 4.0
  - `failure_reason: str | None`

**Tools called:** `HFModelManager.load_utmos()`, `LLMClient.complete()` with `QUALITY_REVIEW_PROMPT`

**Quality gate:** `passed == True` OR max retries exceeded (3 attempts)

---

## HuggingFace Model Selection

| Model | Task | Benchmark | Score | Alternatives Considered |
|-------|------|-----------|-------|------------------------|
| `openai/whisper-large-v3` | ASR | Common Voice WER | 8.4% avg | FunASR-Paraformer: no word timestamps |
| `coqui/XTTS-v2` | Zero-shot TTS | UTMOS MOS | 4.17 | Bark: 3.8 MOS; no duration control |
| `facebook/seamless-m4t-v2-large` | S2TT + T2T translation | FLORES-200 BLEU | 42.3 | MarianMT: single language pair only |
| `Rudrabha/Wav2Lip` | Lip sync | LRS3 LSE-C | 7.02 | SadTalker: slower; requires portrait |
| `microsoft/UTMOS22` | MOS prediction | UTMOS22 challenge | r=0.945 | NISQA: r=0.89; weaker on TTS audio |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity | MTEB STS | 56.26 | BGE-large: overkill for dedup task |

## LLM API Integration Spec

### Provider Chain
```
PROVIDER_PRIORITY = ["claude", "openai", "ollama"]
```

### Prompt Templates

**IDIOMATIC_ADAPT_PROMPT**
```
System: You are a professional dubbing script writer for {target_language_name}.
Your task: rewrite the literal machine translation into natural spoken dialogue.
Rules:
1. Match the emotional tone and register of the original
2. Fit the translated text within {duration_budget_ms}ms when spoken at normal pace
3. Preserve all proper nouns, technical terms, and named entities
4. Output ONLY the adapted text — no explanations

Original ({source_lang}): {original_text}
Literal translation: {literal_translation}
Adapted script:
```

**QUALITY_REVIEW_PROMPT**
```
Evaluate this dubbed script segment for a {target_language_name} audience.
Score two dimensions from 0.0 to 5.0:
- naturalness: Does it sound like natural spoken {target_language_name}? (5=native speaker quality)
- accuracy: Does it preserve the meaning of the original? (5=perfect meaning preservation)

Original: {original_text}
Dubbed script: {dubbed_text}

Respond with JSON only: {"naturalness": X.X, "accuracy": X.X, "issues": ["...", "..."]}
```

**TIMING_ADJUST_PROMPT**
```
The dubbed text for a {duration_ms}ms segment is too {long_or_short} by {delta_ms}ms.
Shorten/expand the text while preserving meaning. Output ONLY the adjusted text.
Current text: {current_text}
```

### Token Budget Estimates
- Idiomatic adaptation: ~300 input + ~200 output tokens per segment
- Quality review: ~200 input + ~100 output tokens per segment
- Research synthesis: ~4000 input + ~2000 output tokens per weekly run

## E2E Execution Flow

```
1. receive_job(video_path, target_languages, reference_audio_path)
   └─ validate inputs: video exists, reference audio ≥ 6s, target_lang in SUPPORTED_LANGUAGES

2. extract_audio(video_path)
   └─ ffmpeg: extract WAV 16kHz mono → {job_id}/source_audio.wav

3. transcribe(source_audio)
   └─ Whisper-large-v3 → TranscriptResult with word timestamps
   └─ quality_gate: n_segments > 0 AND avg no_speech_prob < 0.8

4. FOR each target_language:
   4a. translate(transcript, target_language)
       └─ SeamlessM4T-v2 base translation per segment
       └─ Claude: idiomatic adaptation (IDIOMATIC_ADAPT_PROMPT)
       └─ Claude: timing adjustment if duration delta > 15% (TIMING_ADJUST_PROMPT)
       └─ quality_gate: back-translation semantic similarity ≥ 0.75

   4b. synthesize_voice(translation, reference_audio)
       └─ XTTS-v2 zero-shot: generate WAV per segment
       └─ concatenate segments → full dubbed audio
       └─ quality_gate: duration within ±15% of source

   4c. generate_lipsync(source_video, dubbed_audio)
       └─ Wav2Lip: face detection → lip-sync frames → ffmpeg encode
       └─ quality_gate: sync_confidence ≥ 0.85

   4d. evaluate_quality(dubbed_audio, scripts)
       └─ UTMOS22: MOS prediction on dubbed audio
       └─ LLM: naturalness + accuracy review
       └─ IF failed AND retry_count < 3:
          └─ adjust TTS speed_factor → retry from 4b
       └─ IF failed after 3 retries: deliver with quality warning flag

   4e. assemble_final_video(lipsync_video, dubbed_audio)
       └─ ffmpeg: merge lip-synced frames + dubbed audio → dubbed_{lang}.mp4

5. save_job_results(memory_manager)
   └─ SQLite: job metadata, MOS scores, language, processing time, cost

6. return DubbingResult(output_files, quality_reports, cost_summary)
```

## SECOND-KNOWLEDGE-BRAIN.md Integration

- **Source:** ArXiv cs.SD, cs.CV, cs.CL (daily); Semantic Scholar 5 queries (daily); PwC TTS/lip-sync leaderboards (weekly)
- **Crawl config:** `knowledge_updater.py` — ArXiv Atom XML API + Semantic Scholar graph API
- **Dedup strategy:** SHA-256 hash of DOI/URL stored in `knowledge_hashes` SQLite table
- **Append target:** `## Knowledge Update Log` section in `SECOND-KNOWLEDGE-BRAIN.md`
- **Feedback loop:** New SOTA TTS model → `hf_model_manager.py` registry update → improved output quality

## Quality Gates (7 Gates)

| Gate | Condition | Action on Failure |
|------|-----------|-------------------|
| QG-1 | Input video exists AND duration > 5s | Reject job with error |
| QG-2 | Reference audio duration ≥ 6s | Use Bark fallback TTS |
| QG-3 | Whisper avg no_speech_prob < 0.8 | Flag segment as noisy; use subtitle input |
| QG-4 | Back-translation similarity ≥ 0.75 | Request LLM re-adaptation |
| QG-5 | TTS duration within ±15% of source | Apply speed_factor adjustment (0.85–1.15) |
| QG-6 | Wav2Lip sync_confidence ≥ 0.85 | Skip lip-sync; use original video frames |
| QG-7 | UTMOS22 MOS ≥ 3.5 AND LLM scores ≥ 4.0 | Retry TTS (max 3 attempts); deliver with warning flag |

## Test Scenarios

See `tests/test-scenarios.md` for 8 full end-to-end scenarios.

## Key Design Decisions

1. **XTTS-v2 over Bark:** XTTS-v2 delivers MOS 4.17 vs Bark 3.8 and supports duration control; Bark retained as fallback for systems without GPU.
2. **SeamlessM4T-v2 + LLM two-stage translation:** SeamlessM4T provides fast base translation; Claude rewrites for natural idiom — neither alone achieves both speed and quality.
3. **Wav2Lip over SadTalker:** Wav2Lip is 3× faster and works on full-frame video; SadTalker requires portrait crops and struggles with crowd scenes.
4. **UTMOS22 as automated MOS gate:** Pearson r=0.945 with human listeners; eliminates manual QA bottleneck.
5. **Sidecar architecture:** AI pipeline runs as separate Python service; kyma-dub upstream Go/Python code is unmodified; upgrade isolation is maintained.
6. **Daily ArXiv crawl:** TTS/lip-sync research advances rapidly (new models monthly); daily crawl ensures `SECOND-KNOWLEDGE-BRAIN.md` stays current.
7. **3-retry TTS loop:** Voice quality failure is usually fixable by adjusting speed_factor ±10%; retry avoids full re-run cost while maintaining quality gate.
