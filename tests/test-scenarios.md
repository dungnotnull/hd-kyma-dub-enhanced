# kyma-dub-enhanced — Test Scenarios

## Scenario 1: English → Spanish Full Pipeline (Golden Path)

**Trigger:** User runs `python -m agent.main dub sample_video.mp4 --languages es --output-dir ./output`

**Input:** 5-minute English talking-head video (single speaker, clear speech)

**Expected Steps:**
1. ffmpeg extracts `source_audio.wav` (16kHz mono)
2. ffmpeg extracts `reference_audio.wav` (first 10s, 22050Hz)
3. Whisper-large-v3 transcribes → 30 segments, language=en, word timestamps present
4. SeamlessM4T-v2 base translation (en→es) per segment
5. Claude IDIOMATIC_ADAPT_PROMPT rewrites each segment in natural Spanish
6. XTTS-v2 synthesizes 30 WAV segments using reference voice
7. Wav2Lip generates lip-synced video frames
8. UTMOS22 predicts MOS ≥ 3.8 → quality gate PASSES
9. ffmpeg assembles final `dubbed_es.mp4`
10. Report generated: MOS, naturalness, accuracy scores

**Expected Output:** `output/{job_id}/es/dubbed_es.mp4` with MOS ≥ 3.5, naturalness ≥ 4.0

---

## Scenario 2: Multi-Language Batch Dubbing (3 Languages in Parallel)

**Trigger:** `python -m agent.main dub video.mp4 --languages es --languages fr --languages ja`

**Input:** 10-minute documentary with clear narrator voice

**Expected Behavior:**
- Single ASR transcription for all languages (step 3 runs once)
- Languages processed in parallel via `asyncio.gather`
- Each language: SeamlessM4T translation + Claude adaptation independently
- XTTS-v2 with same reference audio but different target language
- Three final MP4 files delivered

**Expected Output:**
- `dubbed_es.mp4`, `dubbed_fr.mp4`, `dubbed_ja.mp4` all in output directory
- Parallel processing completes within 30% of sequential time on same GPU

---

## Scenario 3: MOS Quality Gate Triggers Retry

**Trigger:** First TTS attempt produces robotic-sounding audio (MOS 3.1)

**Input:** Video with unusual prosody; TTS struggles on first attempt

**Expected Behavior:**
1. Attempt 1: UTMOS22 → MOS 3.1 (< threshold 3.5) → FAIL
2. `compute_retry_speed_factor` adjusts speed_factor from 1.0 to 0.95
3. Attempt 2: Re-synthesize TTS with speed_factor=0.95 → UTMOS22 → MOS 3.6 → PASS
4. Quality gate PASSES on attempt 2
5. `retries_used=1` recorded in result

**Expected Output:** Final video delivered with MOS 3.6, no quality warning flag

---

## Scenario 4: No-Face Video — Graceful Lip-Sync Skip

**Trigger:** Dub an animation video (cartoon characters, no human face)

**Input:** 3-minute animated video with voice narration

**Expected Behavior:**
1. ASR transcription succeeds normally
2. Translation + TTS synthesis succeed normally
3. Wav2Lip face detection: `face_detected=False`
4. `lipsync_engine._skip_lipsync()` called: reason "No face detected"
5. ffmpeg merges dubbed audio with original animation frames
6. `LipSyncResult.skipped=True`, `sync_confidence=0.0`
7. Final video: original animation + dubbed audio (no face modification)

**Expected Output:** `dubbed_es.mp4` with correct dubbed audio; report notes lip-sync skipped

---

## Scenario 5: XTTS-v2 Unavailable — Bark Fallback

**Trigger:** System has no GPU / XTTS-v2 fails to load

**Input:** Any dubbing request on CPU-only machine

**Expected Behavior:**
1. `TTSSynthesizer._load_xtts()` fails (ImportError or CUDA OOM)
2. Falls back to `_synthesize_bark()` automatically
3. Bark voice preset selected based on target language
4. Quality may be lower (MOS ~3.2–3.8) but pipeline completes
5. `SynthesisResult.model_used="bark"` reported

**Expected Output:** Dubbed video delivered; model_used="bark" in report

---

## Scenario 6: Reference Audio Too Short — Auto-Extraction

**Trigger:** User does not provide `--reference-audio`; source video has clean speech

**Expected Behavior:**
1. Orchestrator calls `video_processor.extract_reference_audio(start_sec=0, duration_sec=10)`
2. Extracts first 10 seconds of source audio (22050Hz mono)
3. XTTS-v2 uses auto-extracted reference for voice cloning
4. Pipeline proceeds normally

**Expected Output:** Voice in dubbed video resembles original speaker; no error raised

---

## Scenario 7: All LLM Providers Down — Graceful Degradation

**Trigger:** `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` unset; Ollama not running

**Expected Behavior:**
1. `ScriptTranslator._adapt_idiomatically()` → LLM call fails → returns literal SeamlessM4T translation
2. `MOSEvaluator._llm_review()` → returns default scores (4.0, 4.0)
3. Pipeline completes with SeamlessM4T translation only
4. Quality report shows `script_naturalness=4.0` (default; not LLM-evaluated)
5. Log warning: "LLM providers unavailable; using SeamlessM4T translation without adaptation"

**Expected Output:** Dubbed video delivered (lower naturalness); no crash

---

## Scenario 8: REST API End-to-End Integration

**Trigger:** POST /api/v1/dub with JSON body

**Request:**
```json
{
  "video_path": "/data/sample.mp4",
  "target_languages": ["es", "fr"],
  "voice_style": "formal",
  "output_dir": "/output"
}
```

**Expected Behavior:**
1. Server returns immediately: `{"job_id": "abc12345", "status": "queued"}`
2. Background job processes dubbing pipeline
3. GET /api/v1/job/abc12345 → status updates: running → completed
4. GET /metrics → Prometheus metrics updated (kymadub_jobs_total, kymadub_quality_passed_total)

**Expected Output:** Both `dubbed_es.mp4` and `dubbed_fr.mp4` created; job status = "completed"
