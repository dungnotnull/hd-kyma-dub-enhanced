# Upstream Baseline — kyma-dub

**Upstream Repository:** https://github.com/sonpiaz/kyma-dub
**Upstream Pinned Commit:** `latest stable at time of fork` (pin SHA after cloning)
**Fork Date:** 2026-06-09

## Upstream Capability Comparison

| Capability | Upstream kyma-dub | kyma-dub-enhanced | Improvement |
|-----------|------------------|-------------------|-------------|
| TTS Engine | Basic TTS (eSpeak/gTTS) | XTTS-v2 zero-shot voice cloning | MOS 3.2 → 4.1+ |
| Voice Cloning | Not supported | 6-second reference audio clone | New feature |
| Lip Synchronization | Not supported | Wav2Lip (sync_confidence ≥ 0.85) | New feature |
| Translation Quality | Literal machine translation | SeamlessM4T-v2 + LLM idiomatic adaptation | BLEU 25 → 38+ |
| Languages Supported | ~5–10 basic | 20 languages with idiomatic adaptation | 2–4× more |
| Quality Evaluation | None | UTMOS22 neural MOS + LLM review + retry loop | Automated QA |
| Research Self-Update | None | Daily ArXiv cs.SD/cs.CV/cs.CL crawl | Self-improving |
| API Interface | CLI only | CLI + REST API (FastAPI) | Production-ready |
| Docker Deployment | None | docker-compose with GPU profile | Containerized |

## Architecture Pattern: AI Sidecar

```
kyma-dub upstream code (Python)
         ↓
  [Unchanged — forked but unmodified]
         ↓
kyma-dub-enhanced AI layer (Python, new code in agent/ and tools/)
  - agent/modules/  — 5 AI modules replacing/augmenting upstream
  - tools/          — LLM client, HF model manager, knowledge updater
  - agent/          — orchestrator, main entry point, memory manager
```

The upstream kyma-dub code is preserved in its original form. All AI enhancements are additive — the AI layer either calls the upstream utilities as sub-processes or replaces them with SOTA models.

## Upstream Installation (for baseline benchmarking)

```bash
git clone https://github.com/sonpiaz/kyma-dub upstream/kyma-dub
cd upstream/kyma-dub
pip install -r requirements.txt
# Record baseline MOS score on test clip before any modifications
```

## Quantified Improvement Targets

| Metric | Upstream Baseline | Target | Status |
|--------|------------------|--------|--------|
| Output MOS (UTMOS22) | ~3.2 | ≥ 4.0 | Planned |
| Translation BLEU (FLORES-200) | ~25 | ≥ 38 | Planned |
| Lip-sync coverage | 0% | ≥ 80% face-frames | Planned |
| E2E latency (10-min, 1 lang, GPU) | N/A | ≤ 15 min | Planned |

## Key Improvement Delta

1. **XTTS-v2 voice cloning** (replaces eSpeak/gTTS) — zero-shot; no target language voice actor needed
2. **Wav2Lip lip-sync** — new; transforms audio dubbing into audiovisual dubbing
3. **SeamlessM4T-v2 + LLM** (replaces basic MT) — two-stage: machine translation + idiomatic rewrite
4. **UTMOS22 quality gate** — new; blocks low-quality output automatically
5. **Daily research crawler** — new; SECOND-KNOWLEDGE-BRAIN.md grows with latest TTS/lip-sync papers
