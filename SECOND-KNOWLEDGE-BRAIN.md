# SECOND-KNOWLEDGE-BRAIN — kyma-dub-enhanced Neural Dubbing Platform

> Self-updating knowledge base. Updated daily via `tools/knowledge_updater.py`.
> Last manual seed: 2026-06-09

---

## Core Concepts & Frameworks

### Neural Text-to-Speech (TTS)
Modern neural TTS systems use encoder-decoder architectures (Tacotron2, FastSpeech2) or flow-based models (Glow-TTS) to map text to mel-spectrograms, then convert mel-spectrograms to waveforms using neural vocoders (HiFi-GAN, WaveGlow). Zero-shot voice cloning extends this by conditioning the decoder on a speaker embedding extracted from a reference audio clip — XTTS-v2 uses this paradigm with a 17-language multilingual speaker encoder.

### Automatic Speech Recognition (ASR)
Whisper (Radford et al., 2023) is a transformer encoder-decoder trained on 680K hours of weakly supervised multilingual audio. Whisper-large-v3 achieves SOTA WER on Common Voice across 99 languages. Word-level timestamps are extracted via dynamic time warping on cross-attention weights.

### Neural Machine Translation (NMT) for Speech Dubbing
Dubbing translation differs from document translation: translated text must (a) preserve semantic meaning, (b) fit within the original speaker's timing, and (c) sound natural when spoken aloud. SeamlessM4T-v2 (Barrault et al., 2023) addresses (a) and (b) via isometric translation constraints. LLM post-editing addresses (c).

### Lip Synchronization
Lip-sync generation requires: (1) extracting visual speech features from video frames, (2) conditioning on target audio, (3) synthesizing new lip region pixels. Wav2Lip (Prajwal et al., 2020) uses a sync discriminator trained on LRS2/LRS3 to ensure audio-visual correspondence. SadTalker adds head pose synthesis but requires portrait inputs.

### Mean Opinion Score (MOS) Prediction
Traditional MOS evaluation requires 20+ human listeners per sample. Neural MOS predictors (UTMOS22, NISQA, MOSNet) achieve Pearson r ≥ 0.90 with human ratings using self-supervised speech representations, enabling automated quality gating in production pipelines.

---

## Key Research Papers

| Title | Authors | Year | Venue | Link | Key Finding | Relevance |
|-------|---------|------|-------|------|-------------|-----------|
| Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford et al. | 2023 | ICML | https://arxiv.org/abs/2212.04356 | 680K hours → SOTA multilingual ASR; word timestamps via DTW | Source ASR backbone |
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Tacotron 2) | Shen et al. | 2018 | ICASSP | https://arxiv.org/abs/1712.05884 | Foundational mel-spectrogram + WaveNet architecture | TTS fundamentals |
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | Ren et al. | 2021 | ICLR | https://arxiv.org/abs/2006.04558 | Non-autoregressive TTS; duration/pitch/energy prediction | TTS speed reference |
| VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS | Kim et al. | 2021 | ICML | https://arxiv.org/abs/2106.06103 | End-to-end TTS; flow-based latent alignment; naturalness SOTA | XTTS-v2 architecture basis |
| Wav2Lip: Accurately Lip-syncing Videos In The Wild | Prajwal et al. | 2020 | ACM MM | https://arxiv.org/abs/2008.10010 | Expert sync discriminator; works in-the-wild; LRS3 SOTA | Lip-sync backbone |
| SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation | Zhang et al. | 2023 | CVPR | https://arxiv.org/abs/2211.12194 | 3D face motion from audio; portrait-only limitation | Lip-sync alternative |
| SeamlessM4T: Massively Multilingual & Multimodal Machine Translation | Barrault et al. | 2023 | Meta AI | https://arxiv.org/abs/2308.11596 | 100+ language pairs; speech-to-text + text-to-text; isometric translation | Base translation backbone |
| SeamlessM4T-v2: Seamless Multilingual & Multimodal Machine Translation | Seamless Communication Team | 2023 | Meta AI | https://arxiv.org/abs/2312.05187 | Improved SeamlessM4T; better low-resource languages | Primary translation model |
| UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022 | Saeki et al. | 2022 | Interspeech | https://arxiv.org/abs/2204.02152 | Neural MOS predictor; Pearson r=0.945 on VoiceMOS22 | Quality evaluation gate |
| YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion | Casanova et al. | 2022 | ICML | https://arxiv.org/abs/2112.02418 | Zero-shot TTS with 3s reference; foundation for XTTS | Voice cloning foundation |
| XTTS: Massively Multilingual Zero-Shot Text-to-Speech | Casanova et al. | 2024 | Interspeech | https://arxiv.org/abs/2406.04904 | XTTS-v2: 17 languages; MOS 4.17; 6s reference audio | Primary TTS model |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | Kong et al. | 2020 | NeurIPS | https://arxiv.org/abs/2010.05646 | Universal neural vocoder; 167× real-time on GPU | Vocoder for TTS pipeline |
| Bark: Text-Prompted Generative Audio Model | Suno AI | 2023 | GitHub | https://github.com/suno-ai/bark | Zero-shot TTS + non-speech sounds; CPU-friendly | TTS fallback |
| NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets | Mittag et al. | 2021 | Interspeech | https://arxiv.org/abs/2104.09494 | Multi-dimensional MOS (noisiness, coloration, discontinuity, loudness) | MOS evaluation fallback |
| Isometric Neural Machine Translation | Lakew et al. | 2022 | EMNLP | https://arxiv.org/abs/2205.02577 | Translation with length constraints for dubbing; reduces timing adjustment | Duration-fitting reference |

---

## State-of-the-Art Models

| Task | Model | Benchmark | Score | HuggingFace ID | Updated |
|------|-------|-----------|-------|----------------|---------|
| ASR (multilingual) | Whisper-large-v3 | Common Voice WER | 8.4% avg | `openai/whisper-large-v3` | 2023-11 |
| Zero-shot TTS (17 langs) | XTTS-v2 | UTMOS MOS | 4.17 | `coqui/XTTS-v2` | 2024-01 |
| Speech-to-Text Translation | SeamlessM4T-v2-large | FLORES-200 BLEU | 42.3 | `facebook/seamless-m4t-v2-large` | 2023-12 |
| Lip-sync generation | Wav2Lip | LRS3 LSE-C | 7.02 | `Rudrabha/Wav2Lip` | 2020-08 |
| Neural MOS prediction | UTMOS22 | VoiceMOS22 Pearson r | 0.945 | `microsoft/UTMOS22` | 2022-04 |
| TTS (English, single-speaker) | VITS | LJSpeech MOS | 4.43 | `kakao-enterprise/vits-ljs` | 2021-06 |
| Sentence similarity | all-MiniLM-L6-v2 | MTEB STS | 56.26 | `sentence-transformers/all-MiniLM-L6-v2` | 2021-08 |

---

## LLM Prompt Patterns

### 1. IDIOMATIC_ADAPT_PROMPT
```
System: You are a professional dubbing script writer for {target_language_name}.
Task: Rewrite the literal machine translation into natural spoken dialogue.

Rules:
1. Match emotional tone and register (formal/casual/dramatic/documentary)
2. The text MUST be speakable within {duration_budget_ms}ms at normal speaking pace
3. Preserve all proper nouns, technical terms, and named entities exactly
4. Output ONLY the adapted text — no explanations, no quotes

Original ({source_lang}): {original_text}
Literal translation: {literal_translation}
Adapted script:
```

### 2. QUALITY_REVIEW_PROMPT
```
Evaluate this dubbed script segment for a {target_language_name} audience.
Score two dimensions 0.0–5.0:
- naturalness: Does it sound like natural spoken {target_language_name}? (5 = native speaker quality)
- accuracy: Does it preserve the full meaning of the original? (5 = perfect semantic fidelity)

Original ({source_lang}): {original_text}
Dubbed script ({target_lang}): {dubbed_text}

Respond with JSON only:
{"naturalness": X.X, "accuracy": X.X, "issues": ["issue1", "issue2"]}
```

### 3. TIMING_ADJUST_PROMPT
```
The dubbed segment for a {duration_ms}ms slot is {long_or_short} by {delta_ms}ms.
{'Shorten' if delta_ms > 0 else 'Expand'} the text while fully preserving semantic meaning.
Output ONLY the adjusted text — no preamble, no explanation.

Current text: {current_text}
Adjusted text:
```

### 4. RESEARCH_SYNTHESIS_PROMPT
```
Below are recent research papers on neural dubbing, TTS, and lip-sync technology.
Synthesize the 3 most impactful findings for improving automated video dubbing quality.
Format: numbered list. Each item: finding, evidence (paper + metric), implementation suggestion.

Papers:
{papers_text}

Top 3 improvement recommendations:
```

---

## Authoritative Data Sources

| Source | URL | Use |
|--------|-----|-----|
| ArXiv cs.SD (Sound/Speech) | https://arxiv.org/list/cs.SD/recent | Daily TTS/ASR research |
| ArXiv cs.CV (Computer Vision) | https://arxiv.org/list/cs.CV/recent | Daily lip-sync/talking head research |
| ArXiv cs.CL (Computation and Language) | https://arxiv.org/list/cs.CL/recent | Daily NMT/dubbing research |
| Semantic Scholar API | https://api.semanticscholar.org/graph/v1 | Citation-filtered paper search |
| Papers with Code — TTS | https://paperswithcode.com/task/text-to-speech-synthesis | TTS leaderboard |
| Papers with Code — Talking Head | https://paperswithcode.com/task/talking-head-generation | Lip-sync leaderboard |
| INTERSPEECH proceedings | https://www.isca-speech.org/archive/ | Speech synthesis conferences |
| HuggingFace TTS models | https://huggingface.co/models?pipeline_tag=text-to-speech | Model discovery |
| Coqui TTS releases | https://github.com/coqui-ai/TTS/releases | XTTS-v2 updates |
| Wav2Lip repository | https://github.com/Rudrabha/Wav2Lip | Lip-sync updates |

---

## Self-Update Protocol

```yaml
schedule:
  frequency: daily
  time: "06:00 local"
  trigger: APScheduler CronTrigger (hour=6, minute=0)

sources:
  arxiv:
    categories: ["cs.SD", "cs.CV", "cs.CL"]
    max_results_per_category: 20
    keywords:
      - "text-to-speech"
      - "voice cloning"
      - "zero-shot TTS"
      - "neural dubbing"
      - "lip synchronization"
      - "talking head"
      - "speech synthesis"
      - "voice conversion"
      - "automatic dubbing"
      - "multilingual TTS"

  semantic_scholar:
    queries:
      - "neural TTS voice cloning 2024"
      - "lip sync video synthesis deep learning"
      - "automatic video dubbing translation"
      - "talking head generation"
      - "speech translation dubbing isometric"
    max_results_per_query: 10
    fields: ["title", "authors", "year", "venue", "externalIds", "abstract", "citationCount"]

  papers_with_code:
    leaderboards:
      - "text-to-speech-synthesis"
      - "talking-head-generation"
      - "speech-translation"

scoring:
  recency_weight: 0.6   # papers from last 90 days score highest
  relevance_weight: 0.4  # keyword match density
  min_score: 0.3         # discard below threshold
  top_n: 15              # append top-N entries per run

deduplication:
  method: SHA-256 hash of DOI/URL
  storage: memory_manager knowledge_hashes table

append_target: "## Knowledge Update Log" section in SECOND-KNOWLEDGE-BRAIN.md
```

---

## Knowledge Update Log

### 2026-06-09 — Initial Seed (15 entries)

| Date | Title | Venue | Key Finding | Link |
|------|-------|-------|-------------|------|
| 2026-06-09 | Robust Speech Recognition via Large-Scale Weak Supervision | ICML 2023 | Whisper-large-v3: SOTA multilingual ASR, word timestamps via DTW | https://arxiv.org/abs/2212.04356 |
| 2026-06-09 | XTTS: Massively Multilingual Zero-Shot TTS | Interspeech 2024 | XTTS-v2: 17 languages, MOS 4.17, 6s reference audio sufficient | https://arxiv.org/abs/2406.04904 |
| 2026-06-09 | Wav2Lip: Accurately Lip-syncing Videos In The Wild | ACM MM 2020 | Expert sync discriminator; works on arbitrary video; LRS3 SOTA | https://arxiv.org/abs/2008.10010 |
| 2026-06-09 | SeamlessM4T-v2 | Meta AI 2023 | 100+ language pairs; isometric mode for duration-constrained dubbing | https://arxiv.org/abs/2312.05187 |
| 2026-06-09 | UTMOS: UTokyo-SaruLab MOS Predictor | Interspeech 2022 | Neural MOS; r=0.945 with human ratings; enables automated QA | https://arxiv.org/abs/2204.02152 |
| 2026-06-09 | VITS: Conditional VAE with Adversarial Learning for TTS | ICML 2021 | End-to-end TTS; flow-based alignment; MOS 4.43 LJSpeech | https://arxiv.org/abs/2106.06103 |
| 2026-06-09 | FastSpeech 2: Fast and High-Quality End-to-End TTS | ICLR 2021 | Non-autoregressive TTS; duration/pitch/energy control | https://arxiv.org/abs/2006.04558 |
| 2026-06-09 | HiFi-GAN: Generative Adversarial Networks for Efficient TTS | NeurIPS 2020 | Universal neural vocoder; 167× real-time on GPU | https://arxiv.org/abs/2010.05646 |
| 2026-06-09 | SadTalker: 3D Motion Coefficients for Audio-Driven Talking Face | CVPR 2023 | Portrait-based lip sync with head pose; higher realism than Wav2Lip | https://arxiv.org/abs/2211.12194 |
| 2026-06-09 | YourTTS: Zero-Shot Multi-Speaker TTS | ICML 2022 | Zero-shot TTS with 3s reference; foundation for XTTS architecture | https://arxiv.org/abs/2112.02418 |
| 2026-06-09 | Tacotron 2: Natural TTS Synthesis | ICASSP 2018 | Mel-spectrogram + WaveNet; foundational TTS architecture | https://arxiv.org/abs/1712.05884 |
| 2026-06-09 | Isometric NMT for Dubbing | EMNLP 2022 | Translation with length constraints reduces timing adjustment passes | https://arxiv.org/abs/2205.02577 |
| 2026-06-09 | NISQA: Multi-Dimensional Speech Quality Prediction | Interspeech 2021 | CNN+self-attention MOS on crowdsourced data; fallback evaluator | https://arxiv.org/abs/2104.09494 |
| 2026-06-09 | Bark: Text-Prompted Generative Audio Model | Suno AI 2023 | Zero-shot TTS + non-speech; no GPU required; fallback TTS | https://github.com/suno-ai/bark |
| 2026-06-09 | SeamlessM4T: Massively Multilingual Machine Translation | Meta AI 2023 | First unified model for S2T, T2T, S2S across 100+ languages | https://arxiv.org/abs/2308.11596 |
