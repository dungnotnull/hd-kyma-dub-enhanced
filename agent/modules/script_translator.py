"""Script translation: SeamlessM4T-v2 base translation + LLM idiomatic adaptation."""
from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch", "sv": "Swedish",
    "vi": "Vietnamese", "id": "Indonesian", "th": "Thai", "fa": "Persian",
}

VOICE_STYLES = {"formal", "casual", "dramatic", "documentary"}

IDIOMATIC_ADAPT_PROMPT = """\
You are a professional dubbing script writer for {target_language_name}.
Task: Rewrite the literal machine translation into natural spoken dialogue.

Rules:
1. Match emotional tone and register: {voice_style}
2. The spoken text MUST fit within {duration_budget_ms}ms at normal speaking pace
3. Preserve all proper nouns, technical terms, and named entities exactly
4. Output ONLY the adapted text — no explanations, no quotes

Original ({source_lang_name}): {original_text}
Literal translation: {literal_translation}
Adapted script:"""

TIMING_ADJUST_PROMPT = """\
The dubbed segment for a {duration_ms}ms slot is {direction} by {delta_ms}ms.
{action} the text while fully preserving semantic meaning.
Output ONLY the adjusted text — no preamble, no explanation.

Current text: {current_text}
Adjusted text:"""

QUALITY_REVIEW_PROMPT = """\
Evaluate this dubbed script segment for a {target_language_name} audience.
Score two dimensions 0.0–5.0:
- naturalness: Does it sound like natural spoken {target_language_name}? (5 = native speaker quality)
- accuracy: Does it preserve the full meaning of the original? (5 = perfect semantic fidelity)

Original ({source_lang_name}): {original_text}
Dubbed script ({target_lang_name}): {dubbed_text}

Respond with JSON only:
{{"naturalness": X.X, "accuracy": X.X, "issues": ["issue1", "issue2"]}}"""


@dataclass
class TranslatedSegment:
    id: int
    start: float
    end: float
    duration_ms: float
    original_text: str
    literal_translation: str
    adapted_text: str
    duration_fitted_text: str
    naturalness_score: float = 0.0
    accuracy_score: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    segments: list[TranslatedSegment]
    source_language: str
    target_language: str
    voice_style: str
    avg_naturalness: float = 0.0
    avg_accuracy: float = 0.0
    model_used: str = "seamless-m4t-v2-large"

    def full_script(self) -> str:
        return "\n".join(s.duration_fitted_text for s in self.segments)


class ScriptTranslator:
    """Two-stage translation: SeamlessM4T-v2 base + LLM idiomatic adaptation."""

    def __init__(self, llm_client=None, hf_manager=None):
        self._llm = llm_client
        self._hf = hf_manager
        self._seamless_model = None
        self._seamless_processor = None
        self._similarity_model = None

    def _load_seamless(self):
        if self._seamless_model is not None:
            return
        if self._hf:
            self._seamless_model = self._hf.load_seamless_m4t()
            return
        try:
            from transformers import AutoProcessor, SeamlessM4Tv2Model
            self._seamless_processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large"
            )
            self._seamless_model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large"
            )
            logger.info("Loaded facebook/seamless-m4t-v2-large")
        except Exception as e:
            logger.warning("SeamlessM4T-v2 load failed (%s); using fallback translator", e)
            self._seamless_model = None

    def _load_similarity(self):
        if self._similarity_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self._similarity_model = None

    async def translate(
        self,
        transcript,
        target_language: str,
        voice_style: str = "casual",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        if target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {target_language}. Supported: {list(SUPPORTED_LANGUAGES)}")
        if voice_style not in VOICE_STYLES:
            voice_style = "casual"

        self._load_seamless()

        source_lang = source_language or transcript.language
        target_lang_name = SUPPORTED_LANGUAGES[target_language]
        source_lang_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)

        translated_segments = []
        naturalness_scores = []
        accuracy_scores = []

        for seg in transcript.segments:
            duration_ms = (seg.end - seg.start) * 1000

            literal = await self._base_translate(seg.text, source_lang, target_language)

            adapted = await self._adapt_idiomatically(
                original_text=seg.text,
                literal_translation=literal,
                source_lang_name=source_lang_name,
                target_language_name=target_lang_name,
                voice_style=voice_style,
                duration_budget_ms=int(duration_ms),
            )

            fitted = await self._fit_duration(adapted, duration_ms, target_lang_name)

            nat_score, acc_score, issues = await self._review_quality(
                original_text=seg.text,
                dubbed_text=fitted,
                source_lang_name=source_lang_name,
                target_language_name=target_lang_name,
                target_lang_name=target_language,
            )
            naturalness_scores.append(nat_score)
            accuracy_scores.append(acc_score)

            translated_segments.append(TranslatedSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                duration_ms=duration_ms,
                original_text=seg.text,
                literal_translation=literal,
                adapted_text=adapted,
                duration_fitted_text=fitted,
                naturalness_score=nat_score,
                accuracy_score=acc_score,
                issues=issues,
            ))

        avg_nat = sum(naturalness_scores) / len(naturalness_scores) if naturalness_scores else 0.0
        avg_acc = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0

        return TranslationResult(
            segments=translated_segments,
            source_language=source_lang,
            target_language=target_language,
            voice_style=voice_style,
            avg_naturalness=avg_nat,
            avg_accuracy=avg_acc,
        )

    async def _base_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if self._seamless_model is None:
            return self._fallback_translate(text, source_lang, target_lang)
        try:
            from transformers import AutoProcessor
            if self._seamless_processor is None:
                self._seamless_processor = AutoProcessor.from_pretrained(
                    "facebook/seamless-m4t-v2-large"
                )
            inputs = self._seamless_processor(
                text=[text],
                src_lang=_to_seamless_lang(source_lang),
                return_tensors="pt",
            )
            output_tokens = self._seamless_model.generate(
                **inputs, tgt_lang=_to_seamless_lang(target_lang), generate_speech=False
            )
            translated = self._seamless_processor.decode(
                output_tokens[0].tolist()[0], skip_special_tokens=True
            )
            return translated
        except Exception as e:
            logger.warning("SeamlessM4T inference error (%s); using fallback", e)
            return self._fallback_translate(text, source_lang, target_lang)

    def _fallback_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        except Exception:
            return f"[{target_lang.upper()} translation of: {text}]"

    async def _adapt_idiomatically(
        self,
        original_text: str,
        literal_translation: str,
        source_lang_name: str,
        target_language_name: str,
        voice_style: str,
        duration_budget_ms: int,
    ) -> str:
        if not self._llm:
            return literal_translation
        prompt = IDIOMATIC_ADAPT_PROMPT.format(
            target_language_name=target_language_name,
            voice_style=voice_style,
            duration_budget_ms=duration_budget_ms,
            source_lang_name=source_lang_name,
            original_text=original_text,
            literal_translation=literal_translation,
        )
        try:
            adapted = await self._llm.complete(prompt, max_tokens=300, temperature=0.3)
            return adapted.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning("LLM idiomatic adaptation failed (%s)", e)
            return literal_translation

    async def _fit_duration(self, text: str, duration_ms: float, target_language_name: str) -> str:
        estimated_ms = _estimate_speech_duration_ms(text)
        delta_ms = abs(estimated_ms - duration_ms)
        if delta_ms <= duration_ms * 0.15:
            return text

        direction = "long" if estimated_ms > duration_ms else "short"
        action = "Shorten" if direction == "long" else "Expand"

        if not self._llm:
            return text

        prompt = TIMING_ADJUST_PROMPT.format(
            duration_ms=int(duration_ms),
            direction=direction,
            delta_ms=int(delta_ms),
            action=action,
            current_text=text,
        )
        try:
            adjusted = await self._llm.complete(prompt, max_tokens=200, temperature=0.2)
            return adjusted.strip().strip('"').strip("'") or text
        except Exception:
            return text

    async def _review_quality(
        self,
        original_text: str,
        dubbed_text: str,
        source_lang_name: str,
        target_language_name: str,
        target_lang_name: str,
    ) -> tuple[float, float, list[str]]:
        if not self._llm:
            return 4.0, 4.0, []
        prompt = QUALITY_REVIEW_PROMPT.format(
            target_language_name=target_language_name,
            source_lang_name=source_lang_name,
            original_text=original_text,
            target_lang_name=target_lang_name,
            dubbed_text=dubbed_text,
        )
        try:
            resp = await self._llm.complete(prompt, max_tokens=150, temperature=0.1)
            raw = _extract_json(resp)
            data = json.loads(raw)
            naturalness = float(data.get("naturalness", 4.0))
            accuracy = float(data.get("accuracy", 4.0))
            issues = data.get("issues", [])
            return naturalness, accuracy, issues
        except Exception as e:
            logger.warning("Quality review LLM call failed (%s)", e)
            return 4.0, 4.0, []

    def compute_back_translation_similarity(
        self, original: str, back_translation: str
    ) -> float:
        self._load_similarity()
        if self._similarity_model is None:
            return 0.8
        import numpy as np
        embeddings = self._similarity_model.encode([original, back_translation], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))


def _estimate_speech_duration_ms(text: str, wpm: int = 150) -> float:
    word_count = len(text.split())
    return (word_count / wpm) * 60 * 1000


def _to_seamless_lang(lang_code: str) -> str:
    mapping = {
        "en": "eng", "es": "spa", "fr": "fra", "de": "deu",
        "it": "ita", "pt": "por", "ru": "rus", "zh": "cmn",
        "ja": "jpn", "ko": "kor", "ar": "arb", "hi": "hin",
        "tr": "tur", "pl": "pol", "nl": "nld", "sv": "swe",
        "vi": "vie", "id": "ind", "th": "tha", "fa": "pes",
    }
    return mapping.get(lang_code, lang_code)


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text
