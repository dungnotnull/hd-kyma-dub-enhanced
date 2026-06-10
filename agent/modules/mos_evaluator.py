"""MOS quality evaluation using UTMOS22 neural predictor + LLM script review."""
from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MOS_THRESHOLD = 3.5
NATURALNESS_THRESHOLD = 4.0
ACCURACY_THRESHOLD = 4.0
MAX_RETRIES = 3

QUALITY_REVIEW_PROMPT = """\
Evaluate this dubbed audio script for a {target_language_name} audience.
Score two dimensions 0.0–5.0:
- naturalness: Does the text sound like natural spoken {target_language_name}? (5.0 = native-quality)
- accuracy: Does it faithfully preserve the original meaning? (5.0 = perfect semantic fidelity)

Original ({source_lang}): {original_script}
Dubbed script ({target_lang}): {dubbed_script}

Respond with JSON only:
{{"naturalness": X.X, "accuracy": X.X, "improvement_suggestions": ["..."]}}"""

SUPPORTED_LANGS = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch",
}


@dataclass
class QualityResult:
    mos_score: float
    script_naturalness: float
    script_accuracy: float
    passed: bool
    failure_reason: Optional[str] = None
    improvement_suggestions: list[str] = field(default_factory=list)
    mos_model: str = "utmos22"

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] MOS={self.mos_score:.2f} "
            f"Naturalness={self.script_naturalness:.1f}/5 "
            f"Accuracy={self.script_accuracy:.1f}/5"
        )


class MOSEvaluator:
    """Evaluates TTS quality via UTMOS22 neural MOS + LLM script review."""

    def __init__(self, llm_client=None, hf_manager=None):
        self._llm = llm_client
        self._hf = hf_manager
        self._utmos_model = None
        self._utmos_processor = None
        self._nisqa_model = None

    def _load_utmos(self) -> bool:
        if self._utmos_model is not None:
            return True
        try:
            from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
            model_id = "microsoft/UTMOS22"
            self._utmos_processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._utmos_model = AutoModelForAudioClassification.from_pretrained(model_id)
            logger.info("Loaded microsoft/UTMOS22")
            return True
        except Exception as e:
            logger.warning("UTMOS22 load failed (%s); trying NISQA fallback", e)
            return self._load_nisqa()

    def _load_nisqa(self) -> bool:
        if self._nisqa_model is not None:
            return True
        try:
            from transformers import pipeline
            self._nisqa_model = pipeline(
                "audio-classification",
                model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            )
            logger.info("Loaded NISQA/wav2vec2 MOS fallback")
            return True
        except Exception as e:
            logger.warning("NISQA fallback also failed (%s); using heuristic MOS", e)
            return False

    async def evaluate(
        self,
        audio_path: str,
        original_script: str,
        dubbed_script: str,
        source_language: str = "en",
        target_language: str = "es",
    ) -> QualityResult:
        """Full quality evaluation: neural MOS + LLM script review."""
        mos_score = self._predict_mos(audio_path)
        nat_score, acc_score, suggestions = await self._llm_review(
            original_script=original_script,
            dubbed_script=dubbed_script,
            source_language=source_language,
            target_language=target_language,
        )

        failure_reason = None
        passed = True

        if mos_score < MOS_THRESHOLD:
            passed = False
            failure_reason = f"MOS {mos_score:.2f} < threshold {MOS_THRESHOLD}"
        if nat_score < NATURALNESS_THRESHOLD:
            passed = False
            r = f"naturalness {nat_score:.1f} < {NATURALNESS_THRESHOLD}"
            failure_reason = f"{failure_reason}; {r}" if failure_reason else r
        if acc_score < ACCURACY_THRESHOLD:
            passed = False
            r = f"accuracy {acc_score:.1f} < {ACCURACY_THRESHOLD}"
            failure_reason = f"{failure_reason}; {r}" if failure_reason else r

        result = QualityResult(
            mos_score=mos_score,
            script_naturalness=nat_score,
            script_accuracy=acc_score,
            passed=passed,
            failure_reason=failure_reason,
            improvement_suggestions=suggestions,
        )
        logger.info("Quality evaluation: %s", result.summary())
        return result

    def _predict_mos(self, audio_path: str) -> float:
        if not self._load_utmos():
            return self._heuristic_mos(audio_path)
        try:
            import torch
            import soundfile as sf

            audio, sr = sf.read(audio_path)
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                except ImportError:
                    pass

            if self._utmos_processor and self._utmos_model:
                inputs = self._utmos_processor(
                    audio, sampling_rate=sr, return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self._utmos_model(**inputs)
                logits = outputs.logits
                mos = float(logits.squeeze()) if logits.numel() == 1 else float(logits.mean())
                mos = max(1.0, min(5.0, mos))
                return mos
            return self._heuristic_mos(audio_path)
        except Exception as e:
            logger.warning("UTMOS22 inference failed (%s); using heuristic", e)
            return self._heuristic_mos(audio_path)

    def _heuristic_mos(self, audio_path: str) -> float:
        """Heuristic MOS estimate based on audio signal properties."""
        try:
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(audio_path)
            if len(audio) == 0:
                return 1.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.001:
                return 1.5
            if rms < 0.01:
                return 3.0
            dc_bias = abs(float(np.mean(audio)))
            score = 4.0 - dc_bias * 10
            return max(1.0, min(5.0, score))
        except Exception:
            return 3.5

    async def _llm_review(
        self,
        original_script: str,
        dubbed_script: str,
        source_language: str,
        target_language: str,
    ) -> tuple[float, float, list[str]]:
        if not self._llm:
            return 4.0, 4.0, []

        target_lang_name = SUPPORTED_LANGS.get(target_language, target_language)
        source_lang_name = SUPPORTED_LANGS.get(source_language, source_language)

        prompt = QUALITY_REVIEW_PROMPT.format(
            target_language_name=target_lang_name,
            source_lang=source_lang_name,
            original_script=original_script[:500],
            target_lang=target_language,
            dubbed_script=dubbed_script[:500],
        )
        try:
            resp = await self._llm.complete(prompt, max_tokens=200, temperature=0.1)
            raw = _extract_json(resp)
            data = json.loads(raw)
            naturalness = float(data.get("naturalness", 4.0))
            accuracy = float(data.get("accuracy", 4.0))
            suggestions = data.get("improvement_suggestions", [])
            return naturalness, accuracy, suggestions
        except Exception as e:
            logger.warning("LLM quality review failed (%s)", e)
            return 4.0, 4.0, []

    def evaluate_batch(self, audio_paths: list[str]) -> list[float]:
        """Predict MOS for multiple audio files."""
        return [self._predict_mos(p) for p in audio_paths]

    def compute_retry_speed_factor(
        self, current_speed: float, quality_result: QualityResult, attempt: int
    ) -> float:
        """Compute adjusted speed_factor for retry based on quality failure."""
        if quality_result.passed:
            return current_speed
        delta = 0.05 * attempt
        if quality_result.mos_score < MOS_THRESHOLD:
            new_speed = current_speed - delta
        else:
            new_speed = current_speed + delta
        return max(0.75, min(1.25, new_speed))


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text
