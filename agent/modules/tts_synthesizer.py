"""TTS voice synthesis using XTTS-v2 zero-shot voice cloning with Bark fallback."""
from __future__ import annotations

import os
import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

XTTS_SUPPORTED_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
    "ar", "tr", "pl", "nl", "hi", "hu", "cs",
}

BARK_LANG_PREFIXES = {
    "en": "v2/en_speaker_6",
    "es": "v2/es_speaker_1",
    "fr": "v2/fr_speaker_1",
    "de": "v2/de_speaker_1",
    "it": "v2/it_speaker_1",
    "pt": "v2/pt_speaker_1",
    "zh": "v2/zh_speaker_1",
    "ja": "v2/ja_speaker_1",
    "ko": "v2/ko_speaker_0",
    "ru": "v2/ru_speaker_0",
    "pl": "v2/pl_speaker_0",
    "ar": "v2/ar_speaker_0",
    "tr": "v2/tr_speaker_0",
    "hi": "v2/hi_speaker_0",
}


@dataclass
class AudioSegment:
    segment_id: int
    start: float
    end: float
    duration_ms: float
    audio_path: str
    actual_duration_ms: float
    model_used: str


@dataclass
class SynthesisResult:
    segments: list[AudioSegment]
    full_audio_path: str
    target_language: str
    model_used: str
    total_duration_ms: float


class TTSSynthesizer:
    """XTTS-v2 zero-shot voice cloning with Bark fallback."""

    def __init__(self, hf_manager=None, models_dir: str = "./models"):
        self._hf = hf_manager
        self.models_dir = models_dir
        self._xtts_model = None
        self._bark_loaded = False

    def _load_xtts(self) -> bool:
        if self._xtts_model is not None:
            return True
        try:
            from TTS.api import TTS
            self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("Loaded coqui/XTTS-v2")
            return True
        except Exception as e:
            logger.warning("XTTS-v2 load failed (%s); will use Bark fallback", e)
            return False

    def _load_bark(self) -> bool:
        if self._bark_loaded:
            return True
        try:
            from bark import preload_models
            preload_models()
            self._bark_loaded = True
            logger.info("Loaded suno/bark")
            return True
        except Exception as e:
            logger.warning("Bark load failed (%s)", e)
            return False

    def synthesize(
        self,
        translation,
        reference_audio_path: str,
        target_language: str,
        output_dir: str,
        speed_factor: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize dubbed audio for all segments with voice cloning."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not reference_audio_path or not Path(reference_audio_path).exists():
            logger.warning("No valid reference audio; using default speaker")
            reference_audio_path = None

        use_xtts = (
            target_language in XTTS_SUPPORTED_LANGUAGES
            and self._load_xtts()
        )

        audio_segments = []
        for seg in translation.segments:
            seg_path = str(out_dir / f"segment_{seg.id:04d}.wav")
            target_ms = seg.duration_ms

            if use_xtts:
                actual_ms = self._synthesize_xtts(
                    text=seg.duration_fitted_text,
                    language=target_language,
                    reference_audio=reference_audio_path,
                    output_path=seg_path,
                    speed_factor=speed_factor,
                )
                model_used = "xtts-v2"
            else:
                actual_ms = self._synthesize_bark(
                    text=seg.duration_fitted_text,
                    language=target_language,
                    output_path=seg_path,
                )
                model_used = "bark"

            audio_segments.append(AudioSegment(
                segment_id=seg.id,
                start=seg.start,
                end=seg.end,
                duration_ms=target_ms,
                audio_path=seg_path,
                actual_duration_ms=actual_ms,
                model_used=model_used,
            ))

        full_audio_path = str(out_dir / "dubbed_full.wav")
        self._concatenate_segments(audio_segments, full_audio_path)

        total_duration = sum(s.actual_duration_ms for s in audio_segments)

        return SynthesisResult(
            segments=audio_segments,
            full_audio_path=full_audio_path,
            target_language=target_language,
            model_used="xtts-v2" if use_xtts else "bark",
            total_duration_ms=total_duration,
        )

    def _synthesize_xtts(
        self,
        text: str,
        language: str,
        reference_audio: Optional[str],
        output_path: str,
        speed_factor: float = 1.0,
    ) -> float:
        try:
            kwargs = {
                "text": text,
                "language": language,
                "file_path": output_path,
                "speed": speed_factor,
            }
            if reference_audio:
                kwargs["speaker_wav"] = reference_audio
            self._xtts_model.tts_to_file(**kwargs)
            return _wav_duration_ms(output_path)
        except Exception as e:
            logger.error("XTTS-v2 synthesis error for segment (%s); falling back to Bark", e)
            return self._synthesize_bark(text, language, output_path)

    def _synthesize_bark(self, text: str, language: str, output_path: str) -> float:
        if not self._load_bark():
            return self._synthesize_pyttsx3(text, output_path)
        try:
            from bark import generate_audio, SAMPLE_RATE
            import scipy.io.wavfile as wav

            voice_preset = BARK_LANG_PREFIXES.get(language, "v2/en_speaker_6")
            audio_array = generate_audio(text, history_prompt=voice_preset)
            wav.write(output_path, SAMPLE_RATE, audio_array)
            return _wav_duration_ms(output_path)
        except Exception as e:
            logger.error("Bark synthesis failed (%s); using pyttsx3 fallback", e)
            return self._synthesize_pyttsx3(text, output_path)

    def _synthesize_pyttsx3(self, text: str, output_path: str) -> float:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return _wav_duration_ms(output_path)
        except Exception as e:
            logger.error("pyttsx3 fallback failed (%s); generating silence", e)
            _write_silence(output_path, duration_ms=1000)
            return 1000.0

    def _concatenate_segments(self, segments: list[AudioSegment], output_path: str):
        """Concatenate audio segments with 10ms crossfade."""
        try:
            import soundfile as sf
            arrays = []
            sample_rate = None
            for seg in segments:
                if not Path(seg.audio_path).exists():
                    continue
                data, sr = sf.read(seg.audio_path)
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    try:
                        import librosa
                        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
                    except ImportError:
                        pass
                arrays.append(data)

            if not arrays:
                _write_silence(output_path, duration_ms=1000)
                return

            crossfade_samples = int((sample_rate or 22050) * 0.01)
            combined = arrays[0]
            for arr in arrays[1:]:
                if len(combined) >= crossfade_samples and len(arr) >= crossfade_samples:
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    combined[-crossfade_samples:] = (
                        combined[-crossfade_samples:] * fade_out + arr[:crossfade_samples] * fade_in
                    )
                    combined = np.concatenate([combined, arr[crossfade_samples:]])
                else:
                    combined = np.concatenate([combined, arr])

            sf.write(output_path, combined, sample_rate or 22050)
        except Exception as e:
            logger.error("Audio concatenation failed (%s); using ffmpeg fallback", e)
            self._concat_with_ffmpeg(segments, output_path)

    def _concat_with_ffmpeg(self, segments: list[AudioSegment], output_path: str):
        import subprocess, tempfile
        valid = [s.audio_path for s in segments if Path(s.audio_path).exists()]
        if not valid:
            _write_silence(output_path, duration_ms=1000)
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in valid:
                f.write(f"file '{p}'\n")
            list_path = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, output_path],
            check=True, capture_output=True,
        )
        os.unlink(list_path)

    def check_duration_fit(self, segment: AudioSegment, tolerance: float = 0.15) -> bool:
        """Check if synthesized audio fits within ±tolerance of target duration."""
        if segment.duration_ms == 0:
            return True
        ratio = abs(segment.actual_duration_ms - segment.duration_ms) / segment.duration_ms
        return ratio <= tolerance

    def compute_adjusted_speed(self, segment: AudioSegment) -> float:
        """Compute speed_factor to fit actual audio into target duration."""
        if segment.duration_ms == 0 or segment.actual_duration_ms == 0:
            return 1.0
        raw = segment.actual_duration_ms / segment.duration_ms
        return max(0.85, min(1.15, raw))


def _wav_duration_ms(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration * 1000.0
    except Exception:
        try:
            import wave
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return (frames / rate) * 1000.0
        except Exception:
            return 0.0


def _write_silence(path: str, duration_ms: int = 1000):
    try:
        import soundfile as sf
        samples = int(22050 * duration_ms / 1000)
        sf.write(path, np.zeros(samples), 22050)
    except Exception:
        Path(path).touch()
