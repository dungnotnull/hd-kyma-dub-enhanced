"""ASR transcription using Whisper-large-v3 with word-level timestamps."""
from __future__ import annotations

import os
import sys
import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language: str
    duration_seconds: float
    word_count: int = 0
    model_used: str = "whisper-large-v3"

    def to_dict(self) -> dict:
        return {
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [asdict(w) for w in s.words],
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in self.segments
            ],
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "word_count": self.word_count,
            "model_used": self.model_used,
        }

    def to_srt(self) -> str:
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = _seconds_to_srt_time(seg.start)
            end = _seconds_to_srt_time(seg.end)
            lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
        return "\n".join(lines)


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class ASRTranscriber:
    """Transcribes video/audio using Whisper-large-v3 with word-level timestamps."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._whisper_model = None
        self._faster_whisper_model = None
        self._use_faster = False

    def _load_model(self):
        if self._whisper_model is not None:
            return
        try:
            import whisper
            self._whisper_model = whisper.load_model(self.model_size)
            logger.info("Loaded openai/whisper-%s", self.model_size)
        except Exception as e:
            logger.warning("openai-whisper failed (%s); trying faster-whisper fallback", e)
            try:
                from faster_whisper import WhisperModel
                device = "cuda" if _cuda_available() else "cpu"
                compute = "float16" if device == "cuda" else "int8"
                self._faster_whisper_model = WhisperModel(
                    self.model_size, device=device, compute_type=compute
                )
                self._use_faster = True
                logger.info("Loaded faster-whisper/%s on %s", self.model_size, device)
            except Exception as e2:
                raise RuntimeError(
                    f"Both whisper and faster-whisper failed: {e} / {e2}"
                )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> TranscriptResult:
        """Transcribe audio file → TranscriptResult with word timestamps."""
        self._load_model()
        audio_path = str(audio_path)

        if self._use_faster:
            return self._transcribe_faster(audio_path, language, beam_size, word_timestamps)
        return self._transcribe_openai(audio_path, language, beam_size, word_timestamps)

    def _transcribe_openai(
        self, audio_path: str, language: Optional[str], beam_size: int, word_timestamps: bool
    ) -> TranscriptResult:
        import whisper
        options = {"beam_size": beam_size, "word_timestamps": word_timestamps}
        if language:
            options["language"] = language

        result = self._whisper_model.transcribe(audio_path, **options)

        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            words = []
            for w in seg.get("words", []):
                words.append(Word(
                    start=w["start"],
                    end=w["end"],
                    text=w["word"],
                    probability=w.get("probability", 1.0),
                ))
            segments.append(Segment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                words=words,
                no_speech_prob=seg.get("no_speech_prob", 0.0),
                avg_logprob=seg.get("avg_logprob", 0.0),
            ))

        duration = segments[-1].end if segments else 0.0
        word_count = sum(len(s.words) for s in segments) or sum(
            len(s.text.split()) for s in segments
        )
        detected_lang = result.get("language", language or "en")

        self._validate_quality(segments)

        return TranscriptResult(
            segments=segments,
            language=detected_lang,
            duration_seconds=duration,
            word_count=word_count,
            model_used=f"whisper-{self.model_size}",
        )

    def _transcribe_faster(
        self, audio_path: str, language: Optional[str], beam_size: int, word_timestamps: bool
    ) -> TranscriptResult:
        segments_iter, info = self._faster_whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )

        segments = []
        for i, seg in enumerate(segments_iter):
            words = []
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append(Word(
                        start=w.start,
                        end=w.end,
                        text=w.word,
                        probability=w.probability,
                    ))
            segments.append(Segment(
                id=i,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
                no_speech_prob=seg.no_speech_prob,
                avg_logprob=seg.avg_logprob,
            ))

        duration = segments[-1].end if segments else 0.0
        word_count = sum(len(s.words) for s in segments) or sum(
            len(s.text.split()) for s in segments
        )

        self._validate_quality(segments)

        return TranscriptResult(
            segments=segments,
            language=info.language,
            duration_seconds=duration,
            word_count=word_count,
            model_used=f"faster-whisper-{self.model_size}",
        )

    def _validate_quality(self, segments: list[Segment]):
        if not segments:
            raise ValueError("Transcription produced no segments — audio may be silent")
        avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
        if avg_no_speech > 0.8:
            logger.warning(
                "High average no_speech_prob=%.2f — audio quality may be poor", avg_no_speech
            )

    def save_transcript(self, result: TranscriptResult, output_dir: str) -> dict[str, str]:
        """Save transcript as JSON, SRT, and plain TXT."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        json_path = out / "transcript.json"
        json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        paths["json"] = str(json_path)

        srt_path = out / "transcript.srt"
        srt_path.write_text(result.to_srt(), encoding="utf-8")
        paths["srt"] = str(srt_path)

        txt_path = out / "transcript.txt"
        txt_path.write_text(
            "\n".join(s.text.strip() for s in result.segments), encoding="utf-8"
        )
        paths["txt"] = str(txt_path)

        return paths


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
