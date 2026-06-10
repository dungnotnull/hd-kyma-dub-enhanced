"""Automated tests for kyma-dub-enhanced."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"naturalness": 4.5, "accuracy": 4.5, "issues": []}')
    return client


@pytest.fixture
def mock_hf():
    hf = MagicMock()
    hf.encode.return_value = [[0.1, 0.2, 0.3]]
    return hf


@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal WAV file for testing."""
    import struct, math
    wav_path = str(tmp_path / "sample.wav")
    sample_rate = 22050
    duration_s = 1
    n_samples = sample_rate * duration_s
    data = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    with open(wav_path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_samples * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n_samples * 2))
        for s in data:
            f.write(struct.pack("<h", s))
    return wav_path


# ─────────────────────────────────────────────────────────────────────────────
# ASRTranscriber Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestASRTranscriber:
    def test_import(self):
        from agent.modules.asr_transcriber import ASRTranscriber
        t = ASRTranscriber()
        assert t.model_size == "large-v3"

    def test_transcript_result_to_srt(self):
        from agent.modules.asr_transcriber import TranscriptResult, Segment
        result = TranscriptResult(
            segments=[
                Segment(id=0, start=0.0, end=2.5, text="Hello world"),
                Segment(id=1, start=2.5, end=5.0, text="How are you"),
            ],
            language="en",
            duration_seconds=5.0,
            word_count=5,
        )
        srt = result.to_srt()
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "Hello world" in srt

    def test_transcript_result_to_dict(self):
        from agent.modules.asr_transcriber import TranscriptResult, Segment
        result = TranscriptResult(
            segments=[Segment(id=0, start=0.0, end=1.0, text="Test")],
            language="en",
            duration_seconds=1.0,
        )
        d = result.to_dict()
        assert d["language"] == "en"
        assert len(d["segments"]) == 1

    def test_save_transcript(self, tmp_dir):
        from agent.modules.asr_transcriber import ASRTranscriber, TranscriptResult, Segment
        t = ASRTranscriber()
        result = TranscriptResult(
            segments=[Segment(id=0, start=0.0, end=1.0, text="Hello")],
            language="en",
            duration_seconds=1.0,
        )
        paths = t.save_transcript(result, tmp_dir)
        assert Path(paths["json"]).exists()
        assert Path(paths["srt"]).exists()
        assert Path(paths["txt"]).exists()

    def test_validate_quality_empty_segments(self):
        from agent.modules.asr_transcriber import ASRTranscriber
        t = ASRTranscriber()
        with pytest.raises(ValueError, match="no segments"):
            t._validate_quality([])

    def test_seconds_to_srt_time(self):
        from agent.modules.asr_transcriber import _seconds_to_srt_time
        assert _seconds_to_srt_time(0.0) == "00:00:00,000"
        assert _seconds_to_srt_time(3661.5) == "01:01:01,500"


# ─────────────────────────────────────────────────────────────────────────────
# ScriptTranslator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptTranslator:
    def test_import(self):
        from agent.modules.script_translator import ScriptTranslator
        t = ScriptTranslator()
        assert t is not None

    def test_unsupported_language_raises(self, mock_llm):
        from agent.modules.script_translator import ScriptTranslator
        from agent.modules.asr_transcriber import TranscriptResult
        t = ScriptTranslator(llm_client=mock_llm)
        result = TranscriptResult(segments=[], language="en", duration_seconds=1.0)
        with pytest.raises(ValueError, match="Unsupported target language"):
            asyncio.run(t.translate(result, target_language="xyz"))

    def test_estimate_speech_duration(self):
        from agent.modules.script_translator import _estimate_speech_duration_ms
        ms = _estimate_speech_duration_ms("hello world how are you", wpm=150)
        assert ms > 0
        assert ms < 5000

    def test_to_seamless_lang(self):
        from agent.modules.script_translator import _to_seamless_lang
        assert _to_seamless_lang("en") == "eng"
        assert _to_seamless_lang("es") == "spa"
        assert _to_seamless_lang("zh") == "cmn"

    @pytest.mark.asyncio
    async def test_adapt_idiomatically_no_llm(self):
        from agent.modules.script_translator import ScriptTranslator
        t = ScriptTranslator(llm_client=None)
        result = await t._adapt_idiomatically(
            "Hello", "Hola", "English", "Spanish", "casual", 2000
        )
        assert result == "Hola"

    @pytest.mark.asyncio
    async def test_adapt_idiomatically_with_llm(self, mock_llm):
        from agent.modules.script_translator import ScriptTranslator
        mock_llm.complete = AsyncMock(return_value="¡Hola, ¿cómo estás?")
        t = ScriptTranslator(llm_client=mock_llm)
        result = await t._adapt_idiomatically(
            "Hello, how are you?", "Hola, ¿cómo estás?", "English", "Spanish", "casual", 3000
        )
        assert "Hola" in result or "estás" in result

    @pytest.mark.asyncio
    async def test_review_quality_no_llm(self):
        from agent.modules.script_translator import ScriptTranslator
        t = ScriptTranslator(llm_client=None)
        nat, acc, issues = await t._review_quality("Hello", "Hola", "English", "Spanish", "es")
        assert nat == 4.0
        assert acc == 4.0
        assert isinstance(issues, list)


# ─────────────────────────────────────────────────────────────────────────────
# TTSSynthesizer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSSynthesizer:
    def test_import(self):
        from agent.modules.tts_synthesizer import TTSSynthesizer
        s = TTSSynthesizer()
        assert s is not None

    def test_check_duration_fit_within_tolerance(self):
        from agent.modules.tts_synthesizer import TTSSynthesizer, AudioSegment
        s = TTSSynthesizer()
        seg = AudioSegment(
            segment_id=0, start=0.0, end=2.0, duration_ms=2000,
            audio_path="", actual_duration_ms=2100, model_used="xtts-v2",
        )
        assert s.check_duration_fit(seg, tolerance=0.15) is True

    def test_check_duration_fit_outside_tolerance(self):
        from agent.modules.tts_synthesizer import TTSSynthesizer, AudioSegment
        s = TTSSynthesizer()
        seg = AudioSegment(
            segment_id=0, start=0.0, end=2.0, duration_ms=2000,
            audio_path="", actual_duration_ms=3000, model_used="xtts-v2",
        )
        assert s.check_duration_fit(seg, tolerance=0.15) is False

    def test_compute_adjusted_speed(self):
        from agent.modules.tts_synthesizer import TTSSynthesizer, AudioSegment
        s = TTSSynthesizer()
        seg = AudioSegment(
            segment_id=0, start=0.0, end=2.0, duration_ms=2000,
            audio_path="", actual_duration_ms=2300, model_used="xtts-v2",
        )
        speed = s.compute_adjusted_speed(seg)
        assert 0.85 <= speed <= 1.15

    def test_write_silence(self, tmp_dir, sample_wav):
        from agent.modules.tts_synthesizer import _wav_duration_ms
        duration = _wav_duration_ms(sample_wav)
        assert duration > 0

    def test_pyttsx3_fallback(self, tmp_dir):
        from agent.modules.tts_synthesizer import TTSSynthesizer
        s = TTSSynthesizer()
        output_path = str(Path(tmp_dir) / "test.wav")
        with patch("pyttsx3.init") as mock_init:
            engine = MagicMock()
            mock_init.return_value = engine
            s._synthesize_pyttsx3("test text", output_path)
            engine.save_to_file.assert_called_once_with("test text", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# LipSyncEngine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLipSyncEngine:
    def test_import(self):
        from agent.modules.lipsync_engine import LipSyncEngine
        e = LipSyncEngine()
        assert e is not None

    def test_check_wav2lip_missing_checkpoint(self, tmp_dir):
        from agent.modules.lipsync_engine import LipSyncEngine
        e = LipSyncEngine(wav2lip_checkpoint="/nonexistent/path/model.pth")
        assert e._check_wav2lip() is False

    def test_skip_lipsync_merges_audio(self, tmp_dir, sample_wav):
        from agent.modules.lipsync_engine import LipSyncEngine
        e = LipSyncEngine()
        output_path = str(Path(tmp_dir) / "output.mp4")
        dummy_video = str(Path(tmp_dir) / "video.mp4")
        Path(dummy_video).touch()

        with patch.object(e, "_ffmpeg_merge_audio") as mock_merge:
            result = e._skip_lipsync(dummy_video, sample_wav, output_path, "test skip")
            mock_merge.assert_called_once()
            assert result.skipped is True
            assert result.skip_reason == "test skip"

    def test_parse_wav2lip_confidence_default(self):
        from agent.modules.lipsync_engine import _parse_wav2lip_confidence, WAV2LIP_FACE_DET_CONFIDENCE
        confidence = _parse_wav2lip_confidence("no confidence info here")
        assert confidence == WAV2LIP_FACE_DET_CONFIDENCE

    def test_parse_wav2lip_confidence_found(self):
        from agent.modules.lipsync_engine import _parse_wav2lip_confidence
        confidence = _parse_wav2lip_confidence("face confidence: 0.92 detected")
        assert confidence == 0.92


# ─────────────────────────────────────────────────────────────────────────────
# MOSEvaluator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMOSEvaluator:
    def test_import(self):
        from agent.modules.mos_evaluator import MOSEvaluator
        e = MOSEvaluator()
        assert e is not None

    def test_heuristic_mos_silent_audio(self, tmp_dir):
        from agent.modules.mos_evaluator import MOSEvaluator
        from agent.modules.tts_synthesizer import _write_silence
        silent_path = str(Path(tmp_dir) / "silent.wav")
        _write_silence(silent_path, duration_ms=500)
        e = MOSEvaluator()
        mos = e._heuristic_mos(silent_path)
        assert 1.0 <= mos <= 5.0

    def test_heuristic_mos_nonexistent_file(self):
        from agent.modules.mos_evaluator import MOSEvaluator
        e = MOSEvaluator()
        mos = e._heuristic_mos("/nonexistent/audio.wav")
        assert mos == 3.5

    @pytest.mark.asyncio
    async def test_llm_review_no_llm(self):
        from agent.modules.mos_evaluator import MOSEvaluator
        e = MOSEvaluator(llm_client=None)
        nat, acc, suggestions = await e._llm_review("Hello", "Hola", "en", "es")
        assert nat == 4.0
        assert acc == 4.0

    @pytest.mark.asyncio
    async def test_evaluate_pass(self, sample_wav, mock_llm):
        from agent.modules.mos_evaluator import MOSEvaluator
        mock_llm.complete = AsyncMock(
            return_value='{"naturalness": 4.5, "accuracy": 4.5, "improvement_suggestions": []}'
        )
        e = MOSEvaluator(llm_client=mock_llm)
        with patch.object(e, "_predict_mos", return_value=4.0):
            result = await e.evaluate(sample_wav, "Hello", "Hola", "en", "es")
            assert result.passed is True
            assert result.mos_score == 4.0

    @pytest.mark.asyncio
    async def test_evaluate_fail_low_mos(self, sample_wav, mock_llm):
        from agent.modules.mos_evaluator import MOSEvaluator
        mock_llm.complete = AsyncMock(
            return_value='{"naturalness": 4.5, "accuracy": 4.5, "improvement_suggestions": []}'
        )
        e = MOSEvaluator(llm_client=mock_llm)
        with patch.object(e, "_predict_mos", return_value=2.8):
            result = await e.evaluate(sample_wav, "Hello", "Hola", "en", "es")
            assert result.passed is False
            assert "MOS" in result.failure_reason

    def test_compute_retry_speed_factor(self):
        from agent.modules.mos_evaluator import MOSEvaluator, QualityResult
        e = MOSEvaluator()
        qr = QualityResult(mos_score=2.8, script_naturalness=4.0, script_accuracy=4.0, passed=False)
        new_speed = e.compute_retry_speed_factor(1.0, qr, attempt=1)
        assert 0.75 <= new_speed <= 1.25


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManager:
    def test_save_and_get_job(self, tmp_dir):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=str(Path(tmp_dir) / "test.db"))
        mem.save_job("job-001", "/video.mp4", "es", status="running")
        job = mem.get_job("job-001")
        assert job is not None
        assert job["status"] == "running"
        assert job["target_lang"] == "es"

    def test_update_job_status(self, tmp_dir):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=str(Path(tmp_dir) / "test.db"))
        mem.save_job("job-002", "/video.mp4", "fr")
        mem.update_job_status("job-002", "completed", mos_score=4.1)
        job = mem.get_job("job-002")
        assert job["status"] == "completed"

    def test_save_quality_result(self, tmp_dir):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=str(Path(tmp_dir) / "test.db"))
        mem.save_job("job-003", "/video.mp4", "de")
        mem.save_quality_result("job-003", "de", 4.0, 4.5, 4.3, True)
        history = mem.get_quality_history(target_lang="de")
        assert len(history) == 1
        assert history[0]["mos_score"] == 4.0

    def test_knowledge_hash_dedup(self, tmp_dir):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=str(Path(tmp_dir) / "test.db"))
        url = "https://arxiv.org/abs/2406.04904"
        assert not mem.is_known_paper(url)
        mem.mark_paper_known(url, "XTTS paper", "arxiv")
        assert mem.is_known_paper(url)

    def test_log_and_get_cost(self, tmp_dir):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=str(Path(tmp_dir) / "test.db"))
        mem.log_llm_cost("claude", "claude-opus-4-8", "translate", 500, 200, 0.023)
        summary = mem.get_cost_summary()
        assert "claude" in summary["by_provider"]

    def test_get_stats(self, tmp_dir):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=str(Path(tmp_dir) / "test.db"))
        stats = mem.get_stats()
        assert "total_jobs" in stats
        assert "known_papers" in stats


# ─────────────────────────────────────────────────────────────────────────────
# LLMClient Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMClient:
    def test_build_provider_chain_no_keys(self):
        from tools.llm_client import UnifiedLLMClient
        with patch.dict(os.environ, {}, clear=True):
            client = UnifiedLLMClient()
            assert "ollama" in client.provider_priority

    def test_build_provider_chain_with_anthropic(self):
        from tools.llm_client import UnifiedLLMClient
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = UnifiedLLMClient()
            assert "claude" in client.provider_priority

    def test_privacy_mode_forces_ollama(self):
        from tools.llm_client import UnifiedLLMClient
        with patch.dict(os.environ, {"PRIVACY_MODE": "true", "ANTHROPIC_API_KEY": "key"}):
            client = UnifiedLLMClient()
            assert client.provider_priority == ["ollama"]

    @pytest.mark.asyncio
    async def test_complete_all_fail(self):
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient(provider_priority=["claude"])
        with patch.object(client, "_call_with_retry", side_effect=RuntimeError("API error")):
            with pytest.raises(RuntimeError, match="All LLM providers failed"):
                await client.complete("test prompt")


# ─────────────────────────────────────────────────────────────────────────────
# HFModelManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHFModelManager:
    def test_singleton(self, tmp_dir):
        from tools.hf_model_manager import HFModelManager
        HFModelManager._instance = None
        m1 = HFModelManager(models_dir=tmp_dir)
        m2 = HFModelManager(models_dir=tmp_dir)
        assert m1 is m2

    def test_tfidf_fallback_encode(self, tmp_dir):
        from tools.hf_model_manager import HFModelManager
        HFModelManager._instance = None
        mgr = HFModelManager(models_dir=tmp_dir)
        vecs = mgr._tfidf_fallback_encode(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 3

    def test_get_model_info(self, tmp_dir):
        from tools.hf_model_manager import HFModelManager
        HFModelManager._instance = None
        mgr = HFModelManager(models_dir=tmp_dir)
        info = mgr.get_model_info()
        assert "whisper" in info
        assert "xtts" in info


# ─────────────────────────────────────────────────────────────────────────────
# VideoProcessor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoProcessor:
    def test_import(self):
        from agent.tools.video_processor import VideoProcessor
        vp = VideoProcessor()
        assert vp is not None

    def test_check_ffmpeg(self):
        from agent.tools.video_processor import VideoProcessor
        vp = VideoProcessor()
        result = vp.check_ffmpeg()
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_orchestrator_init(self, tmp_dir):
        from agent.orchestrator import KymaDubOrchestrator
        orch = KymaDubOrchestrator(config={}, output_base=tmp_dir)
        assert orch is not None

    def test_dubbing_job_dataclass(self):
        from agent.orchestrator import DubbingJob
        job = DubbingJob(
            job_id="test-01",
            video_path="/video.mp4",
            target_languages=["es", "fr"],
        )
        assert job.voice_style == "casual"
        assert len(job.target_languages) == 2

    def test_dubbing_result_summary_report(self):
        from agent.orchestrator import DubbingResult, LanguageDubbingResult
        result = DubbingResult(
            job_id="test-01",
            source_video="/video.mp4",
            language_results=[
                LanguageDubbingResult(
                    language="es",
                    output_video_path="./output/dubbed_es.mp4",
                    mos_score=4.1,
                    naturalness_score=4.3,
                    accuracy_score=4.5,
                    lip_sync_confidence=0.91,
                    quality_passed=True,
                )
            ],
            transcript_language="en",
            total_processing_time_s=120.0,
        )
        report = result.summary_report()
        assert "PASS" in report
        assert "es" in report.upper()

    def test_knowledge_updater_score_and_filter(self):
        from tools.knowledge_updater import KnowledgeUpdater, PaperEntry
        updater = KnowledgeUpdater()
        papers = [
            PaperEntry(
                title="Neural TTS voice cloning", authors="A", year=2024,
                venue="Interspeech", url="https://arxiv.org/abs/1",
                abstract="voice cloning text-to-speech", key_finding="x",
                relevance="TTS", recency_score=0.9, relevance_score=0.8,
            ),
            PaperEntry(
                title="Unrelated paper", authors="B", year=2020,
                venue="Nature", url="https://arxiv.org/abs/2",
                abstract="unrelated biology topic", key_finding="y",
                relevance="bio", recency_score=0.1, relevance_score=0.0,
            ),
        ]
        scored = updater._score_and_filter(papers)
        assert scored[0].title == "Neural TTS voice cloning"


# ─────────────────────────────────────────────────────────────────────────────
# CLI Smoke Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCLISmoke:
    def test_cli_imports(self):
        from agent.main import cli
        assert cli is not None

    def test_cli_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "dub" in result.output.lower()

    def test_cli_dub_missing_video(self, tmp_dir):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(Path(tmp_dir) / "empty.yaml"),
            "dub", "/nonexistent/video.mp4", "--languages", "es"
        ])
        assert result.exit_code != 0 or "not found" in (result.output or "").lower()

    def test_cli_cost_report(self, tmp_dir):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        db_path = str(Path(tmp_dir) / "test.db")
        config_path = str(Path(tmp_dir) / "config.yaml")
        Path(config_path).write_text(f"memory:\n  db_path: {db_path}\n")
        result = runner.invoke(cli, ["--config", config_path, "cost-report"])
        assert result.exit_code == 0
        assert "total_usd" in result.output

    def test_cli_languages_listed(self):
        from agent.modules.script_translator import SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) >= 20
