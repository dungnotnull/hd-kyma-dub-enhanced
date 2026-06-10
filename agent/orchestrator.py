"""Core dubbing orchestrator: drives ASR → translation → TTS → lip-sync → QA."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


@dataclass
class DubbingJob:
    job_id: str
    video_path: str
    target_languages: list[str]
    reference_audio_path: Optional[str] = None
    voice_style: str = "casual"
    output_dir: str = "./output"


@dataclass
class LanguageDubbingResult:
    language: str
    output_video_path: str
    mos_score: float
    naturalness_score: float
    accuracy_score: float
    lip_sync_confidence: float
    quality_passed: bool
    quality_warning: bool = False
    retries_used: int = 0
    processing_time_s: float = 0.0


@dataclass
class DubbingResult:
    job_id: str
    source_video: str
    language_results: list[LanguageDubbingResult]
    transcript_language: str
    total_processing_time_s: float
    cost_summary: dict = field(default_factory=dict)
    error: Optional[str] = None

    def summary_report(self) -> str:
        lines = [
            f"# Dubbing Report — Job {self.job_id}",
            f"Source: {self.source_video}",
            f"Source language: {self.transcript_language}",
            f"Total processing time: {self.total_processing_time_s:.1f}s",
            "",
            "## Language Results",
        ]
        for r in self.language_results:
            status = "PASS" if r.quality_passed else ("WARN" if r.quality_warning else "FAIL")
            lines.append(
                f"- **{r.language.upper()}** [{status}] "
                f"MOS={r.mos_score:.2f} Nat={r.naturalness_score:.1f} Acc={r.accuracy_score:.1f} "
                f"LipSync={r.lip_sync_confidence:.2f} → `{r.output_video_path}`"
            )
        if self.cost_summary:
            lines += ["", f"## Cost: ${self.cost_summary.get('total_usd', 0):.4f}"]
        return "\n".join(lines)


class KymaDubOrchestrator:
    """Drives the full dubbing pipeline for one or more target languages."""

    def __init__(self, config: dict, output_base: str = "./output"):
        self.config = config
        self.output_base = output_base
        self._video_processor = None
        self._transcriber = None
        self._translator = None
        self._tts = None
        self._lipsync = None
        self._mos_evaluator = None
        self._memory = None
        self._llm = None
        self._hf = None
        self._scheduler = None

    def _get_memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            self._memory = MemoryManager(
                db_path=self.config.get("memory", {}).get("db_path", "./data/kyma_dub.db")
            )
        return self._memory

    def _get_llm(self):
        if self._llm is None:
            from tools.llm_client import UnifiedLLMClient
            self._llm = UnifiedLLMClient(memory_manager=self._get_memory())
        return self._llm

    def _get_hf(self):
        if self._hf is None:
            from tools.hf_model_manager import HFModelManager
            self._hf = HFModelManager(
                models_dir=self.config.get("models_dir", "./models")
            )
        return self._hf

    def _get_video_processor(self):
        if self._video_processor is None:
            from agent.tools.video_processor import VideoProcessor
            self._video_processor = VideoProcessor()
        return self._video_processor

    def _get_transcriber(self):
        if self._transcriber is None:
            from agent.modules.asr_transcriber import ASRTranscriber
            self._transcriber = ASRTranscriber(
                model_size=self.config.get("asr", {}).get("model_size", "large-v3")
            )
        return self._transcriber

    def _get_translator(self):
        if self._translator is None:
            from agent.modules.script_translator import ScriptTranslator
            self._translator = ScriptTranslator(
                llm_client=self._get_llm(),
                hf_manager=self._get_hf(),
            )
        return self._translator

    def _get_tts(self):
        if self._tts is None:
            from agent.modules.tts_synthesizer import TTSSynthesizer
            self._tts = TTSSynthesizer(
                hf_manager=self._get_hf(),
                models_dir=self.config.get("models_dir", "./models"),
            )
        return self._tts

    def _get_lipsync(self):
        if self._lipsync is None:
            from agent.modules.lipsync_engine import LipSyncEngine
            self._lipsync = LipSyncEngine(
                wav2lip_checkpoint=self.config.get(
                    "lipsync", {}
                ).get("checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth")
            )
        return self._lipsync

    def _get_mos_evaluator(self):
        if self._mos_evaluator is None:
            from agent.modules.mos_evaluator import MOSEvaluator
            self._mos_evaluator = MOSEvaluator(
                llm_client=self._get_llm(),
                hf_manager=self._get_hf(),
            )
        return self._mos_evaluator

    async def dub(self, job: DubbingJob) -> DubbingResult:
        """Run full dubbing pipeline for all target languages."""
        t_start = _now()
        memory = self._get_memory()
        video_proc = self._get_video_processor()

        if not Path(job.video_path).exists():
            raise FileNotFoundError(f"Source video not found: {job.video_path}")

        job_dir = Path(self.output_base) / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        memory.save_job(
            job_id=job.job_id,
            video_path=job.video_path,
            target_lang=",".join(job.target_languages),
            status="running",
        )

        # Step 1: Extract audio
        logger.info("[%s] Extracting audio from video", job.job_id)
        audio_path = str(job_dir / "source_audio.wav")
        video_proc.extract_audio(job.video_path, audio_path, sample_rate=16000, channels=1)

        # Step 2: Get reference audio for voice cloning
        ref_audio = job.reference_audio_path
        if not ref_audio or not Path(ref_audio).exists():
            ref_audio = str(job_dir / "reference_audio.wav")
            logger.info("[%s] Extracting reference audio for voice cloning", job.job_id)
            try:
                video_proc.extract_reference_audio(
                    job.video_path, ref_audio, start_sec=0.0, duration_sec=10.0, sample_rate=22050
                )
            except Exception as e:
                logger.warning("Reference audio extraction failed (%s); proceeding without", e)
                ref_audio = None

        # Step 3: ASR transcription
        logger.info("[%s] Transcribing audio (Whisper-large-v3)", job.job_id)
        transcriber = self._get_transcriber()
        transcript = transcriber.transcribe(audio_path, word_timestamps=True)
        logger.info(
            "[%s] Transcribed %d segments, language=%s",
            job.job_id, len(transcript.segments), transcript.language
        )

        # Step 4: Process each target language
        lang_results = await asyncio.gather(
            *[
                self._dub_language(
                    job=job,
                    job_dir=job_dir,
                    transcript=transcript,
                    target_language=lang,
                    ref_audio=ref_audio,
                )
                for lang in job.target_languages
            ],
            return_exceptions=False,
        )

        total_time = (_now() - t_start).total_seconds() if hasattr(_now() - t_start, 'total_seconds') else 0
        import time
        total_time = time.monotonic()

        cost_summary = memory.get_cost_summary()
        memory.update_job_status(job.job_id, "completed")

        result = DubbingResult(
            job_id=job.job_id,
            source_video=job.video_path,
            language_results=[r for r in lang_results if r is not None],
            transcript_language=transcript.language,
            total_processing_time_s=total_time,
            cost_summary=cost_summary,
        )

        report_path = job_dir / "report.md"
        report_path.write_text(result.summary_report(), encoding="utf-8")
        logger.info("[%s] Dubbing complete. Report: %s", job.job_id, report_path)
        return result

    async def _dub_language(
        self,
        job: DubbingJob,
        job_dir: Path,
        transcript,
        target_language: str,
        ref_audio: Optional[str],
    ) -> Optional[LanguageDubbingResult]:
        import time
        t0 = time.monotonic()
        lang_dir = job_dir / target_language
        lang_dir.mkdir(parents=True, exist_ok=True)
        memory = self._get_memory()

        logger.info("[%s][%s] Starting translation", job.job_id, target_language)
        translator = self._get_translator()
        translation = await translator.translate(
            transcript, target_language=target_language, voice_style=job.voice_style
        )

        speed_factor = 1.0
        lip_sync_result = None
        quality_result = None
        synthesis_result = None

        for attempt in range(1, MAX_RETRIES + 1):
            # TTS synthesis
            logger.info(
                "[%s][%s] TTS synthesis attempt %d (speed_factor=%.2f)",
                job.job_id, target_language, attempt, speed_factor,
            )
            tts_dir = str(lang_dir / f"tts_attempt_{attempt}")
            synthesis_result = self._get_tts().synthesize(
                translation=translation,
                reference_audio_path=ref_audio,
                target_language=target_language,
                output_dir=tts_dir,
                speed_factor=speed_factor,
            )

            # Lip-sync
            lipsync_video = str(lang_dir / f"lipsync_attempt_{attempt}.mp4")
            lip_sync_result = self._get_lipsync().generate(
                video_path=job.video_path,
                dubbed_audio_path=synthesis_result.full_audio_path,
                output_path=lipsync_video,
            )

            # MOS quality evaluation
            logger.info("[%s][%s] MOS evaluation attempt %d", job.job_id, target_language, attempt)
            quality_result = await self._get_mos_evaluator().evaluate(
                audio_path=synthesis_result.full_audio_path,
                original_script=" ".join(s.text for s in transcript.segments),
                dubbed_script=translation.full_script(),
                source_language=transcript.language,
                target_language=target_language,
            )

            memory.save_quality_result(
                job_id=job.job_id,
                target_lang=target_language,
                mos_score=quality_result.mos_score,
                naturalness=quality_result.script_naturalness,
                accuracy=quality_result.script_accuracy,
                passed=quality_result.passed,
                failure_reason=quality_result.failure_reason,
            )

            if quality_result.passed:
                logger.info("[%s][%s] Quality gate PASSED (attempt %d)", job.job_id, target_language, attempt)
                break

            if attempt < MAX_RETRIES:
                logger.warning(
                    "[%s][%s] Quality gate FAILED: %s — retrying",
                    job.job_id, target_language, quality_result.failure_reason,
                )
                mos_eval = self._get_mos_evaluator()
                speed_factor = mos_eval.compute_retry_speed_factor(
                    speed_factor, quality_result, attempt
                )
            else:
                logger.error(
                    "[%s][%s] Quality gate failed after %d attempts — delivering with warning",
                    job.job_id, target_language, MAX_RETRIES,
                )

        # Assemble final output
        final_output = str(lang_dir / f"dubbed_{target_language}.mp4")
        if synthesis_result and lip_sync_result:
            if not lip_sync_result.skipped and lip_sync_result.output_video_path != final_output:
                import shutil
                shutil.copy2(lip_sync_result.output_video_path, final_output)
            elif lip_sync_result.skipped:
                self._get_video_processor().merge_audio_video(
                    job.video_path,
                    synthesis_result.full_audio_path,
                    final_output,
                )
        else:
            import shutil
            shutil.copy2(job.video_path, final_output)

        elapsed = time.monotonic() - t0
        memory.save_job(
            job_id=f"{job.job_id}_{target_language}",
            video_path=job.video_path,
            target_lang=target_language,
            status="completed",
            mos_score=quality_result.mos_score if quality_result else 0.0,
            output_path=final_output,
        )

        return LanguageDubbingResult(
            language=target_language,
            output_video_path=final_output,
            mos_score=quality_result.mos_score if quality_result else 0.0,
            naturalness_score=quality_result.script_naturalness if quality_result else 0.0,
            accuracy_score=quality_result.script_accuracy if quality_result else 0.0,
            lip_sync_confidence=lip_sync_result.sync_confidence if lip_sync_result else 0.0,
            quality_passed=quality_result.passed if quality_result else False,
            quality_warning=not quality_result.passed if quality_result else True,
            retries_used=0,
            processing_time_s=elapsed,
        )

    def start_scheduler(self):
        """Start APScheduler for daily knowledge update."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                self._run_knowledge_update,
                trigger=CronTrigger(hour=6, minute=0),
                id="daily_knowledge_update",
                replace_existing=True,
            )
            self._scheduler.start()
            logger.info("Scheduler started: daily knowledge update at 06:00")
        except ImportError:
            logger.warning("apscheduler not installed; scheduled updates disabled")

    def _run_knowledge_update(self):
        try:
            from tools.knowledge_updater import KnowledgeUpdater
            updater = KnowledgeUpdater(memory_manager=self._get_memory())
            asyncio.run(updater.run_update())
        except Exception as e:
            logger.error("Scheduled knowledge update failed: %s", e)

    def get_prometheus_metrics(self) -> str:
        memory = self._get_memory()
        stats = memory.get_stats()
        cost = memory.get_cost_summary()
        lines = [
            f'kymadub_jobs_total {stats["total_jobs"]}',
            f'kymadub_quality_passed_total {stats["quality_passed"]}',
            f'kymadub_quality_failed_total {stats["quality_failed"]}',
            f'kymadub_knowledge_papers_total {stats["known_papers"]}',
            f'kymadub_llm_cost_usd_30d {cost.get("total_usd", 0):.6f}',
        ]
        return "\n".join(lines) + "\n"


def _now():
    return datetime.utcnow()
