"""kyma-dub-enhanced — CLI and FastAPI server entry point."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kyma-dub")

CONFIG_DEFAULT = Path(__file__).parent.parent / "config" / "agent_config.yaml"


def _load_config(config_path: str) -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_orchestrator(config: dict):
    from agent.orchestrator import KymaDubOrchestrator
    return KymaDubOrchestrator(config=config, output_base=config.get("output_dir", "./output"))


@click.group()
@click.option("--config", default=str(CONFIG_DEFAULT), help="Path to agent_config.yaml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)


@cli.command()
@click.argument("video")
@click.option("--languages", "-l", required=True, multiple=True, help="Target language codes (es fr ja)")
@click.option("--reference-audio", "-r", default=None, help="Reference audio for voice cloning (6s WAV)")
@click.option("--voice-style", default="casual", type=click.Choice(["formal", "casual", "dramatic", "documentary"]))
@click.option("--output-dir", "-o", default="./output", help="Output directory")
@click.pass_context
def dub(ctx, video, languages, reference_audio, voice_style, output_dir):
    """Dub a video into one or more target languages."""
    from agent.orchestrator import DubbingJob
    config = ctx.obj["config"]
    config["output_dir"] = output_dir

    job = DubbingJob(
        job_id=str(uuid.uuid4())[:8],
        video_path=video,
        target_languages=list(languages),
        reference_audio_path=reference_audio,
        voice_style=voice_style,
        output_dir=output_dir,
    )

    orchestrator = _build_orchestrator(config)
    click.echo(f"Starting dubbing job {job.job_id}: {video} → {list(languages)}")

    result = asyncio.run(orchestrator.dub(job))
    click.echo(result.summary_report())


@cli.command()
@click.argument("audio")
@click.option("--language", "-l", default=None, help="Source language code (auto-detected if not specified)")
@click.option("--output", "-o", default="./output/transcript", help="Output directory for transcript files")
@click.pass_context
def transcribe(ctx, audio, language, output):
    """Transcribe audio/video to text with word timestamps."""
    from agent.modules.asr_transcriber import ASRTranscriber
    config = ctx.obj["config"]
    tr = ASRTranscriber(
        model_size=config.get("asr", {}).get("model_size", "large-v3")
    )
    click.echo(f"Transcribing {audio} ...")
    result = tr.transcribe(audio, language=language)
    paths = tr.save_transcript(result, output)
    click.echo(f"Language: {result.language}")
    click.echo(f"Segments: {len(result.segments)}")
    click.echo(f"Words: {result.word_count}")
    click.echo(f"Saved to: {paths}")


@cli.command()
@click.argument("job_id")
@click.pass_context
def status(ctx, job_id):
    """Check the status of a dubbing job."""
    config = ctx.obj["config"]
    from agent.memory.memory_manager import MemoryManager
    mem = MemoryManager(config.get("memory", {}).get("db_path", "./data/kyma_dub.db"))
    job = mem.get_job(job_id)
    if not job:
        click.echo(f"Job {job_id} not found")
        sys.exit(1)
    click.echo(json.dumps(job, indent=2))


@cli.command(name="update-knowledge")
@click.pass_context
def update_knowledge(ctx):
    """Run the research paper crawler to update SECOND-KNOWLEDGE-BRAIN.md."""
    from tools.knowledge_updater import KnowledgeUpdater
    from agent.memory.memory_manager import MemoryManager
    config = ctx.obj["config"]
    mem = MemoryManager(config.get("memory", {}).get("db_path", "./data/kyma_dub.db"))
    updater = KnowledgeUpdater(memory_manager=mem)
    click.echo("Running knowledge update ...")
    count = asyncio.run(updater.run_update())
    click.echo(f"Added {count} new entries to SECOND-KNOWLEDGE-BRAIN.md")


@cli.command(name="cost-report")
@click.option("--days", default=30)
@click.pass_context
def cost_report(ctx, days):
    """Show LLM API cost summary."""
    from agent.memory.memory_manager import MemoryManager
    config = ctx.obj["config"]
    mem = MemoryManager(config.get("memory", {}).get("db_path", "./data/kyma_dub.db"))
    summary = mem.get_cost_summary(days)
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=7821, type=int)
@click.option("--start-scheduler", is_flag=True, default=False)
@click.pass_context
def serve(ctx, host, port, start_scheduler):
    """Start the FastAPI server."""
    import uvicorn
    config = ctx.obj["config"]
    config["_start_scheduler"] = start_scheduler
    app = _build_fastapi_app(config)
    uvicorn.run(app, host=host, port=port)


def _build_fastapi_app(config: dict):
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import PlainTextResponse, FileResponse
    from pydantic import BaseModel

    app = FastAPI(title="kyma-dub-enhanced", version="1.0.0")
    orchestrator = _build_orchestrator(config)

    if config.get("_start_scheduler"):
        orchestrator.start_scheduler()

    class DubRequest(BaseModel):
        video_path: str
        target_languages: list[str]
        reference_audio_path: Optional[str] = None
        voice_style: str = "casual"
        output_dir: str = "./output"

    class DubResponse(BaseModel):
        job_id: str
        status: str
        message: str

    class TranscribeRequest(BaseModel):
        audio_path: str
        language: Optional[str] = None

    class StatusResponse(BaseModel):
        job_id: str
        status: str
        details: dict

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "kyma-dub-enhanced"}

    @app.post("/api/v1/dub", response_model=DubResponse)
    async def dub_video(req: DubRequest, background_tasks: BackgroundTasks):
        from agent.orchestrator import DubbingJob
        job_id = str(uuid.uuid4())[:8]
        job = DubbingJob(
            job_id=job_id,
            video_path=req.video_path,
            target_languages=req.target_languages,
            reference_audio_path=req.reference_audio_path,
            voice_style=req.voice_style,
            output_dir=req.output_dir,
        )
        background_tasks.add_task(_run_dub_job, orchestrator, job)
        return DubResponse(
            job_id=job_id,
            status="queued",
            message=f"Dubbing job {job_id} started for languages: {req.target_languages}",
        )

    @app.post("/api/v1/transcribe")
    async def transcribe_audio(req: TranscribeRequest):
        from agent.modules.asr_transcriber import ASRTranscriber
        tr = ASRTranscriber()
        result = tr.transcribe(req.audio_path, language=req.language)
        return result.to_dict()

    @app.get("/api/v1/job/{job_id}", response_model=StatusResponse)
    def get_job_status(job_id: str):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(config.get("memory", {}).get("db_path", "./data/kyma_dub.db"))
        job = mem.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return StatusResponse(job_id=job_id, status=job.get("status", "unknown"), details=job)

    @app.get("/api/v1/languages")
    def list_languages():
        from agent.modules.script_translator import SUPPORTED_LANGUAGES
        return {"supported_languages": SUPPORTED_LANGUAGES}

    @app.post("/api/v1/knowledge/update")
    async def update_knowledge_endpoint():
        from tools.knowledge_updater import KnowledgeUpdater
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(config.get("memory", {}).get("db_path", "./data/kyma_dub.db"))
        updater = KnowledgeUpdater(memory_manager=mem)
        count = await updater.run_update()
        return {"new_entries": count, "status": "ok"}

    @app.get("/api/v1/cost")
    def get_cost():
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(config.get("memory", {}).get("db_path", "./data/kyma_dub.db"))
        return mem.get_cost_summary()

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics():
        return orchestrator.get_prometheus_metrics()

    return app


async def _run_dub_job(orchestrator, job):
    try:
        await orchestrator.dub(job)
    except Exception as e:
        logger.error("Background dub job %s failed: %s", job.job_id, e)


if __name__ == "__main__":
    cli()
