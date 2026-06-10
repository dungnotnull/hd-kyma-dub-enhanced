"""Persistent memory management for kyma-dub-enhanced (SQLite)."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS dubbing_jobs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    video_path  TEXT,
    target_lang TEXT,
    status      TEXT DEFAULT 'pending',
    mos_score   REAL,
    duration_s  REAL,
    output_path TEXT,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS quality_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    target_lang     TEXT,
    mos_score       REAL,
    naturalness     REAL,
    accuracy        REAL,
    passed          INTEGER,
    failure_reason  TEXT,
    model_used      TEXT,
    FOREIGN KEY (job_id) REFERENCES dubbing_jobs(id)
);

CREATE TABLE IF NOT EXISTS llm_cost_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    provider    TEXT,
    model       TEXT,
    task        TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL
);

CREATE TABLE IF NOT EXISTS knowledge_hashes (
    hash        TEXT PRIMARY KEY,
    title       TEXT,
    source      TEXT,
    added_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_lang ON dubbing_jobs(target_lang);
CREATE INDEX IF NOT EXISTS idx_quality_job ON quality_results(job_id);
CREATE INDEX IF NOT EXISTS idx_cost_provider ON llm_cost_log(provider);
"""


class MemoryManager:
    def __init__(self, db_path: str = "./data/kyma_dub.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save_job(
        self,
        job_id: str,
        video_path: str,
        target_lang: str,
        status: str = "pending",
        mos_score: Optional[float] = None,
        duration_s: Optional[float] = None,
        output_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO dubbing_jobs
                   (id, created_at, video_path, target_lang, status, mos_score, duration_s, output_path, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status, mos_score=excluded.mos_score,
                     duration_s=excluded.duration_s, output_path=excluded.output_path,
                     metadata=excluded.metadata""",
                (
                    job_id, _now(), video_path, target_lang, status,
                    mos_score, duration_s, output_path,
                    json.dumps(metadata or {}),
                ),
            )

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM dubbing_jobs WHERE id=?", (job_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_job_status(self, job_id: str, status: str, **kwargs):
        with self._lock, self._conn() as conn:
            fields = ["status=?"]
            values = [status]
            for k, v in kwargs.items():
                if k in ("mos_score", "output_path", "duration_s"):
                    fields.append(f"{k}=?")
                    values.append(v)
            values.append(job_id)
            conn.execute(
                f"UPDATE dubbing_jobs SET {', '.join(fields)} WHERE id=?",
                values,
            )

    def save_quality_result(
        self,
        job_id: str,
        target_lang: str,
        mos_score: float,
        naturalness: float,
        accuracy: float,
        passed: bool,
        failure_reason: Optional[str] = None,
        model_used: str = "utmos22",
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO quality_results
                   (job_id, created_at, target_lang, mos_score, naturalness, accuracy, passed, failure_reason, model_used)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    job_id, _now(), target_lang, mos_score,
                    naturalness, accuracy, int(passed),
                    failure_reason, model_used,
                ),
            )

    def get_quality_history(self, target_lang: Optional[str] = None, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            if target_lang:
                rows = conn.execute(
                    "SELECT * FROM quality_results WHERE target_lang=? ORDER BY created_at DESC LIMIT ?",
                    (target_lang, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM quality_results ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_avg_mos_by_language(self) -> dict[str, float]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT target_lang, AVG(mos_score) as avg_mos FROM quality_results "
                "WHERE mos_score IS NOT NULL GROUP BY target_lang"
            ).fetchall()
            return {r["target_lang"]: round(r["avg_mos"], 2) for r in rows}

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        task: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO llm_cost_log
                   (created_at, provider, model, task, input_tokens, output_tokens, cost_usd)
                   VALUES (?,?,?,?,?,?,?)""",
                (_now(), provider, model, task, input_tokens, output_tokens, cost_usd),
            )

    def get_cost_summary(self, days: int = 30) -> dict:
        cutoff = datetime.utcnow().isoformat()[:10]
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT provider, SUM(cost_usd) as total_cost,
                          SUM(input_tokens) as total_input,
                          SUM(output_tokens) as total_output,
                          COUNT(*) as call_count
                   FROM llm_cost_log
                   WHERE created_at >= ?
                   GROUP BY provider""",
                (cutoff[:7],),
            ).fetchall()
            total = sum(r["total_cost"] for r in rows)
            return {
                "total_usd": round(total, 4),
                "by_provider": {r["provider"]: dict(r) for r in rows},
            }

    def is_known_paper(self, url_or_doi: str) -> bool:
        h = hashlib.sha256(url_or_doi.encode()).hexdigest()
        with self._conn() as conn:
            return bool(
                conn.execute(
                    "SELECT 1 FROM knowledge_hashes WHERE hash=?", (h,)
                ).fetchone()
            )

    def mark_paper_known(self, url_or_doi: str, title: str = "", source: str = "") -> None:
        h = hashlib.sha256(url_or_doi.encode()).hexdigest()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_hashes (hash, title, source, added_at) VALUES (?,?,?,?)",
                (h, title, source, _now()),
            )

    def get_stats(self) -> dict:
        with self._conn() as conn:
            jobs = conn.execute("SELECT COUNT(*) as n FROM dubbing_jobs").fetchone()["n"]
            passed = conn.execute("SELECT COUNT(*) as n FROM quality_results WHERE passed=1").fetchone()["n"]
            failed = conn.execute("SELECT COUNT(*) as n FROM quality_results WHERE passed=0").fetchone()["n"]
            papers = conn.execute("SELECT COUNT(*) as n FROM knowledge_hashes").fetchone()["n"]
            return {
                "total_jobs": jobs,
                "quality_passed": passed,
                "quality_failed": failed,
                "known_papers": papers,
            }


def _now() -> str:
    return datetime.utcnow().isoformat()
