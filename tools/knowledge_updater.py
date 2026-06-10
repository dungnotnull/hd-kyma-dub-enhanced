"""Research paper crawler: ArXiv + Semantic Scholar → SECOND-KNOWLEDGE-BRAIN.md."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

ARXIV_CATEGORIES = ["cs.SD", "cs.CV", "cs.CL"]

ARXIV_KEYWORDS = [
    "text-to-speech", "voice cloning", "zero-shot TTS", "neural dubbing",
    "lip synchronization", "talking head", "speech synthesis", "voice conversion",
    "automatic dubbing", "multilingual TTS", "neural vocoder",
]

SEMANTIC_SCHOLAR_QUERIES = [
    "neural TTS voice cloning 2024",
    "lip sync video synthesis deep learning",
    "automatic video dubbing neural",
    "talking head generation audio driven",
    "multilingual speech synthesis zero-shot",
]

BRAIN_PATH = Path(__file__).parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"
BRAIN_LOG_HEADER = "## Knowledge Update Log"


@dataclass
class PaperEntry:
    title: str
    authors: str
    year: int
    venue: str
    url: str
    abstract: str
    key_finding: str
    relevance: str
    recency_score: float = 0.0
    relevance_score: float = 0.0

    def combined_score(self) -> float:
        return 0.6 * self.recency_score + 0.4 * self.relevance_score

    def to_table_row(self, date_str: str) -> str:
        return (
            f"| {date_str} | {self.title} | {self.venue} {self.year} | "
            f"{self.key_finding} | {self.url} |"
        )


class KnowledgeUpdater:
    """Daily ArXiv + Semantic Scholar crawler for kyma-dub-enhanced."""

    def __init__(self, memory_manager=None, brain_path: Optional[str] = None):
        self._memory = memory_manager
        self.brain_path = Path(brain_path) if brain_path else BRAIN_PATH

    async def run_update(self) -> int:
        """Run full crawl pipeline; return count of new entries added."""
        logger.info("Starting knowledge update crawl")
        all_papers: list[PaperEntry] = []

        async with aiohttp.ClientSession() as session:
            for cat in ARXIV_CATEGORIES:
                papers = await self._crawl_arxiv(session, cat)
                all_papers.extend(papers)

            for query in SEMANTIC_SCHOLAR_QUERIES:
                papers = await self._crawl_semantic_scholar(session, query)
                all_papers.extend(papers)

        scored = self._score_and_filter(all_papers)
        new_papers = [p for p in scored if not self._is_known(p.url)]
        new_papers = new_papers[:15]

        if new_papers:
            self._append_to_brain(new_papers)
            for p in new_papers:
                self._mark_known(p.url, p.title)
            logger.info("Added %d new entries to SECOND-KNOWLEDGE-BRAIN.md", len(new_papers))
        else:
            logger.info("No new papers found in this crawl")

        return len(new_papers)

    async def _crawl_arxiv(self, session: aiohttp.ClientSession, category: str) -> list[PaperEntry]:
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query=cat:{category}"
            f"&sortBy=submittedDate&sortOrder=descending&max_results=20"
        )
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()
            return self._parse_arxiv_xml(text, category)
        except Exception as e:
            logger.warning("ArXiv crawl failed for %s: %s", category, e)
            return []

    def _parse_arxiv_xml(self, xml_text: str, category: str) -> list[PaperEntry]:
        papers = []
        try:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title = _text(entry, "atom:title", ns).replace("\n", " ").strip()
                abstract = _text(entry, "atom:summary", ns).replace("\n", " ").strip()
                url = _text(entry, "atom:id", ns).strip()
                published = _text(entry, "atom:published", ns)
                year = int(published[:4]) if published else 2024
                authors_els = entry.findall("atom:author/atom:name", ns)
                authors = ", ".join(el.text for el in authors_els[:3] if el.text)
                if len(authors_els) > 3:
                    authors += " et al."

                relevance = self._compute_relevance_score(title + " " + abstract)
                if relevance < 0.1:
                    continue

                recency = self._compute_recency_score(published)
                key_finding = abstract[:120].rstrip(".") + "."

                papers.append(PaperEntry(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"ArXiv {category}",
                    url=url,
                    abstract=abstract[:300],
                    key_finding=key_finding,
                    relevance=f"Neural dubbing / {category}",
                    recency_score=recency,
                    relevance_score=relevance,
                ))
        except Exception as e:
            logger.warning("ArXiv XML parse error: %s", e)
        return papers

    async def _crawl_semantic_scholar(
        self, session: aiohttp.ClientSession, query: str
    ) -> list[PaperEntry]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "fields": "title,authors,year,venue,externalIds,abstract,citationCount",
            "limit": 10,
        }
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            return self._parse_s2_response(data)
        except Exception as e:
            logger.warning("Semantic Scholar crawl failed for '%s': %s", query, e)
            await asyncio.sleep(3)
            return []

    def _parse_s2_response(self, data: dict) -> list[PaperEntry]:
        papers = []
        for item in data.get("data", []):
            title = item.get("title", "")
            abstract = item.get("abstract") or ""
            year = item.get("year") or 2024
            venue = item.get("venue") or "Semantic Scholar"
            authors_list = item.get("authors", [])
            authors = ", ".join(a.get("name", "") for a in authors_list[:3])
            if len(authors_list) > 3:
                authors += " et al."

            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI") or ext_ids.get("ArXiv")
            url = f"https://doi.org/{doi}" if doi and "." in str(doi) else (
                f"https://arxiv.org/abs/{doi}" if doi else ""
            )
            if not url:
                continue

            relevance = self._compute_relevance_score(title + " " + abstract)
            if relevance < 0.1:
                continue

            recency = self._compute_recency_score(f"{year}-01-01")
            key_finding = abstract[:120].rstrip(".") + "." if abstract else title

            papers.append(PaperEntry(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                abstract=abstract[:300],
                key_finding=key_finding,
                relevance="Neural dubbing / speech synthesis",
                recency_score=recency,
                relevance_score=relevance,
            ))
        return papers

    def _compute_relevance_score(self, text: str) -> float:
        text_lower = text.lower()
        matches = sum(1 for kw in ARXIV_KEYWORDS if kw.lower() in text_lower)
        return min(1.0, matches / max(1, len(ARXIV_KEYWORDS) * 0.3))

    def _compute_recency_score(self, date_str: str) -> float:
        try:
            pub_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            days_old = (datetime.utcnow() - pub_date).days
            if days_old <= 30:
                return 1.0
            if days_old <= 90:
                return 0.8
            if days_old <= 365:
                return 0.5
            return 0.2
        except Exception:
            return 0.4

    def _score_and_filter(self, papers: list[PaperEntry]) -> list[PaperEntry]:
        scored = sorted(papers, key=lambda p: p.combined_score(), reverse=True)
        seen_urls: set[str] = set()
        deduped = []
        for p in scored:
            if p.url and p.url not in seen_urls and p.combined_score() >= 0.15:
                seen_urls.add(p.url)
                deduped.append(p)
        return deduped

    def _is_known(self, url: str) -> bool:
        if not self._memory:
            return False
        return self._memory.is_known_paper(url)

    def _mark_known(self, url: str, title: str):
        if self._memory:
            self._memory.mark_paper_known(url, title, "knowledge_updater")

    def _append_to_brain(self, papers: list[PaperEntry]):
        if not self.brain_path.exists():
            logger.warning("SECOND-KNOWLEDGE-BRAIN.md not found at %s", self.brain_path)
            return

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        new_rows = "\n".join(p.to_table_row(date_str) for p in papers)
        new_section = (
            f"\n### {date_str} — Automated Crawl ({len(papers)} new entries)\n\n"
            "| Date | Title | Venue | Key Finding | Link |\n"
            "|------|-------|-------|-------------|------|\n"
            f"{new_rows}\n"
        )

        content = self.brain_path.read_text(encoding="utf-8")
        if BRAIN_LOG_HEADER in content:
            idx = content.index(BRAIN_LOG_HEADER) + len(BRAIN_LOG_HEADER)
            content = content[:idx] + "\n" + new_section + content[idx:]
        else:
            content += f"\n\n{BRAIN_LOG_HEADER}\n{new_section}"

        self.brain_path.write_text(content, encoding="utf-8")

    def start_scheduled(self):
        """Start APScheduler daily 06:00 cron job."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            scheduler = BackgroundScheduler()
            scheduler.add_job(
                lambda: asyncio.run(self.run_update()),
                trigger=CronTrigger(hour=6, minute=0),
                id="daily_knowledge_update",
                replace_existing=True,
            )
            scheduler.start()
            logger.info("Knowledge updater scheduled: daily at 06:00")
            return scheduler
        except ImportError:
            logger.warning("apscheduler not installed; no scheduled updates")
            return None


def _text(element, tag: str, ns: dict) -> str:
    el = element.find(tag, ns)
    return el.text.strip() if el is not None and el.text else ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(KnowledgeUpdater().run_update())
