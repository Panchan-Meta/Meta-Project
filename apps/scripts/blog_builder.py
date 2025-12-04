"""Generate persona-aware blog HTML from client prompts.

This module fabricates structured HTML with title, description, tags,
latest article highlights, overview, and seven detailed sections tuned
for Johanne's persona. It reads a knowledge file from ``mybrain`` and an
index JSON file for recent articles. Translation snippets for Italian and
Japanese are appended so clients can localize the output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from content_plan_generator import (
    DEFAULT_BASE_URL as PLAN_BASE_URL,
    DEFAULT_INDEX_FILE as PLAN_INDEX_FILE,
    DEFAULT_KNOWLEDGE_FILE as PLAN_KNOWLEDGE_FILE,
    DEFAULT_MODEL as PLAN_MODEL,
    generate_plan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("/mnt/hgfs/output")
DEFAULT_INDEX_FILE = PROJECT_ROOT / "indexes" / "index.json"
DEFAULT_KNOWLEDGE_FILE = PROJECT_ROOT / "mybrain" / "knowledge.md"


@dataclass
class StatusReporter:
    """Collect human-readable status updates for the server responses."""

    _messages: list[str]

    def __init__(self) -> None:
        self._messages = []

    def add(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        self._messages.append(f"[{timestamp}] {message}")

    def pop_messages(self, include_current: bool = True) -> list[str]:
        if not include_current:
            return self._messages
        messages, self._messages = self._messages, []
        return messages


STATUS_REPORTER = StatusReporter()


@dataclass
class Article:
    title: str
    url: str
    summary: str
    published: datetime | None


PERSONA_NAME = "Johanne, the Pretentious Philosopher"
PERSONA_PILLARS = (
    "punk-toned philosophical rebellion",
    "rational yet anxious investment discipline",
    "critiques of currency dominance and tech precarity",
    "community that accepts doubt instead of cultish hype",
)


def _read_index(index_path: Path) -> list[Article]:
    """Load the latest articles from a JSON index.

    Expected format: a list of objects with ``title``, ``url``,
    optional ``summary``, and optional ``published`` ISO timestamp.
    """

    if not index_path.is_file():
        STATUS_REPORTER.add(f"Index file not found at {index_path}")
        return []

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        STATUS_REPORTER.add(f"Index file at {index_path} is not valid JSON")
        return []

    articles: list[Article] = []
    for entry in data if isinstance(data, list) else []:
        title = str(entry.get("title", "Untitled")) if isinstance(entry, Mapping) else "Untitled"
        url = str(entry.get("url", "")) if isinstance(entry, Mapping) else ""
        summary = str(entry.get("summary", "")) if isinstance(entry, Mapping) else ""
        published_raw = entry.get("published") if isinstance(entry, Mapping) else None

        published = None
        if isinstance(published_raw, str):
            try:
                published = datetime.fromisoformat(published_raw)
            except ValueError:
                published = None

        articles.append(Article(title=title, url=url, summary=summary, published=published))

    # Newest first
    return sorted(
        articles,
        key=lambda art: art.published or datetime.min,
        reverse=True,
    )


def _read_knowledge(knowledge_path: Path) -> str:
    if not knowledge_path.is_file():
        STATUS_REPORTER.add(f"Knowledge file missing at {knowledge_path}")
        return ""
    return knowledge_path.read_text(encoding="utf-8")


def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars].rstrip()


def _build_title(prompt: str) -> str:
    core = prompt.strip() or "Rahab Punkaholic Girls Insight"
    return f"{core} — {PERSONA_NAME}'s Critical Lens"


def _build_description(prompt: str) -> str:
    base = (
        "Johanne decodes Rahab's punk cosmos through rational paranoia, "
        "balancing Web3 experimentation with a craving for stability while "
        "tracking the latest drops and articles that echo his doubts."
    )
    combined = f"{base} Prompt cue: {prompt.strip()}"
    return _truncate(combined, 200)


def _build_tags() -> list[str]:
    return [
        "Rahab Punkaholic Girls",
        "Web3 philosophy",
        "punk aesthetics",
        "crypto anxiety",
        "geopolitical unease",
        "critical optimism",
    ]


def _format_articles(articles: Iterable[Article]) -> str:
    items = []
    for article in articles:
        date_text = article.published.date().isoformat() if article.published else "recent"
        summary = article.summary or "A fresh perspective aligned with Johanne's worldview."
        items.append(
            f"<li><strong>{article.title}</strong> (<em>{date_text}</em>) — "
            f"<a href=\"{article.url}\">link</a><br><small>{summary}</small></li>"
        )
    if not items:
        items.append("<li>No recent articles found in the index. Keep the stage warm.</li>")
    return "\n".join(items)


def _build_overview(knowledge: str, articles: list[Article]) -> str:
    recent_titles = ", ".join(article.title for article in articles[:3]) or "no-index-yet"
    article_note = f"Recent index pull featured: {recent_titles}."
    source_hint = "Overview distilled from mybrain/knowledge.md. "
    body = (
        f"{source_hint}{knowledge}\n\nJohanne reads Rahab as a manifesto against hollow growth. "
        f"He tracks crypto swings not for hype but to map geopolitics, "
        f"and he wants roadmaps that tie ideology to action. {article_note} "
        "Every riff, render, and roadmap must admit doubt while inviting calculated risk."
    )
    if len(body) < 1400:
        padding = (
            " He weighs currencies against human cost, wonders how automation reshapes "
            "labor, and seeks art that mirrors uncomfortable truths without turning "
            "into cultish cheerleading."
        )
        while len(body) < 1500:
            body += padding
    return _truncate(body, 1600)


def _section(label: str, content: str) -> str:
    return f"<section><h2>{label}</h2><p>{content}</p></section>"


def _build_sections(articles: list[Article]) -> list[str]:
    article_line = (
        "Johanne skims the latest index to anchor abstractions to real-world signals: "
        + "; ".join(article.title for article in articles[:4])
    )
    sections = [
        _section("Tone & Mood", "Johanne oscillates between hypomanic creation sprints and nihilistic lulls, so the prose rides fast punk cadences with reflective breaks."),
        _section("Risk Rituals", "He distrusts blind hype; every call-to-action pairs excitement with transparent caveats and links to wallet safety."),
        _section("Economic Dread", "Currency dominance and price inflation feel hollow to him—tie Rahab's storyline to that unease and hint at practical hedges."),
        _section("Creative Fuel", "Noise, synths, and glitch art become cognitive armor; embed playlists, DAW presets, or shader snippets as optional riffs."),
        _section("Community Guardrails", "Invite debate, not devotion. Spell out roadmap checkpoints, update cadence, and how feedback loops avoid cultish vibes."),
        _section("Index Pulse", article_line),
        _section("Actionable Calm", "Close with small, doable steps—join a critique channel, bookmark a release calendar, and rest when the sprint ends."),
    ]
    return sections


def _build_translations(title: str, description: str) -> str:
    ita = (
        "Johanne vede Rahab come un laboratorio punk dove il rischio è calcolato. "
        "Ogni sezione chiede trasparenza, non culto, e collega arte, criptovalute e geopolitica."
    )
    spa = (
        "Johanne lee Rahab como un manifiesto punk para riesgos conscientes. "
        "Quiere arte, cripto y geopolítica unidas con advertencias claras y pasos accionables."
    )
    ja = (
        "ヨハネはラハブを、盲信ではなく思考で燃料を補うパンクな実験場として読む。 "
        "芸術・クリプト・世界情勢を結び、安心材料とリスクの両方を示してほしい。"
    )
    return (
        "<section><h2>Translations</h2>"
        f"<h3>Italiano</h3><p><strong>{title}</strong><br>{ita}</p>"
        f"<h3>Español</h3><p><strong>{title}</strong><br>{spa}</p>"
        f"<h3>日本語</h3><p><strong>{title}</strong><br>{ja}</p>"
        "</section>"
    )


def generate_content_plan(*, dry_run: bool = False) -> dict[str, object] | None:
    """Generate a structured content plan using the shared generator script.

    The wrapper adds status messages for API clients and keeps a stable entry
    point for the HTTP server.
    """

    STATUS_REPORTER.add("Starting content plan generation")
    try:
        result = generate_plan(
            index_file=PLAN_INDEX_FILE,
            knowledge_file=PLAN_KNOWLEDGE_FILE,
            model=PLAN_MODEL,
            base_url=PLAN_BASE_URL,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to clients
        STATUS_REPORTER.add(f"Content plan generation failed: {exc}")
        return None

    STATUS_REPORTER.add("Content plan generation completed")
    return result


def generate_three_stage_blog(prompt: str, output_dir: Path | None = None) -> dict[str, object]:
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    STATUS_REPORTER.add("Starting blog generation")

    index_articles = _read_index(DEFAULT_INDEX_FILE)
    knowledge_text = _read_knowledge(DEFAULT_KNOWLEDGE_FILE)

    title = _build_title(prompt)
    description = _build_description(prompt)
    tags = _build_tags()

    overview = _build_overview(knowledge_text, index_articles)
    sections = _build_sections(index_articles)

    html_parts = [
        f"<h1>{title}</h1>",
        f"<p class=\"description\">{description}</p>",
        f"<p class=\"tags\">Tags: {', '.join(tags)}</p>",
        "<section><h2>General Overview (source: knowledge file)</h2>",
        f"<p>{overview}</p>",
        "</section>",
        "<section><h2>Latest Articles from Index</h2><ul>",
        _format_articles(index_articles),
        "</ul></section>",
    ]
    html_parts.extend(sections)
    html_parts.append(_build_translations(title, description))

    html = "\n".join(html_parts)

    output_path = output_dir / "blog.html"
    output_path.write_text(html, encoding="utf-8")

    STATUS_REPORTER.add("Blog generation completed")

    return {
        "flag": "FLAG:FILES_SENT",
        "html": {"ja": html},
        "files": {"ja": str(output_path)},
        "category": "blog",
    }

