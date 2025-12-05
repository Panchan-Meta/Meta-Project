"""Client instruction responder for automated blog generation.

This module orchestrates the multi-step LLM workflow described by the client:
1) Read the most recent index file that matches provided keywords.
2) Ask the LLM for metadata (title, description, tags, summary).
3) Ask for detailed article copy plus a diagram snippet.
4) Ask for a concluding synthesis.

All prompts are issued in English while keeping the source content as context.
The functions rely on a local Ollama API endpoint and default models but can be
customised per request.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
DEFAULT_MODEL = "llama3:8b"
DEFAULT_PROVIDER = "ollama"
DEFAULT_SAVE_DIR = Path(os.environ.get("BLOG_OUTPUT_DIR", "/var/www/Meta-Project/data/blogs"))
DEFAULT_INDEX_DIR = Path(
    os.environ.get("INDEX_ROOT", "/var/www/Meta-Project/indexes")
)
FALLBACK_INDEX_DIR = PROJECT_ROOT / "indexes"
DEFAULT_KNOWLEDGE_ROOT = Path(
    os.environ.get("KNOWLEDGE_ROOT", "/var/www/Meta-Project/indexes/mybrain")
)
DIAGRAM_MODEL = "codegemma:2b"

KEYWORDS = [
    "Punk Rock NFT",
    "Web3 Dystopia",
    "Crypto Resistance",
    "Spy × Punk Girls",
    "Chain of Heresy",
    "Faith vs. Market",
    "Digital Anarchy",
    "Post-Dollar World",
    "Shadow Intelligence",
    "Mental Health × NFT",
    "Punk Theology",
    "Risk & Redemption",
    "Cyber Confession",
    "Underground Worship",
    "Geopolitics & Girls",
    "Broken Utopia",
    "DeFi Church",
    "World Bug Hunters",
    "Anti-Propaganda Art",
    "Hold Your Anxiety",
]


@dataclass
class LLMResult:
    model: str
    prompt: str
    response: str


class LLMClient:
    """Minimal Ollama client for single-shot generations."""

    def __init__(self, api_base: str = DEFAULT_API_BASE) -> None:
        self.api_base = api_base.rstrip("/")

    def generate(self, model: str, prompt: str, timeout: int = 480) -> LLMResult:
        payload = {"model": model, "prompt": prompt, "stream": False}
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.api_base}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 (local API)
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:  # pragma: no cover - network exceptions
            raise RuntimeError(f"Failed to reach LLM API: {exc}") from exc

        content = data.get("response") or ""
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected LLM response: {data!r}")

        return LLMResult(model=model, prompt=prompt, response=content.strip())


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _gather_knowledge(root: Path) -> str:
    """Concatenate text files under the knowledge root, skipping Markdown."""

    if not root.exists():
        return ""

    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".md":
            continue
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def _file_matches_keywords(path: Path, keywords: Iterable[str]) -> bool:
    haystack = f"{path.name}\n{_read_optional_text(path)}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def _find_latest_index(index_root: Path, keywords: Iterable[str]) -> Path:
    search_roots = [index_root]
    if FALLBACK_INDEX_DIR not in search_roots:
        search_roots.append(FALLBACK_INDEX_DIR)

    for root in search_roots:
        candidates = [
            path
            for path in root.rglob("*.json")
            if _file_matches_keywords(path, keywords)
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

        fallback = root / "index.json"
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"No index files found in {search_roots} matching keywords"
    )


def _build_metadata_prompt(index_text: str) -> str:
    return (
        "You are drafting English blog metadata from the latest index file.\n"
        "Return compact JSON with the following keys: title, description, tags (exactly 6 items), summary.\n"
        "Constraints:\n"
        "- Title: concise and under 60 characters.\n"
        "- Description: about 200 characters.\n"
        "- Tags: six short English tags.\n"
        "- Summary: about 500 characters.\n"
        "Use the provided keywords and index content as context and do not invent unrelated topics.\n\n"
        f"Keywords:\n{json.dumps(KEYWORDS, ensure_ascii=False)}\n\n"
        f"Index content:\n{index_text}\n\n"
        "Respond with JSON only."
    )


def _build_article_prompt(metadata: dict[str, object], index_text: str, knowledge: str) -> str:
    return (
        "Write an English blog article (~1,500 characters) expanding the provided metadata.\n"
        "Blend the index takeaways with the knowledge file. Maintain a coherent narrative"
        " about punk, Web3, and the listed themes without adding URLs.\n"
        "Output plain text paragraphs.\n\n"
        f"Metadata JSON:\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Index content:\n{index_text}\n\n"
        f"Knowledge file:\n{knowledge}\n"
    )


def _build_diagram_prompt(metadata: dict[str, object], index_text: str) -> str:
    return (
        "Produce a self-contained HTML snippet with inline CSS and minimal JavaScript"
        " that visualizes the article themes. Avoid external assets."
        " Include a short caption in the markup.\n"
        "Return only HTML.\n\n"
        f"Metadata JSON:\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Index content:\n{index_text}\n"
    )


def _build_conclusion_prompt(metadata: dict[str, object], article: str, diagram_html: str) -> str:
    return (
        "Summarize the article and diagram into an English conclusion of about 1,500 characters."
        " Keep the tone reflective and actionable. Do not repeat the full HTML, only reference"
        " what the diagram represents.\n"
        f"Metadata JSON:\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Article body:\n{article}\n\n"
        f"Diagram outline:\n{diagram_html}\n"
    )


def _parse_json_response(content: str) -> dict[str, object]:
    """Parse JSON content while tolerating markdown fences or extra text."""

    def _loads(candidate: str) -> dict[str, object]:
        return json.loads(candidate)

    candidates: list[str] = [content]

    # Remove markdown-style code fences (```json ... ```)
    fence_pattern = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)
    stripped = fence_pattern.sub("", content)
    if stripped != content:
        candidates.append(stripped)

    # Extract the first JSON object if extra narration surrounds it
    object_match = re.search(r"\{[\s\S]*\}", content)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            return _loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM did not return valid JSON")


def respond_to_instruction(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    api_base: str = DEFAULT_API_BASE,
    filename: str | None = None,
) -> dict[str, object]:
    """Generate blog HTML following the requested multi-step workflow."""

    index_root = DEFAULT_INDEX_DIR if DEFAULT_INDEX_DIR.exists() else FALLBACK_INDEX_DIR
    knowledge_root = (
        DEFAULT_KNOWLEDGE_ROOT if DEFAULT_KNOWLEDGE_ROOT.exists() else index_root / "mybrain"
    )
    client = LLMClient(api_base=api_base)

    latest_index_path = _find_latest_index(index_root, KEYWORDS)
    index_text = _read_optional_text(latest_index_path)
    knowledge = _gather_knowledge(knowledge_root)
    if not knowledge:
        fallback_md = knowledge_root / "knowledge.md"
        knowledge = _read_optional_text(fallback_md)

    metadata_prompt = _build_metadata_prompt(index_text)
    metadata_raw = client.generate(model=model, prompt=metadata_prompt)
    metadata = _parse_json_response(metadata_raw.response)

    article_prompt = _build_article_prompt(metadata, index_text, knowledge)
    article_result = client.generate(model=model, prompt=article_prompt)

    diagram_prompt = _build_diagram_prompt(metadata, index_text)
    diagram_result = client.generate(model=DIAGRAM_MODEL, prompt=diagram_prompt)

    conclusion_prompt = _build_conclusion_prompt(
        metadata, article_result.response, diagram_result.response
    )
    conclusion_result = client.generate(model=model, prompt=conclusion_prompt)

    html_parts = [
        "<article>",
        f"<h1>{metadata.get('title', 'Blog Draft')}</h1>",
        "<section>",
        f"<p><strong>Description:</strong> {metadata.get('description', '')}</p>",
        f"<p><strong>Tags:</strong> {', '.join(map(str, metadata.get('tags', [])))}" "</p>",
        f"<p><strong>Summary:</strong> {metadata.get('summary', '')}</p>",
        "</section>",
        "<section><h2>Article</h2>",
        f"<p>{article_result.response}</p>",
        "</section>",
        "<section><h2>Diagram</h2>",
        diagram_result.response,
        "</section>",
        "<section><h2>Conclusion</h2>",
        f"<p>{conclusion_result.response}</p>",
        "</section>",
        "</article>",
    ]

    html = "\n".join(html_parts)

    saved_path: str | None = None
    if filename:
        DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        target = DEFAULT_SAVE_DIR / filename
        target.write_text(html, encoding="utf-8")
        saved_path = str(target)

    return {
        "ok": True,
        "html": html,
        "filename": filename,
        "path": saved_path,
        "model": model,
        "provider": provider,
        "metadata": metadata,
    }


__all__ = [
    "respond_to_instruction",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_API_BASE",
]
