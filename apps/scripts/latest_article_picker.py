"""Generate an English HTML article from the latest index entry.

This script listens for the client instruction "pick one latest article and
create it" (in English) and responds by selecting a random entry from the
most recently modified index file. It then builds the standardized prompt for
Codex's writing workflow (headline+summary expansion into an HTML article with a
diagram) and optionally calls a configured LLM to generate the output.

Features
--------
- Detect client intent from a free-form message.
- Locate the newest index file under the articles directory.
- Parse index content in JSON/JSONL or plain-text formats to extract headline
  and summary pairs.
- Randomly select one entry and construct the client-facing prompt.
- Optional HTTP API for integration with other services.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

BASE_DIR = Path("/var/www/Meta-Project/indexes/articles")
DEFAULT_MODEL = os.environ.get("LATEST_ARTICLE_MODEL", "phi3:mini")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "600"))
DEFAULT_SERVER_PORT = int(os.environ.get("LATEST_ARTICLE_SERVER_PORT", "8010"))


class IndexEntry:
    """Lightweight representation of an article candidate."""

    def __init__(self, headline: str, summary: str, *, source: str = "") -> None:
        self.headline = headline.strip()
        self.summary = summary.strip()
        self.source = source

    def is_valid(self) -> bool:
        return bool(self.headline and self.summary)

    def to_dict(self) -> dict[str, str]:
        return {"headline": self.headline, "summary": self.summary, "source": self.source}


def is_latest_article_request(message: str) -> bool:
    """Return True if the message asks to pick and create the latest article.

    The detection is intentionally permissive to accommodate minor wording
    variations (e.g., "latest article", "pick one of the newest posts").
    """

    lowered = message.lower()
    keywords = ["latest article", "newest article", "pick up the latest", "one latest article"]
    return any(key in lowered for key in keywords)


def find_latest_index_file(index_dir: Path = BASE_DIR) -> Path:
    """Return the newest index file under the given directory.

    Supported extensions: .json, .jsonl, .ndjson, .txt
    """

    candidates = [
        path
        for ext in ("*.json", "*.jsonl", "*.ndjson", "*.txt")
        for path in index_dir.glob(ext)
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No index files found in {index_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_mapping(entry: Mapping[str, Any], *, source: str) -> IndexEntry | None:
    title = str(entry.get("headline") or entry.get("title") or entry.get("name") or "").strip()
    summary = str(entry.get("summary") or entry.get("description") or entry.get("content") or "").strip()
    if not title and "article" in entry:
        nested = entry.get("article")
        if isinstance(nested, Mapping):
            title = str(nested.get("headline") or nested.get("title") or "")
            summary = str(nested.get("summary") or nested.get("description") or "")
    candidate = IndexEntry(title, summary, source=source)
    return candidate if candidate.is_valid() else None


def _parse_json_payload(payload: Any, *, source: str) -> list[IndexEntry]:
    if isinstance(payload, list):
        return [entry for item in payload if (entry := _normalize_mapping(item, source=source))]
    if isinstance(payload, Mapping):
        entries: list[IndexEntry] = []
        if "entries" in payload and isinstance(payload["entries"], Iterable):
            entries.extend(
                entry
                for item in payload["entries"]
                if isinstance(payload["entries"], Iterable)
                and (entry := _normalize_mapping(item, source=source))
            )
        if not entries:
            maybe = _normalize_mapping(payload, source=source)
            entries.extend([maybe] if maybe else [])
        return entries
    return []


def _strip_markdown_fence(text: str) -> str:
    match = re.match(r"```(?:json)?\s*(.*)```\s*$", text.strip(), flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _relaxed_json_loads(candidate: str) -> Any:
    cleaned = _strip_markdown_fence(candidate)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return json.loads(cleaned)


def _parse_relaxed_json_blocks(text: str, *, source: str) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    candidates = [text]
    candidates.extend(match.group(1) for match in re.finditer(r"```json(.*?)```", text, flags=re.DOTALL))
    candidates.extend(
        match.group(1)
        for match in re.finditer(r"```(?!json)(.*?)```", text, flags=re.DOTALL)
        if match.group(1).strip().startswith("{")
    )

    for candidate in candidates:
        try:
            payload = _relaxed_json_loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        entries.extend(_parse_json_payload(payload, source=source))
        if entries:
            break

    return entries


def load_index_entries(index_file: Path) -> list[IndexEntry]:
    """Parse the index file into a list of IndexEntry objects."""

    text = index_file.read_text(encoding="utf-8", errors="replace").strip()
    entries: list[IndexEntry] = []

    # JSON or JSON Lines
    if index_file.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        try:
            payload = json.loads(text)
            entries.extend(_parse_json_payload(payload, source=index_file.name))
        except json.JSONDecodeError:
            entries.extend(_parse_relaxed_json_blocks(text, source=index_file.name))
        except Exception:  # noqa: BLE001
            entries.extend(_parse_relaxed_json_blocks(text, source=index_file.name))

        if not entries:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    entries.extend(_parse_json_payload(payload, source=index_file.name))
                    continue
                except json.JSONDecodeError:
                    pass
                entry = IndexEntry(headline=line[:120], summary=line, source=index_file.name)
                if entry.is_valid():
                    entries.append(entry)

    # Plain-text fallback: treat paragraphs as entries (first line = headline, rest = summary)
    if not entries and text:
        for block in (segment.strip() for segment in text.split("\n\n")):
            if not block:
                continue
            lines = block.splitlines()
            headline = lines[0][:120]
            summary = "\n".join(lines[1:]).strip() or headline
            entry = IndexEntry(headline=headline, summary=summary, source=index_file.name)
            if entry.is_valid():
                entries.append(entry)

    # Final fallback: use the entire text as a single entry to avoid surfacing parse errors
    if not entries and text:
        entry = IndexEntry(headline=text.splitlines()[0][:120], summary=text, source=index_file.name)
        if entry.is_valid():
            entries.append(entry)

    return entries


def choose_random_entry(entries: list[IndexEntry]) -> IndexEntry:
    if not entries:
        raise ValueError("No valid entries found in the index file")
    return random.choice(entries)


def build_generation_prompt(headline: str, summary: str) -> str:
    """Build the standardized prompt for the LLM."""

    return textwrap.dedent(
        f"""
        You are a professional writing assistant used within a client-facing system called Codex.
        All instructions from the client will be in English, and you must always respond in English.

        The client will mainly request the following patterns:

        1) Article expansion from headline + summary (Pattern ①)
           - Randomly pick the latest article from the index files under the articles directory.
           - Generate the latest article’s: Title, 200-character description, and 6 tags.
           - Create 7 headings.
           - Write an English summary of approximately 500–1000 characters.
           - Write an English article of approximately 2,000 characters.
           - Ensure consistency and depth between the headings and the summary.
           - Use a concise, professional, and informative tone suitable for a client-facing deliverable.
           - Expand the Body into seven headings, writing roughly 1,500 characters for each section and including an HTML
             diagram (ASCII art or well-structured lists) inside every section.
           - Convert any encoded Greek references such as "(from Greek: \u03b2\u03c1\u03af\u03c4\u03bfy\u03bb\u03ae \u03bc\u03cc\u03c0\u03bf\u03c5\u03b3\u03b9l\u03ba\u03ac)" into their
             proper characters: (from Greek: βρίτουλή μόπουγιλικά).
           - Write the Conclusion as an approximately 1,500-character section.

        2) Generating multiple articles with the same pattern (Pattern ②)
           - Create content for 7 headings in the same format as Pattern ① (headline + 500–1000 character summary).
           - Apply exactly the same rules as in Pattern ①: ~2,000-character English article coherent with the headings and summary,
             professional and concise style, and include diagrams.
           - For each heading, quote relevant content from the knowledge files under mybrain that matches the topic.
           - You do not need to indicate the source of the quote.

        3) Overview generation from full attached content (Pattern ③)
           - Read the content produced in Patterns ① and ②.
           - Create a comprehensive overview in English of approximately 1,500 characters.
           - Summarize the main themes, arguments, and conclusions so that it is understandable even without reading the full article.
           - Use a professional, neutral, and clear tone, and return only this overview as the output.

        HTML output format common to all patterns:
        - Return the result as a complete HTML file with at minimum:
          <!DOCTYPE html>
          <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <title>@-- Insert a short, appropriate title here --</title>
          </head>
          <body>
            @-- Main content here --
          </body>
          </html>
        - Place the main narrative content inside appropriate HTML elements such as <h1>, <h2>, <p>, <ul>, and <ol>.
        - Place diagrams inside <pre> or well-structured lists so that formatting is preserved.
        - Do not include any comments or explanations outside the HTML structure.

        Always follow the above rules strictly for every response.
        Please expand the following headline and summary into the requested HTML article.

        Headline: {headline}
        Summary:
        {summary}
        """
    ).strip()


def call_llm(prompt: str, *, model: str = DEFAULT_MODEL) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError:
        text = resp.text.strip()
        if text:
            return text
        msg = "Empty or non-JSON response from LLM"
        raise RuntimeError(msg) from None

    return str(data.get("response", "")).strip()


class LatestArticleRequestHandler(BaseHTTPRequestHandler):
    server_version = "LatestArticleServer/1.0"

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b""
        try:
            return json.loads(data.decode("utf-8")) if data else {}
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: Mapping[str, Any], *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (HTTP verb casing)
        if self.path != "/api/generate_latest_article":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        data = self._read_body()
        message = str(data.get("message", "")).strip() if isinstance(data, Mapping) else ""
        model = str(data.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
        no_call_flag = bool(data.get("no_call")) if isinstance(data, Mapping) else False
        index_dir_raw = data.get("index_dir") if isinstance(data, Mapping) else None
        index_dir = Path(str(index_dir_raw)) if index_dir_raw else BASE_DIR

        if not message:
            self._send_json({"ok": False, "error": "message_missing"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not is_latest_article_request(message):
            self._send_json(
                {"ok": False, "error": "not_latest_article_request", "message": message},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            index_file = find_latest_index_file(index_dir)
            entries = load_index_entries(index_file)
            chosen = choose_random_entry(entries)
            prompt = build_generation_prompt(chosen.headline, chosen.summary)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"ok": False, "error": "index_error", "message": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        if no_call_flag:
            self._send_json(
                {
                    "ok": True,
                    "called_llm": False,
                    "prompt": prompt,
                    "model": model,
                    "selected": chosen.to_dict(),
                    "index_file": str(index_file),
                }
            )
            return

        try:
            response_text = call_llm(prompt, model=model)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {
                    "ok": False,
                    "error": "llm_request_failed",
                    "message": str(exc),
                    "prompt": prompt,
                    "selected": chosen.to_dict(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(
            {
                "ok": True,
                "called_llm": True,
                "prompt": prompt,
                "response": response_text,
                "model": model,
                "selected": chosen.to_dict(),
                "index_file": str(index_file),
            }
        )


def handle_message(
    message: str,
    *,
    index_dir: Path = BASE_DIR,
    model: str = DEFAULT_MODEL,
    no_call: bool = False,
) -> dict[str, Any]:
    """Process a single client message and optionally call the LLM."""

    if not is_latest_article_request(message):
        raise ValueError("Client message does not request the latest article")

    index_file = find_latest_index_file(index_dir)
    entries = load_index_entries(index_file)
    chosen = choose_random_entry(entries)
    prompt = build_generation_prompt(chosen.headline, chosen.summary)

    if no_call:
        return {
            "ok": True,
            "called_llm": False,
            "prompt": prompt,
            "model": model,
            "selected": chosen.to_dict(),
            "index_file": str(index_file),
        }

    response_text = call_llm(prompt, model=model)
    return {
        "ok": True,
        "called_llm": True,
        "prompt": prompt,
        "response": response_text,
        "model": model,
        "selected": chosen.to_dict(),
        "index_file": str(index_file),
    }


def run_server(port: int = DEFAULT_SERVER_PORT, *, index_dir: Path = BASE_DIR) -> None:
    """Start the HTTP server for client integration."""

    class _Handler(LatestArticleRequestHandler):
        INDEX_DIR = index_dir

    HTTPServer(("", port), _Handler).serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", help="Client message requesting the latest article")
    parser.add_argument("--index-dir", type=Path, default=BASE_DIR, help="Directory containing index files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--no-call", action="store_true", help="Do not call the LLM; only print the prompt")
    parser.add_argument("--serve", action="store_true", help="Run as an HTTP server")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help="Port for the HTTP server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        run_server(port=args.port, index_dir=args.index_dir)
        return

    if not args.message:
        raise SystemExit("--message is required unless --serve is used")

    result = handle_message(
        args.message,
        index_dir=args.index_dir,
        model=args.model,
        no_call=args.no_call,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
