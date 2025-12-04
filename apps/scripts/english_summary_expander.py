"""Build and optionally send the English summary-expansion prompt to an LLM.

This utility helps operators hand a headline and 500–1000 character summary to
an LLM while enforcing the client's formatting expectations: roughly 2,000
characters of polished English plus a diagram (ASCII or structured bullets),
returned as a complete HTML document. The prompt mirrors the latest
client-facing guidance for all patterns (single article, repeated articles,
and overviews from provided content).
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "600"))
DEFAULT_MODEL = os.environ.get("SUMMARY_EXPANDER_MODEL", "phi3:mini")
DEFAULT_SERVER_PORT = int(os.environ.get("SUMMARY_EXPANDER_PORT", "8000"))


def build_client_prompt(headline: str, summary: str) -> str:
    """Return the standardized client prompt for the LLM.

    Parameters
    ----------
    headline: str
        The headline to anchor the narrative.
    summary: str
        A 500–1000 character English summary provided by the client.
    """

    headline = headline.strip()
    summary = summary.strip()
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
           - Organize the Body into clearly labeled sections every ~2,000 characters, give each section its own heading,
             and place an HTML illustration directly after every paragraph to visualize that paragraph.
           - Do not format the Body or any section as JSON or key/value pairs—use standard HTML headings, paragraphs, and
             diagram blocks only.
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
        Please produce the requested HTML given the following inputs.
        Headline: {headline}
        Summary:
        {summary}
        """
    ).strip()


def call_llm(prompt: str, *, model: str = DEFAULT_MODEL) -> str:
    """Send a single prompt to the configured LLM and return its response."""

    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", "")).strip()


def _read_summary(summary: Optional[str], summary_file: Optional[Path]) -> str:
    if summary is not None:
        return summary
    if summary_file is None:
        return ""
    return summary_file.read_text(encoding="utf-8", errors="replace")


class SummaryRequestHandler(BaseHTTPRequestHandler):
    server_version = "SummaryExpanderServer/1.0"

    def _read_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b""
        try:
            return json.loads(data.decode("utf-8")) if data else {}
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (HTTP verb casing)
        if self.path != "/api/expand_summary":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        data = self._read_body()
        headline = str(data.get("headline", "")).strip() if isinstance(data, dict) else ""
        summary = str(data.get("summary", "")).strip() if isinstance(data, dict) else ""
        model = str(data.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
        no_call_flag = bool(data.get("no_call")) if isinstance(data, dict) else False

        if not headline or not summary:
            self._send_json({"ok": False, "error": "headline_or_summary_missing"}, status=HTTPStatus.BAD_REQUEST)
            return

        prompt = build_client_prompt(headline, summary)
        if no_call_flag:
            self._send_json({"ok": True, "prompt": prompt, "called_llm": False, "response": "", "model": model})
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
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(
            {"ok": True, "prompt": prompt, "response": response_text, "called_llm": True, "model": model}
        )


def run_server(host: str = "0.0.0.0", port: int = DEFAULT_SERVER_PORT) -> None:
    server = HTTPServer((host, port), SummaryRequestHandler)
    print(f"Serving summary expander on http://{host}:{port}/api/expand_summary")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or send a client LLM prompt")
    parser.add_argument("--serve", action="store_true", help="Run an HTTP server for client requests")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for --serve (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Port for --serve (default: {DEFAULT_SERVER_PORT})",
    )
    parser.add_argument("--headline", help="Headline text for the deliverable")

    summary_group = parser.add_mutually_exclusive_group(required=False)
    summary_group.add_argument("--summary", help="500–1000 character summary text")
    summary_group.add_argument("--summary-file", type=Path, help="Path to a file containing the summary")

    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model identifier (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--no-call",
        action="store_true",
        help="Only print the constructed prompt without sending it to the LLM",
    )

    args = parser.parse_args()
    if not args.serve:
        if not args.headline:
            parser.error("--headline is required unless using --serve")
        if args.summary is None and args.summary_file is None:
            parser.error("either --summary or --summary-file is required unless using --serve")
    return args


def main() -> int:
    args = parse_args()
    if args.serve:
        run_server(host=args.host, port=args.port)
        return 0

    summary_text = _read_summary(args.summary, args.summary_file)
    prompt = build_client_prompt(args.headline, summary_text)

    print("=== LLM PROMPT (copy/paste to your model if needed) ===")
    print(prompt)
    print()

    if args.no_call:
        return 0

    try:
        response = call_llm(prompt, model=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] LLM request failed: {exc}", flush=True)
        return 1

    print("=== LLM RESPONSE ===")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
