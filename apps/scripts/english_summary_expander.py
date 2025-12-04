"""Build and optionally send the English summary-expansion prompt to an LLM.

This utility helps operators hand a headline and 500–1000 character summary to
an LLM while enforcing the client's formatting expectations: roughly 2,000
characters of polished English plus a diagram (ASCII or structured bullets).
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import Optional

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "600"))
DEFAULT_MODEL = os.environ.get("SUMMARY_EXPANDER_MODEL", "phi3:mini")


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
        You are a professional writer. I will provide:
        1) A headline.
        2) A summary between 500–1000 characters (English).

        Your task:
        - Produce an English response of about 2,000 characters.
        - Include a clear diagram (use ASCII art or structured bullets) to illustrate key relationships or flows.
        - Maintain coherence and depth consistent with the headline and summary.
        - Use concise, informative language suitable for a client-facing deliverable.

        Please respond with:
        1) A polished ~2,000-character narrative expanding on the summary.
        2) A diagram that visually explains the main concepts or processes.
        3) Ensure the tone is professional and the structure is easy to follow.

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or send a client LLM prompt")
    parser.add_argument("--headline", required=True, help="Headline text for the deliverable")

    summary_group = parser.add_mutually_exclusive_group(required=True)
    summary_group.add_argument("--summary", help="500–1000 character summary text")
    summary_group.add_argument("--summary-file", type=Path, help="Path to a file containing the summary")

    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model identifier (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--no-call",
        action="store_true",
        help="Only print the constructed prompt without sending it to the LLM",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
