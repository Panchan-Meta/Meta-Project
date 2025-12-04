"""Generate structured Japanese content prompts from index and knowledge files.

This script selects a random article from the index, builds a detailed prompt
for an LLM, and optionally calls an Ollama-compatible endpoint to obtain the
filled content. It enforces the requested sections: title, 200-character
description, tiered tags, 1000-character overview, 1500-character body with
seven headings and diagram notes, and a 1500-character conclusion informed by
mybrain knowledge.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_FILE = PROJECT_ROOT / "indexes" / "index.json"
DEFAULT_KNOWLEDGE_FILE = PROJECT_ROOT / "mybrain" / "knowledge.md"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3:8b"


@dataclass
class IndexEntry:
    """A single record from the index file."""

    title: str
    url: str
    summary: str
    published: str | None


class GenerationError(RuntimeError):
    """Raised when the LLM endpoint returns an unexpected response."""


def load_index(path: Path) -> list[IndexEntry]:
    """Parse the index JSON into typed entries."""

    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[IndexEntry] = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        entries.append(
            IndexEntry(
                title=str(entry.get("title", "Untitled")),
                url=str(entry.get("url", "")),
                summary=str(entry.get("summary", "")),
                published=str(entry.get("published")) if entry.get("published") else None,
            )
        )
    return entries


def load_knowledge(path: Path) -> str:
    """Read the knowledge file as UTF-8 text."""

    if not path.is_file():
        raise FileNotFoundError(f"Knowledge file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def pick_random_entry(entries: list[IndexEntry]) -> IndexEntry | None:
    """Return a random index entry when available."""

    if not entries:
        return None
    return random.choice(entries)


def format_index_context(entry: IndexEntry | None, all_entries: list[IndexEntry]) -> str:
    """Describe the randomly chosen entry plus a short list of alternates."""

    if entry is None:
        return "インデックスが空です。テンプレートに従い汎用的な内容を作成してください。"

    bullet_lines = [
        f"- ランダム抽出: {entry.title} (URL: {entry.url or '不明'})",
        f"  要約: {entry.summary or '要約なし'}",
    ]

    if entry.published:
        bullet_lines.append(f"  公開日: {entry.published}")

    # Add up to three other recent titles to keep the LLM aware of surrounding topics.
    others = [candidate for candidate in all_entries if candidate is not entry][:3]
    for idx, other in enumerate(others, start=1):
        bullet_lines.append(f"- 参考{idx}: {other.title} — {other.summary}")

    return "\n".join(bullet_lines)


def build_prompt(entry: IndexEntry | None, knowledge: str, all_entries: list[IndexEntry]) -> str:
    """Create the Japanese prompt instructing the LLM to deliver the full structure."""

    index_context = format_index_context(entry, all_entries)
    return f"""
最新情報として以下のインデックス内容を参照してください。
{index_context}

加えてナレッジベース(mybrain/knowledge.md)の要旨を踏まえてください:
{knowledge}

# 要求
- すべて日本語で書くこと。
- 各セクションの文字数目安を守り、過剰な空白を避けてください。
- 図解ではイメージを短く説明するだけで、ASCIIアートは不要。
- タグは6つ: ビッグ2件、普通2件、スモール2件。ラベルを `[BIG]`, `[MID]`, `[SMALL]` と先頭に付ける。

# 出力フォーマット
タイトル: <ここにタイトル>
ディスクリプション(200文字): <200文字で要約>
タグ:
- [BIG] ...
- [BIG] ...
- [MID] ...
- [MID] ...
- [SMALL] ...
- [SMALL] ...

概要(1000文字): <インデックス情報を元にした概要>

本文(約1500文字): 見出しごとに本文と図解を含める。
- 見出し1: <タイトル>
  図解: <図の説明>
- 見出し2: <タイトル>
  図解: <図の説明>
- 見出し3: <タイトル>
  図解: <図の説明>
- 見出し4: <タイトル>
  図解: <図の説明>
- 見出し5: <タイトル>
  図解: <図の説明>
- 見出し6: <タイトル>
  図解: <図の説明>
- 見出し7: <タイトル>
  図解: <図の説明>

総論(1500文字): 本文を踏まえ、mybrain知識を再引用しながら核心をまとめる。
""".strip()


def generate_plan(
    *,
    index_file: Path = DEFAULT_INDEX_FILE,
    knowledge_file: Path = DEFAULT_KNOWLEDGE_FILE,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    dry_run: bool = False,
) -> dict[str, object]:
    """Produce a content plan by sampling the index and invoking an LLM.

    This helper is import-friendly for other modules (e.g., blog_server,
    blog_builder) so they can reuse the same prompt structure without
    shelling out to the script.
    """

    index_entries = load_index(index_file)
    knowledge = load_knowledge(knowledge_file)
    picked_entry = pick_random_entry(index_entries)
    prompt = build_prompt(picked_entry, knowledge, index_entries)

    if dry_run:
        return {
            "prompt": prompt,
            "content": None,
            "entry": picked_entry.__dict__ if picked_entry else None,
        }

    content = call_ollama(prompt, model=model, base_url=base_url)
    return {
        "prompt": prompt,
        "content": content,
        "entry": picked_entry.__dict__ if picked_entry else None,
    }


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> str:
    """Send the prompt to an Ollama-compatible /api/generate endpoint."""

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:  # noqa: PERF203 - explicit error path improves UX
        raise GenerationError(f"LLM endpoint unreachable at {base_url}: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GenerationError("LLM endpoint returned non-JSON response") from exc

    text = data.get("response") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise GenerationError("LLM endpoint response missing 'response' text")
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMコンテンツ生成用のプロンプト実行スクリプト")
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE, help="インデックスJSONのパス")
    parser.add_argument("--knowledge-file", type=Path, default=DEFAULT_KNOWLEDGE_FILE, help="ナレッジファイルのパス")
    parser.add_argument("--output", type=Path, help="生成結果を保存するパス (拡張子は自由)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollamaモデル名")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama APIのベースURL")
    parser.add_argument("--dry-run", action="store_true", help="プロンプトのみ出力しLLMへ送信しない")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_entries = load_index(args.index_file)
    knowledge = load_knowledge(args.knowledge_file)
    picked = pick_random_entry(index_entries)
    prompt = build_prompt(picked, knowledge, index_entries)

    if args.dry_run:
        print(prompt)
        return

    content = call_ollama(prompt, model=args.model, base_url=args.base_url)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
