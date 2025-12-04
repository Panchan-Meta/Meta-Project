"""Generate a static blog HTML from the draft and index metadata.

This script reads `indexes/mybrain/blog_draft.md`, picks a keyword,
selects the most recent index entry that matches it, and renders the draft
into a minimal HTML article that keeps the original sections and diagrams.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = ROOT / "indexes" / "mybrain" / "blog_draft.md"
DEFAULT_INDEX = ROOT / "indexes" / "index.json"
DEFAULT_OUTPUT = ROOT / "apps" / "scripts" / "blog_output.html"

BLOG_KEYWORDS = [
    "Punk Rock NFT",
    "Web3 Dystopia（ウェブスリー・ディストピア）",
    "Crypto Resistance（クリプト・レジスタンス）",
    "Spy × Punk Girls（スパイ×パンクガールズ）",
    "Chain of Heresy（異端者たちのチェーン）",
    "Faith vs. Market（信仰かマーケットか）",
    "Digital Anarchy（デジタル・アナーキー）",
    "Post-Dollar World（ポスト・ドル世界）",
    "Shadow Intelligence（影のインテリジェンス）",
    "Mental Health × NFT（メンタルヘルスNFT）",
    "Punk Theology（パンク神学）",
    "Risk & Redemption（リスクと贖い）",
    "Cyber Confession（サイバー懺悔室）",
    "Underground Worship（アンダーグラウンド礼拝）",
    "Geopolitics & Girls（地政学と少女たち）",
    "Broken Utopia（壊れたユートピア）",
    "DeFi Church（ディーファイ教会）",
    "World Bug Hunters（世界のバグハンター）",
    "Anti-Propaganda Art（アンチプロパガンダ・アート）",
    "Hold Your Anxiety（不安ごとHODLする）",
]


@dataclass
class Section:
    title: str
    body: str
    diagram: str | None = None


@dataclass
class Draft:
    title: str
    description: str
    tags: list[str]
    overview: str
    sections: list[Section]
    conclusion: str


@dataclass
class IndexEntry:
    title: str
    url: str
    summary: str
    published: str

    @property
    def published_dt(self) -> datetime:
        try:
            return datetime.fromisoformat(self.published)
        except Exception:
            return datetime.min


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the draft into an HTML blog post.")
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT, help="Path to the draft Markdown file")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to index.json")
    parser.add_argument("--keyword", help="Keyword to prioritize when picking the latest index entry")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the generated HTML (default: apps/scripts/blog_output.html)",
    )
    return parser.parse_args(argv)


def _paragraphs(text: str) -> list[str]:
    blocks: list[str] = []
    for raw in text.strip().split("\n\n"):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        blocks.append(" ".join(lines))
    return blocks


def _split_body_and_diagram(text: str) -> tuple[str, str | None]:
    marker = "図解（HTML）:"
    if marker not in text:
        return text.strip(), None
    head, tail = text.split(marker, 1)
    return head.strip(), tail.strip()


def parse_draft(path: Path) -> Draft:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    title_line = next((line for line in lines if line.startswith("# ")), "# 無題")
    title = title_line.removeprefix("# ").strip()

    sections: list[Tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.removeprefix("## ").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))

    def find_block(name: str) -> str:
        for sec_title, sec_body in sections:
            if sec_title.startswith(name):
                return sec_body.strip()
        return ""

    def find_sections(prefix: str) -> list[Section]:
        collected: list[Section] = []
        for sec_title, sec_body in sections:
            if not sec_title.startswith(prefix):
                continue
            body_text, diagram_html = _split_body_and_diagram(sec_body)
            collected.append(Section(title=sec_title, body=body_text, diagram=diagram_html))
        return collected

    description = find_block("ディスクリプション")
    tags_block = find_block("タグ")
    tags = [line.removeprefix("-").strip() for line in tags_block.splitlines() if line.startswith("-")]
    overview = find_block("概論")
    conclusion = find_block("総論")
    body_sections = find_sections("セクション")

    return Draft(
        title=title,
        description=description,
        tags=tags,
        overview=overview,
        sections=body_sections,
        conclusion=conclusion,
    )


def load_index_entries(path: Path) -> list[IndexEntry]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries: list[IndexEntry] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            entries.append(
                IndexEntry(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    summary=str(item.get("summary", "")),
                    published=str(item.get("published", "")),
                )
            )
    return entries


def select_latest_entry(entries: list[IndexEntry], keyword: str) -> IndexEntry | None:
    if not entries:
        return None
    kw = keyword.casefold()

    def matches(entry: IndexEntry) -> bool:
        haystack = f"{entry.title} {entry.summary}".casefold()
        return kw in haystack

    filtered = [entry for entry in entries if matches(entry)]
    candidates = filtered or entries
    candidates.sort(key=lambda e: e.published_dt, reverse=True)
    return candidates[0] if candidates else None


def _render_paragraphs(blocks: Iterable[str]) -> str:
    return "\n".join(f"    <p>{para}</p>" for para in blocks)


def _render_section(section: Section) -> str:
    paragraphs = _render_paragraphs(_paragraphs(section.body))
    diagram_block = f"\n    {section.diagram}" if section.diagram else ""
    return (
        f"  <section class=\"body\">\n"
        f"    <h2>{section.title}</h2>\n"
        f"{paragraphs}\n"
        f"{diagram_block}\n"
        f"  </section>"
    )


def render_html(
    draft: Draft,
    entry: IndexEntry | None,
    keyword: str,
    output_path: Path,
    draft_path: Path,
) -> str:
    entry_block = (
        "<p>インデックスから最新の記事を取得できませんでした。</p>"
        if entry is None
        else (
            "<ul>"
            f"<li>タイトル: {entry.title}</li>"
            f"<li>URL: {entry.url}</li>"
            f"<li>要約: {entry.summary}</li>"
            f"<li>公開日時: {entry.published}</li>"
            "</ul>"
        )
    )

    sections_html = "\n".join(_render_section(section) for section in draft.sections)

    html = (
        "<!doctype html>\n"
        "<html lang=\"ja\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\" />\n"
        "  <title>自動生成ブログ</title>\n"
        "  <style>\n"
        "    body { font-family: 'Noto Sans JP', system-ui, sans-serif; line-height: 1.7; margin: 1.5rem; color: #111827; }\n"
        "    header, footer { background: #f3f4f6; padding: 1rem; border-radius: 12px; }\n"
        "    h1 { margin: 0 0 0.5rem; font-size: 1.7rem; }\n"
        "    h2 { margin-top: 1.25rem; font-size: 1.3rem; }\n"
        "    ul.tags { list-style: none; padding: 0; display: flex; gap: 0.5rem; flex-wrap: wrap; }\n"
        "    ul.tags li { background: #e5e7eb; padding: 0.25rem 0.6rem; border-radius: 999px; }\n"
        "    section.body { margin: 1.5rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #e5e7eb; }\n"
        "    figure.diagram { background: #f9fafb; border: 1px dashed #d1d5db; padding: 0.75rem; border-radius: 10px; margin-top: 0.75rem; }\n"
        "    figure.diagram figcaption { font-weight: 600; margin-bottom: 0.5rem; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <article>\n"
        "    <header>\n"
        f"      <h1>{draft.title}</h1>\n"
        f"      <p>{draft.description}</p>\n"
        "      <ul class=\"tags\">\n"
        + "\n".join(f"        <li>{tag}</li>" for tag in draft.tags)
        + "\n      </ul>\n"
        f"      <p>選択キーワード: {keyword}</p>\n"
        "    </header>\n"
        "    <section class=\"summary\">\n"
        "      <h2>概論</h2>\n"
        f"{_render_paragraphs(_paragraphs(draft.overview))}\n"
        "    </section>\n"
        "    <section class=\"source\">\n"
        "      <h2>参照した最新インデックス</h2>\n"
        f"      {entry_block}\n"
        "    </section>\n"
        f"{sections_html}\n"
        "    <footer>\n"
        "      <h2>総論</h2>\n"
        f"{_render_paragraphs(_paragraphs(draft.conclusion))}\n"
        f"      <p>生成元ドラフト: {draft_path}</p>\n"
        f"      <p>生成先: {output_path}</p>\n"
        "    </footer>\n"
        "  </article>\n"
        "</body>\n"
        "</html>\n"
    )
    return html


def build_blog_from_draft(
    *,
    draft_path: Path = DEFAULT_DRAFT,
    index_path: Path = DEFAULT_INDEX,
    output_path: Path = DEFAULT_OUTPUT,
    keyword: str | None = None,
) -> dict[str, object]:
    """Render the draft to HTML and persist it.

    Returns a small dict that can be used by orchestrators (e.g. blog_builder.py)
    to log or chain follow-up actions.
    """
    draft = parse_draft(draft_path)
    chosen_keyword = keyword or random.choice(BLOG_KEYWORDS)
    entries = load_index_entries(index_path)
    latest_entry = select_latest_entry(entries, chosen_keyword)

    html = render_html(draft, latest_entry, chosen_keyword, output_path, draft_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return {
        "html": html,
        "output": output_path,
        "keyword": chosen_keyword,
        "entry": latest_entry,
        "draft": draft,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    result = build_blog_from_draft(
        draft_path=args.draft,
        index_path=args.index,
        output_path=args.output,
        keyword=args.keyword,
    )
    print(
        "Generated blog HTML at "
        f"{result['output']} (keyword: {result['keyword']})"
    )


if __name__ == "__main__":
    main()
