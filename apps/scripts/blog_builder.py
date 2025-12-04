"""Orchestrate blog generation flows.

Currently supports rendering the static draft (blog_draft.md) into HTML by
leveraging the generate_blog_from_draft module. Designed so other builders can
be added later without changing callers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from generate_blog_from_draft import (
    DEFAULT_DRAFT,
    DEFAULT_INDEX,
    DEFAULT_OUTPUT,
    build_blog_from_draft,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build blog artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    draft_cmd = sub.add_parser(
        "from_draft", help="Render the predefined draft and index metadata to HTML"
    )
    draft_cmd.add_argument(
        "--draft", type=Path, default=DEFAULT_DRAFT, help="Path to the draft Markdown"
    )
    draft_cmd.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX, help="Path to index.json"
    )
    draft_cmd.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the generated HTML",
    )
    draft_cmd.add_argument(
        "--keyword", help="Keyword to prioritize when choosing the latest index entry"
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)

    if args.command == "from_draft":
        result = build_blog_from_draft(
            draft_path=args.draft,
            index_path=args.index,
            output_path=args.output,
            keyword=args.keyword,
        )
        print(
            "[from_draft] Generated blog HTML at "
            f"{result['output']} (keyword: {result['keyword']})"
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
