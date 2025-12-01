"""
Script to scrape specified language/style guide, documentation, and RSS resources.

The script creates per-category directories (languages plus Design/Architecture)
and downloads each configured URL into its category directory. Files older than
seven days are removed before fetching new content.
"""
from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

BASE_DIR = Path(__file__).resolve().parent / "scraped_data"

CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "Python": (
        "https://peps.python.org/pep-0008/",
        "https://docs.python.org/",
        "https://rss.feedspot.com/folder/5hvItF4d6A==/rss/rsscombiner",
    ),
    "JavaScript": (
        "https://github.com/airbnb/javascript",
        "https://google.github.io/styleguide/jsguide.html",
        "https://ecma-international.org/publications-and-standards/standards/ecma-262/",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference",
        "https://rss.feedspot.com/folder/5hvItF4d6Q==/rss/rsscombiner",
    ),
    "Java": (
        "https://www.oracle.com/java/technologies/javase/codeconventions-contents.html",
        "https://docs.oracle.com/en/java/javase/22/docs/api/",
        "https://rss.feedspot.com/folder/5hvItF4d6g==/rss/rsscombiner",
    ),
    "TypeScript": (
        "https://www.typescriptlang.org/docs/handbook/",
        "https://www.typescriptlang.org/docs/",
        "https://rss.feedspot.com/folder/5hvItF4e4w==/rss/rsscombiner",
    ),
    "Swift": (
        "https://swift.org/documentation/api-design-guidelines/",
        "https://docs.swift.org/swift-book/documentation/the-swift-programming-language/",
        "https://developer.apple.com/documentation/swift/swift-standard-library",
        "https://rss.feedspot.com/folder/5hvItF4e5A==/rss/rsscombiner",
    ),
    "Kotlin": (
        "https://kotlinlang.org/docs/coding-conventions.html",
        "https://kotlinlang.org/api/core/kotlin-stdlib/",
        "https://kotlinlang.org/docs/home.html",
        "https://rss.feedspot.com/folder/5hvItF4e5Q==/rss/rsscombiner",
    ),
    "Design": (
        "https://rss.feedspot.com/folder/5hvIs2AZ5g==/rss/rsscombiner",
        "https://rss.feedspot.com/folder/5hvIs2AZ6A==/rss/rsscombiner",
        "https://rss.feedspot.com/folder/5hvIs2AZ6g==/rss/rsscombiner",
    ),
    "Architecture": (
        "https://rss.feedspot.com/folder/5hvIs2AZ6Q==/rss/rsscombiner",
    ),
}

OLD_FILE_THRESHOLD = timedelta(days=7)


def sanitize_filename(url: str) -> str:
    """Convert a URL into a safe filename fragment."""
    parsed = re.sub(r"[^a-zA-Z0-9]+", "_", url)
    trimmed = parsed.strip("_")
    return trimmed or "resource"


def remove_old_files(category_dir: Path, threshold: timedelta) -> None:
    now = datetime.now(timezone.utc)
    for path in category_dir.glob("*"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if now - modified > threshold:
            path.unlink(missing_ok=True)


def fetch_and_save(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    destination.write_bytes(content)


def ensure_category(category: str, urls: Iterable[str]) -> None:
    category_dir = BASE_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    remove_old_files(category_dir, OLD_FILE_THRESHOLD)

    for url in urls:
        filename = sanitize_filename(url) + ".html"
        target = category_dir / filename
        try:
            fetch_and_save(url, target)
            print(f"Saved {url} -> {target}")
        except urllib.error.HTTPError as exc:
            print(f"HTTP error while fetching {url}: {exc}", file=sys.stderr)
        except urllib.error.URLError as exc:
            print(f"Request error while fetching {url}: {exc}", file=sys.stderr)
        except OSError as exc:
            print(f"Filesystem error for {target}: {exc}", file=sys.stderr)


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for category, urls in CATEGORIES.items():
        ensure_category(category, urls)


if __name__ == "__main__":
    main()
