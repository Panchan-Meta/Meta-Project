"""Scrape Feedspot RSS feeds and download full articles plus linked pages.

For each configured category, the script:
- Creates a directory for the category if it does not already exist.
- Removes files older than the configured threshold (3 days).
- Downloads the RSS feed, extracts post links, and saves each article.
- Crawls the article for hyperlinks and saves each linked page once.
"""
from __future__ import annotations

import hashlib
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

BASE_DIR = Path("indexes")
OLD_FILE_THRESHOLD = timedelta(days=3)

CATEGORIES: dict[str, tuple[str, ...]] = {
    "Web3（Web3 Blogs）": ("https://rss.feedspot.com/folder/5hvFsWUc5w==/rss/rsscombiner",),
    "NFT（NFT Blogs）": ("https://rss.feedspot.com/folder/5hvFsWQd4w==/rss/rsscombiner",),
    "Cryptocurrency _ Crypto Investors（Cryptocurrency Blogs）": (
        "https://rss.feedspot.com/folder/5hvFsWUc5Q==/rss/rsscombiner",
    ),
    "DeFi（DeFi Blogs）": ("https://rss.feedspot.com/folder/5hvItF4h6A==/rss/rsscombiner",),
    "Blockchain & Cryptocurrency Security": (
        "https://rss.feedspot.com/folder/5hvItF4h6Q==/rss/rsscombiner",
    ),
    "Cybersecurity（Cyber Security Blogs）": (
        "https://rss.feedspot.com/folder/5hvItF4h6g==/rss/rsscombiner",
    ),
    "Hacking _ Hacker Blogs": (
        "https://rss.feedspot.com/folder/5hvItF4i4w==/rss/rsscombiner",
        "https://rss.feedspot.com/folder/5hvItF4h7A==/rss/rsscombiner",
    ),
    "Geopolitics（Geopolitics Blogs）": (
        "https://rss.feedspot.com/folder/5hvItF4i5A==/rss/rsscombiner",
    ),
    "Mental Health _ Mind _ Psychology": (
        "https://rss.feedspot.com/folder/5hvItF4i5Q==/rss/rsscombiner",
    ),
    "Dystopian _ Sci-Fi（Dystopian Book Blogs）": (
        "https://rss.feedspot.com/folder/5hvItF4i5g==/rss/rsscombiner",
    ),
}


def safe_directory_name(name: str) -> str:
    """Return a filesystem-safe directory name while retaining readability."""

    cleaned = re.sub(r"[\\/:<>\"|?*]+", "_", name).strip()
    return cleaned or "category"


def sanitize_filename(url: str, max_length: int = 200) -> str:
    """Convert a URL into a safe, bounded-length filename fragment."""

    parsed = re.sub(r"[^a-zA-Z0-9]+", "_", url)
    trimmed = parsed.strip("_") or "resource"

    if len(trimmed) > max_length:
        digest = hashlib.sha256(url.encode("utf-8", "ignore")).hexdigest()[:12]
        available = max_length - len(digest) - 1
        trimmed = f"{trimmed[:available]}_{digest}" if available > 0 else digest

    return trimmed


def normalize_url(url: str) -> str | None:
    """Percent-encode unsafe characters and validate the scheme."""

    trimmed = url.strip()
    if not trimmed:
        return None

    parts = urlsplit(trimmed)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None

    safe_path = quote(parts.path, safe="/:%@+$,;*=()[]'")
    safe_query = urlencode(parse_qsl(parts.query, keep_blank_values=True), doseq=True)

    return urlunsplit((parts.scheme, parts.netloc, safe_path, safe_query, ""))


def fetch_and_save(url: str, destination: Path) -> bytes:
    normalized_url = normalize_url(url)
    if not normalized_url:
        raise ValueError(f"Invalid URL: {url!r}")

    request = urllib.request.Request(
        normalized_url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()

    destination.write_bytes(content)
    return content


def remove_old_files(category_dir: Path, threshold: timedelta) -> None:
    now = datetime.now(timezone.utc)
    for path in category_dir.glob("*"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if now - modified > threshold:
            path.unlink(missing_ok=True)


def extract_links_from_html(base_url: str, html_content: bytes) -> set[str]:
    class LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.links: set[str] = set()

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag.lower() != "a":
                return
            for attr, value in attrs:
                if attr.lower() == "href" and value:
                    joined = urljoin(base_url, value)
                    if joined.startswith(("http://", "https://")):
                        self.links.add(joined)

    parser = LinkParser()
    try:
        parser.feed(html_content.decode("utf-8", errors="ignore"))
    except Exception:
        return set()
    return parser.links


def parse_rss_links(content: bytes) -> list[str]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    links: list[str] = []
    for item in root.findall(".//item"):
        link_el = item.find("link")
        if link_el is not None and link_el.text:
            trimmed = link_el.text.strip()
            if trimmed:
                links.append(trimmed)

    return links


def process_category(category: str, feeds: Iterable[str]) -> None:
    category_dir = BASE_DIR / safe_directory_name(category)
    category_dir.mkdir(parents=True, exist_ok=True)
    remove_old_files(category_dir, OLD_FILE_THRESHOLD)

    for feed_url in feeds:
        feed_name = sanitize_filename(feed_url) + ".xml"
        feed_path = category_dir / feed_name
        try:
            feed_content = fetch_and_save(feed_url, feed_path)
            print(f"Saved feed {feed_url} -> {feed_path}")
        except urllib.error.HTTPError as exc:
            print(f"HTTP error while fetching feed {feed_url}: {exc}", file=sys.stderr)
            continue
        except urllib.error.URLError as exc:
            print(f"Request error while fetching feed {feed_url}: {exc}", file=sys.stderr)
            continue
        except (ValueError, OSError) as exc:
            print(f"Failed to store feed {feed_url}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:  # pragma: no cover - safety net
            print(
                f"Unexpected error while processing feed {feed_url}: {exc}",
                file=sys.stderr,
            )
            continue

        item_links = parse_rss_links(feed_content)
        for item_link in item_links:
            article_name = sanitize_filename(item_link) + ".html"
            article_path = category_dir / article_name
            try:
                article_content = fetch_and_save(item_link, article_path)
                print(f"Saved article {item_link} -> {article_path}")
            except urllib.error.HTTPError as exc:
                print(
                    f"HTTP error while fetching article {item_link}: {exc}",
                    file=sys.stderr,
                )
                continue
            except urllib.error.URLError as exc:
                print(
                    f"Request error while fetching article {item_link}: {exc}",
                    file=sys.stderr,
                )
                continue
            except (ValueError, OSError) as exc:
                print(f"Failed to store article {item_link}: {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # pragma: no cover - safety net
                print(
                    f"Unexpected error while handling article {item_link}: {exc}",
                    file=sys.stderr,
                )
                continue

            for nested_link in extract_links_from_html(item_link, article_content):
                nested_name = sanitize_filename(nested_link) + ".html"
                nested_path = category_dir / nested_name
                if nested_path.exists():
                    continue
                try:
                    fetch_and_save(nested_link, nested_path)
                    print(f"Saved nested link {nested_link} -> {nested_path}")
                except urllib.error.HTTPError as exc:
                    print(
                        f"HTTP error while fetching nested {nested_link}: {exc}",
                        file=sys.stderr,
                    )
                except urllib.error.URLError as exc:
                    print(
                        f"Request error while fetching nested {nested_link}: {exc}",
                        file=sys.stderr,
                    )
                except (ValueError, OSError) as exc:
                    print(
                        f"Skipped nested link {nested_link} due to error: {exc}",
                        file=sys.stderr,
                    )
                except Exception as exc:  # pragma: no cover - safety net
                    print(
                        f"Unexpected error while fetching nested {nested_link}: {exc}",
                        file=sys.stderr,
                    )


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for category, feeds in CATEGORIES.items():
        process_category(category, feeds)


if __name__ == "__main__":
    main()
