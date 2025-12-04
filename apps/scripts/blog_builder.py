"""
高速化版 Persona ブログ生成スクリプト（STATUS_REPORTER 対応版、図解付き）

・/var/www/Meta-Project/indexes/articles からカテゴリを検出
・クライアントの文章を LLM でカテゴリ分類
・該当カテゴリがなければ NO_CATEGORY を返して終了
・カテゴリ配下のインデックスファイルと /indexes/mybrain 配下の知識ファイルを使用
・7 セクション（各 ~1500 文字を目安） + 総論
・各セクションごとに「図解」テキストを生成して HTML に出力
・末尾に タイトル / ディスクリプション(200文字) / タグ6つ(ビッグ2,普通2,スモール2)

制約（読者向け出力）:
- 本文・図解・タイトル・ディスクリプション・タグに
  「インデックス」「クライアント」という語を出さない。
- LLM に「(注: …」「(注2: …」のようなメタ注記を書かせない。
  万一混入しても後処理で削除する。
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import threading
import random

import requests


# ======================= 設定 ================================

BASE_DIR = Path("/var/www/Meta-Project")
INDEX_DIR = BASE_DIR / "indexes"
ARTICLES_DIR = INDEX_DIR / "articles"
MYBRAIN_DIR = INDEX_DIR / "mybrain"
DEFAULT_OUTPUT_DIR = Path("/mnt/hgfs/output")

# Ollama 等の LLM エンドポイント
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "600"))

SECTION_MODEL = os.environ.get("BLOG_BUILDER_SECTION_MODEL", "phi3:mini")
CLASSIFIER_MODEL = os.environ.get("BLOG_BUILDER_CLASSIFIER_MODEL", "phi3:mini")

MIN_SECTION_CHARS = 1500
SUMMARY_TARGET_CHARS = 1500

LLM_CALL_COUNT = 0


LATEST_TOPIC_KEYWORDS = [
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


# ======================= ステータスレポート ===================


class StatusReporter:
    """定期的に現在の処理状況を記録する簡易機構。

    blog_server.py から STATUS_REPORTER.pop_messages(...) が呼ばれる想定。
    """

    def __init__(self, *, interval_seconds: int = 300) -> None:
        self.interval_seconds = interval_seconds
        self._current: dict[str, str] = {"program": "", "function": "", "file": ""}
        self._messages: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, *, program: str, function: str, file: str) -> None:
        with self._lock:
            self._current = {"program": program, "function": function, "file": file}

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval_seconds)
            with self._lock:
                snapshot = {
                    **self._current,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self._messages.append(snapshot)

    def pop_messages(self, *, include_current: bool = False) -> list[dict[str, str]]:
        with self._lock:
            messages = list(self._messages)
            self._messages.clear()
            if include_current and any(self._current.values()):
                messages.append(
                    {
                        **self._current,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            return messages


STATUS_REPORTER = StatusReporter()


@contextlib.contextmanager
def status_scope(function_name: str, *, program: str = "blog_builder", file: str = __file__):
    """generate_blogs 内部などで使用するためのコンテキストマネージャ。"""
    STATUS_REPORTER.update(program=program, function=function_name, file=file)
    try:
        yield
    finally:
        STATUS_REPORTER.update(program=program, function="idle", file=file)


# ======================= LLM ラッパ ==========================


def _post_llm(model: str, prompt: str, *, system: str | None = None) -> str:
    """Ollama / ローカル LLM への単発呼び出し。"""
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[LLM error: {exc}]"


def _is_llm_error(text: str | None) -> bool:
    return not text or text.startswith("[LLM error")


# ======================= テキストユーティリティ ==============


def _clean_text(text: str) -> str:
    """HTML/PDF ノイズや宣伝行をざっくり除去してプレーンテキスト化。"""
    if not text:
        return ""

    cleaned = text.replace("\ufeff", "").replace("\ufffd", "")
    cleaned = re.sub(r"�+", "", cleaned)

    # PDFブロック
    cleaned = re.sub(r"%PDF-\d\.\d[\s\S]*?(?:%%EOF|$)", "", cleaned, flags=re.IGNORECASE)

    # HTML 除去
    cleaned = re.sub(r"(?is)<!DOCTYPE.*?>", " ", cleaned)
    cleaned = re.sub(r"(?is)<head[^>]*>.*?</head>", " ", cleaned)
    cleaned = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", cleaned)
    cleaned = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)

    # ナビ・宣伝っぽい行
    nav_keywords = [
        "skip to main content",
        "navigation",
        "privacy policy",
        "terms and conditions",
        "your account",
        "sign in",
        "sign out",
        "contact us",
    ]
    promo_keywords = [
        "binance",
        "bybit",
        "coinbase",
        "bitfinex",
        "bitflyer",
        "okx",
        "kucoin",
    ]

    filtered: list[str] = []
    for line in cleaned.splitlines():
        ln = line.strip()
        if not ln:
            continue
        low = ln.lower()

        if any(k in low for k in nav_keywords):
            continue
        if any(k in low for k in promo_keywords):
            continue

        filtered.append(ln)

    cleaned = "\n".join(filtered)

    # 文レベルで重複除去（簡易）
    paras = [p.strip() for p in cleaned.split("\n") if p.strip()]
    normalized: list[str] = []
    for para in paras:
        sentences = re.split(r"(?<=[。．.!?])\s+|\s{2,}", para)
        dedup: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if dedup and s == dedup[-1]:
                continue
            dedup.append(s)
        joined = " ".join(dedup)
        if normalized and joined == normalized[-1]:
            continue
        normalized.append(joined)

    cleaned = "\n".join(normalized)
    cleaned = re.sub(r"\s{3,}", "  ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_output_text(text: str) -> str:
    """読者向けに出す本文などを最終整形する。

    - _clean_text でノイズ除去
    - (注: …) (注2: …) のようなメタ注記行を削除
    - 「インデックス」「クライアント」という語を別語に差し替え
    """
    if not text:
        return ""
    text = _clean_text(text)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(注:") or stripped.startswith("（注:"):
            continue
        if stripped.startswith("(注2:") or stripped.startswith("（注2:"):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # 読者には意味が伝わりづらい語を別表現に
    text = text.replace("インデックス", "データソース")
    text = text.replace("クライアント", "読者")

    return text.strip()


def _safe_json_parse(text: str) -> dict[str, Any]:
    """Extract the first JSON object from ``text`` and return it as a dict."""
    if not text:
        return {}

    # よくあるラッパー (```json ... ``` など) を剥がす
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^[^\{\[]*", "", cleaned)  # 先頭の説明文をざっくり削る

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _parse_loose_body_json(text: str) -> dict[str, Any]:
    """
    3段階ブログ生成の Body 用に、ゆるい JSON っぽい文字列から
    body / diagram をできるだけ取り出すためのフォールバックパーサ。

    想定している入力:
    {
      "body": "...(ここに長文。改行や <br> などがそのまま入っていることもある)...",
      "diagram": [
        { "title": "...", "content": "..." }
      ]
    }

    - strict な json.loads に失敗したときだけ使う
    - ダメなら {} を返す
    """
    if not text:
        return {}

    # まずはコードフェンス系を削ぐ
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # "body": " .... " ～ "diagram" の手前までを抜き出す
    m_body = re.search(
        r'"body"\s*:\s*"(.*?)"\s*,\s*\n\s*"diagram"',
        cleaned,
        re.S,
    )
    if not m_body:
        # "diagram" がない / フォーマットが違う場合もあるので、
        # "body" だけでも取れれば使う
        m_body_only = re.search(r'"body"\s*:\s*"(.*?)"', cleaned, re.S)
        if not m_body_only:
            return {}
        raw_body = m_body_only.group(1)
    else:
        raw_body = m_body.group(1)

    # JSON 由来のエスケープをある程度戻す
    body_str = raw_body.replace('\\"', '"')
    body_str = body_str.replace("\\n", "\n")

    # diagram 部分を雑に抜いてみる（取れればラッキー）
    m_diag = re.search(r'"diagram"\s*:\s*(\[[\s\S]*\])', cleaned)
    diagram_val: Any = ""
    if m_diag:
        diag_raw = m_diag.group(1).strip()
        # まずはちゃんとした JSON として読んでみる
        try:
            diagram_val = json.loads(diag_raw)
        except Exception:
            # 無理なら文字列としてそのまま使う
            diagram_val = diag_raw

    return {"body": body_str, "diagram": diagram_val}


def _parse_loose_body_json(text: str) -> dict[str, Any]:
    """
    3段階ブログ生成の Body 用に、ゆるい JSON っぽい文字列から
    body / diagram をできるだけ取り出すためのフォールバックパーサ。

    想定している入力:
    {
      "body": "...(ここに長文。改行や <br> などがそのまま入っていることもある)...",
      "diagram": [
        { "title": "...", "content": "..." }
      ]
    }

    - strict な json.loads に失敗したときだけ使う
    - ダメなら {} を返す
    """
    if not text:
        return {}

    # まずはコードフェンス系を削ぐ
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # "body": " .... " ～ "diagram" の手前までを抜き出す
    m_body = re.search(
        r'"body"\s*:\s*"(.*?)"\s*,\s*\n\s*"diagram"',
        cleaned,
        re.S,
    )
    if not m_body:
        # "diagram" がない / フォーマットが違う場合もあるので、
        # "body" だけでも取れれば使う
        m_body_only = re.search(r'"body"\s*:\s*"(.*?)"', cleaned, re.S)
        if not m_body_only:
            return {}
        raw_body = m_body_only.group(1)
    else:
        raw_body = m_body.group(1)

    # JSON 由来のエスケープをある程度戻す
    body_str = raw_body.replace('\\"', '"')
    body_str = body_str.replace("\\n", "\n")

    # diagram 部分を雑に抜いてみる（取れればラッキー）
    m_diag = re.search(r'"diagram"\s*:\s*(\[[\s\S]*\])', cleaned)
    diagram_val: Any = ""
    if m_diag:
        diag_raw = m_diag.group(1).strip()
        # まずはちゃんとした JSON として読んでみる
        try:
            diagram_val = json.loads(diag_raw)
        except Exception:
            # 無理なら文字列としてそのまま使う
            diagram_val = diag_raw

    return {"body": body_str, "diagram": diagram_val}


def _normalize_token(token: str) -> str:
    return re.sub(r"[\s_\-・、,。()（）\[\]]+", "", token).casefold()


def is_latest_topic_request(prompt: str) -> bool:
    normalized = _normalize_token(prompt)
    return (
        ("最新" in normalized and "トピック" in normalized)
        or "latesttopic" in normalized
        or "picklatest" in normalized
    )


def pick_latest_topic_index(articles_dir: Path, keywords: list[str]) -> tuple[str | None, Path | None]:
    """キーワードに紐づくインデックスファイルをランダムに1つ選ぶ。"""

    if not articles_dir.exists():
        return None, None

    keyword_pool = list(keywords)
    random.shuffle(keyword_pool)

    def iter_candidate_files() -> list[Path]:
        return [
            p
            for p in articles_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".html", ".htm"}
        ]

    haystacks: dict[str, list[Path]] = {kw: [] for kw in keyword_pool}
    for path in iter_candidate_files():
        normalized_path = _normalize_token(str(path.relative_to(articles_dir)))
        for kw in keyword_pool:
            if not kw:
                continue
            if _normalize_token(kw) in normalized_path:
                haystacks.setdefault(kw, []).append(path)

    for kw in keyword_pool:
        if haystacks.get(kw):
            return kw, random.choice(haystacks[kw])

    # キーワード一致がない場合でも、インデックスファイルから1つ選んで返す
    candidates = iter_candidate_files()
    if candidates:
        return None, random.choice(candidates)
    return None, None


def build_latest_topic_prompt(keyword: str | None, source_path: Path, source_text: str) -> str:
    keyword_line = keyword or "(no keyword match)"
    return textwrap.dedent(
        f"""
        You are a concise, news-savvy English blog writer. Using the source text below, prepare an output for the client when they ask for "the latest topic." All answers must be in English.

        - Selected keyword: {keyword_line}
        - Source path: {source_path}
        - Keyword catalog for reference: {', '.join(LATEST_TOPIC_KEYWORDS)}

        First, read the entire source and write an English summary of about 1000 characters. Then, based on that summary, produce persona-resonating items:
        1. Title (within 50 characters)
        2. Description (about 200 characters)
        3. Tags (one line: two BIG words, two NORMAL words, two SMALL words, separated by half-width commas)
        4. The ~1000-character summary
        5. Seven persona-resonating headlines

        Output format (keep this order and use English only):
        Title: <title>
        Description: <description>
        Tags: <BIG,BIG,NORMAL,NORMAL,SMALL,SMALL>
        Summary: <~1000-character summary>
        Headlines:
        1. <headline 1>
        2. <headline 2>
        3. <headline 3>
        4. <headline 4>
        5. <headline 5>
        6. <headline 6>
        7. <headline 7>

        --- BEGIN SOURCE ---
        {source_text}
        --- END SOURCE ---
        """
    ).strip()


def read_snippet(path: Path, *, max_chars: int | None = None) -> str:
    """ディレクトリ/ファイル配下のテキストを結合して返す（高速版）。

    max_chars を指定すると、その文字数を超えたところで打ち切る。
    """
    if not path.exists():
        return ""

    texts: list[str] = []

    if path.is_file():
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        cleaned = _clean_text(raw)
        if max_chars is not None:
            return cleaned[:max_chars]
        return cleaned

    # ディレクトリの場合：深い再帰はせず、rglob で拾いつつ早めに打ち切り
    total_len = 0
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".html", ".htm"}:
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            cleaned = _clean_text(raw)
            texts.append(cleaned)
            total_len += len(cleaned)
            if max_chars is not None and total_len >= max_chars:
                break

    merged = "\n\n".join(texts)
    if max_chars is not None:
        return merged[:max_chars]
    return merged


def discover_article_taxonomy(articles_dir: Path) -> Mapping[str, list[Path]]:
    """カテゴリ名 -> パス一覧 のマッピングを構築する。"""
    taxonomy: dict[str, list[Path]] = {}
    if not articles_dir.exists():
        return taxonomy

    for category_path in sorted(articles_dir.iterdir()):
        if category_path.is_dir():
            taxonomy[category_path.name] = [category_path]
        elif category_path.is_file():
            taxonomy.setdefault(category_path.stem, []).append(category_path)
    return taxonomy


def classify_category_with_llm(prompt: str, categories: list[str]) -> str | None:
    """LLM によるカテゴリ分類。カテゴリ名のみを返す。"""
    if not categories:
        return None

    category_list = "\n".join(f"- {c}" for c in categories)
    system = (
        "あなたは分類器です。読者からの日本語テキストを、"
        "与えられたカテゴリ一覧の中から最も関連する1つに分類します。"
        "出力はカテゴリ名のみ、日本語の説明や装飾は禁止です。"
    )
    body = textwrap.dedent(
        f"""
        読者からの文章:
        {prompt}

        利用可能なカテゴリ:
        {category_list}

        一番関連度が高いと思うカテゴリ名だけを、そのまま1行で出力してください。
        見つからない場合は「NONE」とだけ出力してください。
        """
    )

    answer = _post_llm(CLASSIFIER_MODEL, body, system=system)
    if _is_llm_error(answer):
        return None

    ans = answer.strip()
    if ans.upper() == "NONE":
        return None

    # 完全一致 or 大文字小文字無視でマッチ
    for c in categories:
        if ans == c or ans.lower() == c.lower():
            return c
    for c in categories:
        if c.lower() in ans.lower():
            return c
    return None


def choose_category_fallback(prompt: str, categories: list[str]) -> str | None:
    """LLM で決まらなかった場合のフォールバック：キーワード重みづけ。"""
    if not categories:
        return None
    normalized = prompt.lower()

    scores: dict[str, int] = {}
    for cat in categories:
        score = 0
        simple = re.sub(r"[（(].*?[）)]", "", cat).strip()
        if simple.lower() in normalized:
            score += 3
        for token in re.split(r"[^\w]+", simple.lower()):
            if token and token in normalized:
                score += 1
        scores[cat] = score

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


def _extract_relevant_snippet(
    cleaned_text: str,
    *,
    keywords: list[str],
    max_chars: int = 1200,
) -> str:
    """クリーン済みテキストからキーワードに関連しそうな文をスコアリングして抽出。"""
    if not cleaned_text:
        return ""

    units: list[str] = []
    for block in cleaned_text.splitlines():
        block = block.strip()
        if not block:
            continue
        for sent in re.split(r"(?<=[。．.!?])\s+|\s{2,}", block):
            sent = sent.strip()
            if sent:
                units.append(sent)
    if not units:
        return ""

    keys = [k.lower() for k in keywords if k]

    scored: list[tuple[int, int, str]] = []
    for i, s in enumerate(units):
        low = s.lower()
        score = sum(2 for k in keys if k in low)
        scored.append((score, i, s))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected: list[str] = []
    total = 0
    for score, _, s in scored:
        if score == 0 and selected:
            break
        ln = len(s)
        if total + ln > max_chars:
            break
        selected.append(s)
        total += ln + 1

    return "\n".join(selected)


@dataclass
class Section:
    titles: dict[str, str]
    bodies: dict[str, str]
    diagrams: dict[str, str]


# ======================= セクション構成 =======================


def plan_sections(prompt: str, category: str, index_excerpt: str, knowledge_excerpt: str) -> list[dict[str, str]]:
    """7つのセクション構成（タイトル＋フォーカス）を LLM で決める。"""
    system = (
        "あなたは日本語の長文ブログ編集者です。"
        "読者からの依頼文とデータソース/知識ファイルをもとに、"
        "7つのセクション構成（タイトルとフォーカス）を設計してください。"
    )
    body = textwrap.dedent(
        f"""
        読者からの依頼文:
        {prompt}

        選択されたカテゴリ: {category}

        データソース抜粋:
        {index_excerpt}

        mybrain 抜粋:
        {knowledge_excerpt}

        出力フォーマット（JSONのみ・日本語）:
        {{
          "sections": [
            {{"title": "セクション1のタイトル", "focus": "このセクションで掘り下げる観点"}},
            ...
            （合計7件）
          ]
        }}

        条件:
        - セクションは7つちょうど作る。
        - タイトルは簡潔で、日本語で15〜30文字程度。
        - focusは、そのセクションで何を論じるかを1〜2文で説明する。
        - Punk Rock / NFT / Web3 など、読者のテーマからずれない。
        - 同じ意味のセクションを重複させない。
        - セクションタイトルやfocusに「インデックス」「クライアント」という語を含めない。
        - JSON以外の文字は出力しない。
        """
    )

    answer = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(answer):
        fallback_titles = [
            "序章: テーマと違和感",
            "歴史と文脈",
            "技術と仕組み",
            "市場とお金の流れ",
            "文化・コミュニティ",
            "リスクと倫理",
            "未来への問い",
        ]
        return [{"title": t, "focus": ""} for t in fallback_titles]

    try:
        data = json.loads(answer)
        sections = data.get("sections", [])
        if not isinstance(sections, list) or len(sections) != 7:
            raise ValueError("sections length mismatch")
        cleaned: list[dict[str, str]] = []
        for sec in sections:
            title = str(sec.get("title", "")).strip()
            focus = str(sec.get("focus", "")).strip()
            if not title:
                continue
            cleaned.append({"title": title, "focus": focus})
        if len(cleaned) != 7:
            raise ValueError("sections cleaned length mismatch")
        return cleaned
    except Exception:
        fallback_titles = [
            "序章: テーマと違和感",
            "歴史と文脈",
            "技術と仕組み",
            "市場とお金の流れ",
            "文化・コミュニティ",
            "リスクと倫理",
            "未来への問い",
        ]
        return [{"title": t, "focus": ""} for t in fallback_titles]


def build_section_body(
    prompt: str,
    category: str,
    section_title: str,
    focus: str,
    cleaned_index_text: str,
    cleaned_knowledge_text: str,
) -> str:
    """各セクション本文を LLM で生成（フォールバック用・本文のみ）。"""
    keywords = []
    for token in re.split(r"[^\w]+", f"{prompt} {category} {section_title} {focus}"):
        token = token.strip()
        if len(token) >= 3:
            keywords.append(token.lower())

    index_snippet = _extract_relevant_snippet(cleaned_index_text, keywords=keywords, max_chars=600)
    knowledge_snippet = _extract_relevant_snippet(cleaned_knowledge_text, keywords=keywords, max_chars=600)

    system = (
        "あなたは日本語の長文ブログライターです。"
        "データソースは客観的な最新情報、mybrainはヨハネの知識・解釈として扱い、"
        "両方を統合して自然な文章を作成してください。"
        "英語やスペイン語など他言語の翻訳併記、特定サービスの宣伝文句、"
        "同じ文章の繰り返しは禁止です。"
    )

    body = textwrap.dedent(
        f"""
        読者からの依頼文:
        {prompt}

        カテゴリ: {category}
        セクションタイトル: {section_title}
        セクションの焦点: {focus}

        データソースから関連しそうな情報（客観情報）:
        {index_snippet or "（利用可能な情報なし）"}

        mybrain から関連しそうな情報（ヨハネの知識・解釈）:
        {knowledge_snippet or "（利用可能な情報なし）"}

        要件:
        - 上記の情報を参考に、このセクション専用の本文を日本語で書く。
        - 「データソース」「mybrain」「インデックス」「クライアント」といった語は本文に出さない。
        - 全体でおよそ {MIN_SECTION_CHARS} 文字以上になるようにする。
        - 特定の取引所やサービス（Binanceなど）を褒めちぎる宣伝文句は禁止。
        - Español など他言語の単語は出さない。
        - 同じ文をコピペしたような繰り返しは禁止。
        - 「(注: …」「(注2: …」のようなメタ注記や、この文章が自動生成であることの説明は書かない。
        - 哲学者気取りのヨハネが論理展開しながら、静かに読者に語りかけるトーンにする。
        """
    )

    text = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(text):
        # LLM 失敗時はスニペットを返す
        fallback = (index_snippet + "\n\n" + knowledge_snippet).strip()
        return _sanitize_output_text(fallback or section_title)

    return _sanitize_output_text(text)


def infer_section_model(
    prompt: str,
    category: str,
    section_title: str,
    focus: str,
    index_snippet: str,
    knowledge_snippet: str,
) -> dict[str, str]:
    """セクションの概要をLLMに投げてモデリング方針を推論させる。"""

    system = (
        "あなたは事実確認を重視する編集者です。"
        "与えられた概要と素材から、セクションの論旨・展開・必要な証拠のモデリングを行います。"
    )

    body = textwrap.dedent(
        f"""
        読者からの依頼文:
        {prompt}

        カテゴリ: {category}
        セクションタイトル: {section_title}
        セクションの焦点: {focus}

        データソースから関連しそうな情報（客観情報）:
        {index_snippet or "（利用可能な情報なし）"}

        mybrain から関連しそうな情報（ヨハネの知識・解釈）:
        {knowledge_snippet or "（利用可能な情報なし）"}

        出力フォーマット（JSON のみ）:
        {{
          "model": {{
            "thesis": "中心となる主張",  
            "angles": ["掘り下げる観点A", "掘り下げる観点B"],
            "evidence": "根拠として触れる事実やデータの種類",
            "risk_check": "誇張や不確実な点を避けるための確認事項",
            "arc": "導入から結論までのストーリーライン"
          }}
        }}

        条件:
        - 証拠や年代は確証があるものだけを書く。不明なら「未確認」と明示し、それ以上踏み込まない。
        - Berkshireが1977年にNFT理論を提案した、など確証のない話は書かない。
        - JSON以外の文字は出力しない。
        """
    )

    answer = _post_llm(SECTION_MODEL, body, system=system)
    fallback = {
        "thesis": focus or section_title,
        "angles": ["背景", "現在地", "影響"],
        "evidence": "最新の公開情報と一次データを確認",
        "risk_check": "年代や出典が曖昧なら断定しない",
        "arc": "問いかけから問題提起、洞察、読者への提案までの流れ",
    }

    if _is_llm_error(answer):
        return fallback

    try:
        data = json.loads(answer)
        model = data.get("model") or {}
        cleaned: dict[str, str] = {}
        cleaned["thesis"] = _sanitize_output_text(str(model.get("thesis", "") or fallback["thesis"]))
        cleaned["angles"] = ", ".join(
            _sanitize_output_text(str(x)) for x in model.get("angles", []) if str(x).strip()
        ) or ", ".join(fallback["angles"])
        cleaned["evidence"] = _sanitize_output_text(str(model.get("evidence", "") or fallback["evidence"]))
        cleaned["risk_check"] = _sanitize_output_text(str(model.get("risk_check", "") or fallback["risk_check"]))
        cleaned["arc"] = _sanitize_output_text(str(model.get("arc", "") or fallback["arc"]))
        return cleaned
    except Exception:
        return fallback


def _format_modeling_summary(model: Mapping[str, str]) -> str:
    return textwrap.dedent(
        f"""
        - Thesis: {model.get('thesis', '')}
        - Angles: {model.get('angles', '')}
        - Evidence plan: {model.get('evidence', '')}
        - Risk/accuracy check: {model.get('risk_check', '')}
        - Narrative arc: {model.get('arc', '')}
        """
    ).strip()


def generate_english_section_body(
    prompt: str,
    category: str,
    section_title: str,
    focus: str,
    modeling: Mapping[str, str],
    index_snippet: str,
    knowledge_snippet: str,
) -> str:
    """英語本文を先に生成し、途切れなく書き切る。"""

    system = (
        "You are a long-form blog writer who values factual accuracy and complete narratives."
        " Finish every thought even if it takes extra length, and avoid unverifiable claims."
    )

    modeling_summary = _format_modeling_summary(modeling)
    body = textwrap.dedent(
        f"""
        Reader request:
        {prompt}

        Category: {category}
        Section title: {section_title}
        Focus: {focus}

        Modeling plan inferred earlier:
        {modeling_summary}

        Evidence snippets (objective data):
        {index_snippet or "(no direct evidence available)"}

        Background knowledge (Johanne's interpretation):
        {knowledge_snippet or "(no additional notes)"}

        Write the full section body in English.

        Requirements:
        - Honor the modeling plan and weave the angles into a single, cohesive narrative arc.
        - Write at least {MIN_SECTION_CHARS} characters and keep sentences complete. Do not truncate or leave paragraphs hanging.
        - If a fact is uncertain, mark it as uncertain and move on without speculation. Never invent claims such as "Berkshire proposed an NFT theory in 1977".
        - Do not mention the words "データソース", "mybrain", "インデックス", or "クライアント".
        - Keep a calm tone, thoughtful and reflective.
        - Do not include meta-commentary like "(note: ...)".
        - Output only the final English body text.
        """
    )

    text = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(text):
        fallback = build_section_body(
            prompt,
            category,
            section_title,
            focus,
            index_snippet,
            knowledge_snippet,
        )
        return fallback

    return _sanitize_output_text(text)


def translate_body(
    english_body: str,
    *,
    target_lang: str,
    section_title: str,
    focus: str,
    modeling: Mapping[str, str],
) -> str:
    """英語本文を他言語へ翻訳しつつ、内容の完全性を保つ。"""

    system = (
        "You are a precise translator. Preserve meaning and keep paragraphs complete."
        " Avoid inventing facts and keep the calm tone."
    )

    body = textwrap.dedent(
        f"""
        Translate the following English section into {target_lang}.

        Section title: {section_title}
        Focus: {focus}
        Modeling reminder: {_format_modeling_summary(modeling)}

        English body:
        {english_body}

        Requirements:
        - Keep the narrative whole; do not shorten or omit parts even if it becomes long.
        - Avoid speculative or unverified claims; if something was marked uncertain, keep that qualifier.
        - Do not add meta notes or mention data sources.
        - Output only the translated body in {target_lang}.
        """
    )

    translated = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(translated):
        return english_body

    return _sanitize_output_text(translated)


def localize_section_titles(
    section_title: str, modeling: Mapping[str, str]
) -> dict[str, str]:
    """セクション見出しを英語・イタリア語へ自然に翻訳する。"""

    system = (
        "You are a bilingual editor. Propose concise, natural titles in English and Italian."
        " Keep the nuance of the source title and modeling plan."
    )

    body = textwrap.dedent(
        f"""
        元のセクションタイトル: {section_title}
        モデリング概要: {_format_modeling_summary(modeling)}

        出力フォーマット（JSONのみ）:
        {{"titles": {{"en": "English title", "it": "Titolo in italiano"}}}}

        条件:
        - それぞれ15〜30文字程度で簡潔に。
        - 不確かな固有名詞や年代は避け、事実に基づく表現だけを使う。
        - JSON以外は出力しない。
        """
    )

    answer = _post_llm(SECTION_MODEL, body, system=system)
    titles = {"ja": section_title, "en": section_title, "it": section_title}
    if _is_llm_error(answer):
        return titles

    try:
        data = json.loads(answer)
        raw = data.get("titles") or {}
        titles["en"] = _sanitize_output_text(str(raw.get("en", section_title)) or section_title)
        titles["it"] = _sanitize_output_text(str(raw.get("it", section_title)) or section_title)
        return titles
    except Exception:
        return titles


def _extract_chart_keywords(text: str, *, limit: int = 6) -> list[str]:
    tokens = re.findall(r"[\w\-]{3,}", text.lower())
    freq: dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [t for t, _ in sorted_tokens[:limit]] or ["insight", "trend", "impact"]


def _choose_chart_type(text: str) -> str:
    lowered = text.lower()
    mapping = {
        "line": ["推移", "trend", "growth", "増加", "減少", "timeline"],
        "pie": ["割合", "比率", "share", "構成", "distribution"],
        "bar": ["比較", "ranking", "対比", "versus", "survey"],
        "scatter": ["相関", "関係", "関連", "correlation", "relationship"],
        "table": ["手順", "ステップ", "process", "checklist", "項目"],
    }
    for chart, keywords in mapping.items():
        if any(k in lowered for k in keywords):
            return chart
    return "bar"


def _build_chart_dataset(keywords: list[str]) -> list[tuple[str, int]]:
    dataset: list[tuple[str, int]] = []
    for i, key in enumerate(keywords):
        value = 50 + ((hash(key) + i * 13) % 50)
        dataset.append((key.title(), value))
    return dataset or [("Idea", 60), ("Context", 55), ("Action", 70)]


def _translate_label(label: str, lang: str) -> str:
    translations = {
        "ja": {
            "Visualization": "図解",
            "Key Points": "要点",
            "Data": "データ",
            "Value": "値",
        },
        "en": {
            "Visualization": "Visualization",
            "Key Points": "Key Points",
            "Data": "Data",
            "Value": "Value",
        },
        "it": {
            "Visualization": "Visualizzazione",
            "Key Points": "Punti chiave",
            "Data": "Dati",
            "Value": "Valore",
        },
    }
    return translations.get(lang, translations["en"]).get(label, label)


def _render_chart_html(chart_type: str, dataset: list[tuple[str, int]], section_title: str, lang: str) -> str:
    safe_title = html.escape(section_title)
    label_data = _translate_label("Data", lang)
    label_value = _translate_label("Value", lang)
    intro = {
        "ja": f"{safe_title}の{_translate_label('Visualization', 'ja')} ({chart_type})",
        "en": f"{safe_title} {_translate_label('Visualization', 'en')} ({chart_type})",
        "it": f"{safe_title} {_translate_label('Visualization', 'it')} ({chart_type})",
    }
    if chart_type == "table":
        rows = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>" for label, value in dataset
        )
        return textwrap.dedent(
            f"""
            <div class="diagram rich-diagram table">
              <h3>{intro.get(lang, intro['en'])}</h3>
              <table>
                <thead><tr><th>{label_data}</th><th>{label_value}</th></tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>
            """
        ).strip()

    if chart_type == "pie":
        total = max(sum(v for _, v in dataset), 1)
        segments = "".join(
            f"<li><span style='--share:{value / total:.4f}'>{html.escape(label)}</span></li>"
            for label, value in dataset
        )
        return textwrap.dedent(
            f"""
            <div class="diagram rich-diagram pie">
              <h3>{intro.get(lang, intro['en'])}</h3>
              <ul class="pie-chart">{segments}</ul>
            </div>
            """
        ).strip()

    if chart_type == "scatter":
        points = "".join(
            f"<div class='point' style='--x:{(i+1)*12};--y:{value/2};' title='{html.escape(label)}: {value}'></div>"
            for i, (label, value) in enumerate(dataset)
        )
        return textwrap.dedent(
            f"""
            <div class="diagram rich-diagram scatter">
              <h3>{intro.get(lang, intro['en'])}</h3>
              <div class="scatter-plot" aria-label="{label_data} vs {label_value}">{points}</div>
            </div>
            """
        ).strip()

    if chart_type == "line":
        points = "".join(
            f"<span style='--pos:{i};--val:{value};' title='{html.escape(label)}: {value}'></span>"
            for i, (label, value) in enumerate(dataset)
        )
        return textwrap.dedent(
            f"""
            <div class="diagram rich-diagram line">
              <h3>{intro.get(lang, intro['en'])}</h3>
              <div class="line-chart">{points}</div>
            </div>
            """
        ).strip()

    # default to bar
    bars = "".join(
        f"<div class='bar' style='--height:{value};'><span class='label'>{html.escape(label)}</span><span class='value'>{value}</span></div>"
        for label, value in dataset
    )
    return textwrap.dedent(
        f"""
        <div class="diagram rich-diagram bar">
          <h3>{intro.get(lang, intro['en'])}</h3>
          <div class="bar-chart">{bars}</div>
        </div>
        """
    ).strip()


def build_multilingual_diagram(section_title: str, bodies: Mapping[str, str]) -> dict[str, str]:
    base_text = bodies.get("ja") or " ".join(bodies.values())
    chart_type = _choose_chart_type(base_text)
    keywords = _extract_chart_keywords(base_text)
    dataset = _build_chart_dataset(keywords)
    diagrams: dict[str, str] = {}
    for lang in ("ja", "en", "it"):
        diagrams[lang] = _render_chart_html(chart_type, dataset, section_title, lang)
    return diagrams


def build_section(
    prompt: str,
    category: str,
    section_title: str,
    focus: str,
    cleaned_index_text: str,
    cleaned_knowledge_text: str,
) -> Section:
    """各セクションを英語で書き切り、その後に翻訳する。"""

    keywords = []
    for token in re.split(r"[^\w]+", f"{prompt} {category} {section_title} {focus}"):
        token = token.strip()
        if len(token) >= 3:
            keywords.append(token.lower())

    index_snippet = _extract_relevant_snippet(cleaned_index_text, keywords=keywords, max_chars=600)
    knowledge_snippet = _extract_relevant_snippet(cleaned_knowledge_text, keywords=keywords, max_chars=600)

    modeling = infer_section_model(prompt, category, section_title, focus, index_snippet, knowledge_snippet)

    english_body = generate_english_section_body(
        prompt,
        category,
        section_title,
        focus,
        modeling,
        index_snippet,
        knowledge_snippet,
    )

    ja_body = translate_body(
        english_body,
        target_lang="Japanese",
        section_title=section_title,
        focus=focus,
        modeling=modeling,
    )
    it_body = translate_body(
        english_body,
        target_lang="Italian",
        section_title=section_title,
        focus=focus,
        modeling=modeling,
    )

    titles = localize_section_titles(section_title, modeling)
    bodies = {"en": english_body, "ja": ja_body, "it": it_body}
    diagrams = build_multilingual_diagram(section_title, bodies)
    return Section(titles=titles, bodies=bodies, diagrams=diagrams)


def build_conclusion(
    prompt: str,
    category: str,
    sections: list[Section],
    cleaned_knowledge_text: str,
) -> dict[str, str]:
    """7セクションを踏まえた総論を英語でまとめ、後から翻訳する。"""

    body_excerpt = "\n\n".join(
        f"[{i+1}] {sec.titles.get('en', sec.titles.get('ja', ''))}\n{sec.bodies.get('en', '')[:240]}"  # noqa: E501
        for i, sec in enumerate(sections)
    )
    body_excerpt = body_excerpt[:1500]
    knowledge_excerpt = _extract_relevant_snippet(cleaned_knowledge_text, keywords=[category], max_chars=500)

    modeling = {
        "thesis": f"Overall perspective on {category}",
        "angles": "weave insights from all seven sections",
        "evidence": "only verifiable timelines and public facts",
        "risk_check": "avoid speculative claims; mark uncertainties",
        "arc": "from common thread to implications and a calm closing",
    }

    system = (
        "You are a multilingual essayist. Write the conclusion in English first, then it will be translated."
        " Keep the discussion holistic rather than listing section by section, and finish every paragraph."
    )

    body = textwrap.dedent(
        f"""
        Reader request:
        {prompt}

        Category: {category}

        Insights gathered across sections (English excerpts):
        {body_excerpt}

        Background context from knowledge files:
        {knowledge_excerpt or "(no extra knowledge available)"}

        Requirements:
        - Discuss the article as a whole; do not summarize section by section.
        - Write around {SUMMARY_TARGET_CHARS} characters or more if needed to finish the thoughts—never leave ideas unfinished.
        - Prefer facts that can be traced; if unsure, label them as uncertain and move on without speculation. Avoid fabrications such as "Berkshire proposed an NFT theory in 1977".
        - Maintain a calm tone and avoid promotional language.
        - Do not mention data sources or clients.
        - Output only the final English conclusion text.
        """
    )

    fallback = _sanitize_output_text(body_excerpt or prompt)
    english = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(english):
        english = fallback[:SUMMARY_TARGET_CHARS].rstrip() + "…"
    else:
        english = _sanitize_output_text(english)

    ja = translate_body(
        english,
        target_lang="Japanese",
        section_title="総論",
        focus="記事全体のまとめ",
        modeling=modeling,
    )
    it = translate_body(
        english,
        target_lang="Italian",
        section_title="Conclusione",
        focus="Sintesi complessiva",
        modeling=modeling,
    )

    return {"en": english, "ja": ja, "it": it}


def build_meta(
    prompt: str,
    category: str,
    sections: list[Section],
    conclusion: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """各言語のタイトル・ディスクリプション・タグ6個を1回で生成。"""

    def _sections_excerpt(lang: str) -> str:
        return "\n\n".join(f"[{i+1}] {s.titles.get(lang, s.titles.get('ja', ''))}" for i, s in enumerate(sections))

    system = (
        "あなたは多言語のブログ編集者です。"
        "記事本文と総論から内容を推測し、タイトル・ディスクリプション・タグ6つを日本語・英語・イタリア語で設計してください。"
    )

    body = textwrap.dedent(
        f"""
        読者からの依頼文:
        {prompt}

        カテゴリ: {category}

        セクション一覧（日本語ベース）:
        {_sections_excerpt('ja')}

        総論抜粋:
        {conclusion.get('ja', '')[:600]}

        出力フォーマット（JSONのみ）:
        {{
          "meta": {{
            "ja": {{"title": "タイトル", "description": "200文字の説明", "tags": {{"big": [], "normal": [], "small": []}}}},
            "en": {{"title": "Title in English", "description": "~200 chars", "tags": {{"big": [], "normal": [], "small": []}}}},
            "it": {{"title": "Titolo in italiano", "description": "~200 caratteri", "tags": {{"big": [], "normal": [], "small": []}}}}
          }}
        }}

        条件:
        - 各言語で同じ内容・ニュアンスを保つ。
        - タイトル・ディスクリプション・タグに「インデックス」「クライアント」という語を含めない。
        - ディスクリプションは200文字を目安にし、210文字以内。
        - タグは合計6個（big2, normal2, small2）。
        - 特定サービスの宣伝ワードは禁止。中立的なキーワードにする。
        - 確認できない年代・主張（例: Berkshireが1977年にNFT理論を提案）は書かない。事実ベースで構成する。
        - 「(注: …」などのメタ注記や、この文章が自動生成であることの説明は書かない。
        - JSON以外は出力しない。タイトルやタグはLLMの推論結果を必ず返す。
        """
    )

    answer = _post_llm(SECTION_MODEL, body, system=system)
    base_title = (prompt.strip() or category or "ブログ").replace("\n", " ")
    if len(base_title) > 32:
        base_title = base_title[:32].rstrip() + "…"
    base_desc = prompt.strip().replace("\n", " ") or "読者から与えられたテーマをもとにしたブログ記事。"
    if len(base_desc) > 200:
        base_desc = base_desc[:200].rstrip() + "…"

    fallback_meta = {
        lang: {
            "title": _sanitize_output_text(base_title),
            "description": _sanitize_output_text(base_desc),
            "tags": "PunkRock,NFT",
        }
        for lang in ("ja", "en", "it")
    }

    if _is_llm_error(answer):
        return fallback_meta

    try:
        data = json.loads(answer)
        metas = data.get("meta") or {}
        result: dict[str, dict[str, str]] = {}
        for lang in ("ja", "en", "it"):
            meta = metas.get(lang) or {}
            title = _sanitize_output_text(str(meta.get("title", "") or base_title))
            description = str(meta.get("description", "") or base_desc)
            if len(description) > 200:
                description = description[:200].rstrip() + "…"
            description = _sanitize_output_text(description)

            tags = meta.get("tags") or {}
            big = [str(x).strip() for x in tags.get("big", []) if str(x).strip()]
            normal = [str(x).strip() for x in tags.get("normal", []) if str(x).strip()]
            small = [str(x).strip() for x in tags.get("small", []) if str(x).strip()]
            all_tags = big[:2] + normal[:2] + small[:2]
            tags_line = ",".join(_sanitize_output_text(tag) for tag in all_tags) or fallback_meta[lang]["tags"]

            result[lang] = {"title": title, "description": description, "tags": tags_line}
        return result
    except Exception:
        return fallback_meta


def _format_html_text(text: str) -> str:
    escaped = html.escape(text)
    return escaped.replace("\n", "<br />\n")


def compose_html(
    lang: str,
    meta: Mapping[str, str],
    sections: list[Section],
    conclusion: Mapping[str, str],
) -> str:
    """指定言語の最終 HTML を組み立てる。"""
    title = meta["title"]
    description = meta["description"]
    tags_line = meta["tags"]

    labels = {
        "ja": {"chapter": "第{num}章", "summary": "総論", "tags": "タグ", "description": "ディスクリプション"},
        "en": {"chapter": "Chapter {num}", "summary": "Conclusion", "tags": "Tags", "description": "Description"},
        "it": {"chapter": "Capitolo {num}", "summary": "Conclusione", "tags": "Tag", "description": "Descrizione"},
    }
    lbl = labels.get(lang, labels["en"])

    style_block = textwrap.dedent(
        """
        <style>
        body { font-family: sans-serif; line-height: 1.6; max-width: 960px; margin: 0 auto; padding: 1rem; }
        article section { margin-bottom: 2rem; }
        .description { color: #444; }
        .rich-diagram { background: #f7f7fb; border: 1px solid #e0e0ef; padding: 1rem; margin-top: 0.5rem; border-radius: 8px; }
        .bar-chart { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; }
        .bar-chart .bar { background: linear-gradient(180deg, #6c8efb, #4a64e4); color: #fff; padding: 0.5rem; border-radius: 6px; min-height: 80px; position: relative; }
        .bar-chart .bar::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: calc(var(--height,40) * 0.8%); background: rgba(255,255,255,0.2); border-radius: 0 0 6px 6px; }
        .bar-chart .label { display: block; font-weight: bold; }
        .bar-chart .value { font-size: 0.9rem; }
        .line-chart { display: flex; gap: 0.5rem; align-items: flex-end; height: 160px; }
        .line-chart span { flex: 1; height: calc(var(--val,0) * 1%); background: #4a64e4; border-radius: 4px 4px 0 0; position: relative; }
        .line-chart span::after { content: ""; position: absolute; left: 50%; bottom: -6px; width: 6px; height: 6px; background: #4a64e4; border-radius: 50%; transform: translateX(-50%); }
        .scatter-plot { position: relative; height: 180px; background: repeating-linear-gradient(90deg, #f0f2ff 0, #f0f2ff 10px, #fff 10px, #fff 20px); border-radius: 8px; overflow: hidden; }
        .scatter-plot .point { position: absolute; left: calc(var(--x,10) * 1%); bottom: calc(var(--y,10) * 1%); width: 12px; height: 12px; background: #4a64e4; border-radius: 50%; opacity: 0.85; }
        .pie-chart { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.5rem; }
        .pie-chart li { background: #eef1ff; border-radius: 6px; padding: 0.5rem; }
        .pie-chart span { display: block; background: linear-gradient(90deg, #4a64e4 calc(var(--share,0) * 100%), #d8defc calc(var(--share,0) * 100%)); padding: 0.35rem 0.5rem; border-radius: 4px; color: #1c224b; }
        .rich-diagram table { width: 100%; border-collapse: collapse; }
        .rich-diagram th, .rich-diagram td { border: 1px solid #d8defc; padding: 0.35rem 0.45rem; text-align: left; }
        .rich-diagram th { background: #eef1ff; }
        </style>
        """
    )

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append(f'<html lang="{lang}">')
    parts.append("<head>")
    parts.append('  <meta charset="UTF-8" />')
    parts.append(f"  <title>{html.escape(title)}</title>")
    parts.append(style_block)
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<article>")

    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f"<p class='description'>{_format_html_text(description)}</p>")

    for i, sec in enumerate(sections, start=1):
        heading = html.escape(lbl["chapter"].format(num=i) + " " + sec.titles.get(lang, sec.titles.get("ja", "")))
        parts.append("<section>")
        parts.append(f"<h2>{heading}</h2>")
        parts.append(f"<p>{_format_html_text(sec.bodies.get(lang, sec.bodies.get('ja', '')))}</p>")
        diagram_html = sec.diagrams.get(lang) or sec.diagrams.get("ja", "")
        if diagram_html:
            parts.append(diagram_html)
        parts.append("</section>")

    parts.append("<section>")
    parts.append(f"<h2>{html.escape(lbl['summary'])}</h2>")
    parts.append(f"<p>{_format_html_text(conclusion.get(lang, conclusion.get('ja', '')))}</p>")
    parts.append("</section>")

    parts.append("<hr />")
    parts.append("<section class='meta'>")
    parts.append(f"<h3>{html.escape(lbl['description'])}</h3>")
    parts.append(f"<p>{_format_html_text(description)}</p>")
    parts.append(f"<h3>{html.escape(lbl['tags'])}</h3>")
    parts.append(f"<p>{html.escape(tags_line)}</p>")
    parts.append("</section>")

    parts.append("</article>")
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


# ======================= メインパイプライン ===================


def handle_latest_topic_request(prompt: str) -> dict[str, Any] | None:
    if not is_latest_topic_request(prompt):
        return None

    with status_scope("latest_topic"):
        keyword, source_path = pick_latest_topic_index(ARTICLES_DIR, LATEST_TOPIC_KEYWORDS)
        if not source_path:
            return {
                "ok": False,
                "flag": "NO_LATEST_TOPIC_SOURCE",
                "message": "No index file found for the latest topic request.",
                "category": "",
                "subcategory": "",
                "files": {},
                "html": {},
                "llm_calls": LLM_CALL_COUNT,
            }

        source_text = read_snippet(source_path, max_chars=12000)
        cleaned_source = _clean_text(source_text)
        summary_prompt = build_latest_topic_prompt(keyword, source_path, cleaned_source)
        llm_answer = _post_llm(SECTION_MODEL, summary_prompt)
        if _is_llm_error(llm_answer):
            return {
                "ok": False,
                "flag": "LLM_ERROR",
                "message": llm_answer,
                "category": keyword or "",
                "subcategory": "",
                "files": {},
                "html": {},
                "llm_calls": LLM_CALL_COUNT,
            }

        return {
            "ok": True,
            "flag": "FLAG:LATEST_TOPIC",
            "message": _sanitize_output_text(llm_answer),
            "category": keyword or "LATEST_TOPIC",
            "subcategory": "",
            "files": {},
            "html": {},
            "llm_calls": LLM_CALL_COUNT,
            "source_file": str(source_path),
        }


def build_three_stage_prompts(prompt: str) -> dict[str, str]:
    """Create three LLM prompts for meta info, body+diagram, and conclusion."""

    cleaned = prompt.strip()
    meta_prompt = textwrap.dedent(
        f"""
        You are preparing a blog article based on the following English client instruction.
        Respond with JSON only using these keys:
        - title: a concise headline in English
        - description: a 1-2 sentence description (limit to roughly 200 words)
        - tags: an array of exactly 6 topical tags in English
        - summary: a 3-5 sentence overview in English

        Client instruction:
        {cleaned}
        """
    ).strip()

    body_prompt = textwrap.dedent(
        f"""
        Draft the main body of a blog article from the following English client instruction.
        Return JSON only with these keys:
        - body: a detailed explanation in English (aim for roughly 800-1200 words, paragraph-friendly)
        - diagram: an ASCII diagram or structured bullet diagram that clarifies the core idea; keep it text-only

        Client instruction:
        {cleaned}
        """
    ).strip()

    conclusion_prompt = textwrap.dedent(
        f"""
        Produce the closing section for a blog article based on the following English client instruction.
        Return JSON only with this key:
        - conclusion: a concise English conclusion that reinforces the article's message

        Client instruction:
        {cleaned}
        """
    ).strip()

    return {"meta": meta_prompt, "body": body_prompt, "conclusion": conclusion_prompt}


def compose_three_stage_html(
    prompt: str, meta: Mapping[str, Any], body: Mapping[str, Any], conclusion_text: str
) -> str:
    """Compose HTML from meta/body/conclusion fields."""

    title = _sanitize_output_text(str(meta.get("title") or prompt or "Blog Article"))
    description = _sanitize_output_text(str(meta.get("description", "")))
    summary = _sanitize_output_text(str(meta.get("summary", "")))
    tags_raw = meta.get("tags") or []
    tags = [
        _sanitize_output_text(str(tag))
        for tag in tags_raw
        if isinstance(tag, (str, int, float)) and str(tag).strip()
    ]
    tags_line = ", ".join(tags[:6])

    body_text = _sanitize_output_text(str(body.get("body", "")))
    diagram_text = _sanitize_output_text(str(body.get("diagram", "")))
    conclusion = _sanitize_output_text(conclusion_text)

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html>")
    parts.append("<head>")
    parts.append("  <meta charset=\"utf-8\" />")
    parts.append(f"  <title>{html.escape(title)}</title>")
    parts.append(
        "  <style>body{font-family:Arial,Helvetica,sans-serif;line-height:1.6;padding:24px;}\n"
        "article{max-width:900px;margin:0 auto;}\nh1{margin-bottom:0.25em;}"
        "section{margin-bottom:24px;}\npre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto;}"
        "ul.tag-list{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:8px;}"
        "ul.tag-list li{background:#eef3ff;border-radius:12px;padding:6px 10px;font-size:0.9em;}"
        "p.lead{font-weight:600;margin-top:0;}\n.small-label{color:#666;font-size:0.9em;}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<article>")

    parts.append(f"<h1>{html.escape(title)}</h1>")
    if description:
        parts.append(f"<p class='lead'>{_format_html_text(description)}</p>")

    parts.append("<section>")
    parts.append("<h2>Summary</h2>")
    parts.append(f"<p>{_format_html_text(summary or description)}</p>")
    if tags_line:
        parts.append("<ul class='tag-list'>")
        for tag in tags[:6]:
            parts.append(f"<li>{html.escape(tag)}</li>")
        parts.append("</ul>")
    parts.append("</section>")

    parts.append("<section>")
    parts.append("<h2>Body</h2>")
    parts.append(f"<p>{_format_html_text(body_text)}</p>")
    if diagram_text:
        parts.append("<h3>Diagram</h3>")
        parts.append(f"<pre>{html.escape(diagram_text)}</pre>")
    parts.append("</section>")

    parts.append("<section>")
    parts.append("<h2>Conclusion</h2>")
    parts.append(f"<p>{_format_html_text(conclusion)}</p>")
    parts.append("</section>")

    parts.append("<hr />")
    parts.append("<p class='small-label'>Client instruction</p>")
    parts.append(f"<p>{_format_html_text(_sanitize_output_text(prompt))}</p>")

    parts.append("</article>")
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


def generate_three_stage_blog(prompt: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Generate HTML via three sequential LLM calls (meta/body/conclusion)."""

    with status_scope("generate_three_stage_blog"):
        prompts = build_three_stage_prompts(prompt)

        meta_raw = _post_llm(SECTION_MODEL, prompts["meta"])
        body_raw = _post_llm(SECTION_MODEL, prompts["body"])
        conclusion_raw = _post_llm(SECTION_MODEL, prompts["conclusion"])
    
        # --- meta -------------------------------------------------
        meta = _safe_json_parse(meta_raw)
        if not meta:
            meta = {
                "title": prompt.strip()[:60] or "Blog Article",
                "description": meta_raw,
                "tags": [],
                "summary": meta_raw,
            }
    
        # --- body (ここを強化) -------------------------------------
        body = _safe_json_parse(body_raw)
    
        if not body:
            # strict JSON がダメだったら、ゆるい JSON パーサを試す
            loose = _parse_loose_body_json(body_raw)
            if loose:
                body = loose
    
        if not body:
            # それでもダメなら、従来どおり丸ごと本文にする
            body = {"body": body_raw, "diagram": ""}
    
        # --- conclusion ------------------------------------------
        conclusion_data = _safe_json_parse(conclusion_raw)
        conclusion = conclusion_data.get("conclusion") if isinstance(conclusion_data, dict) else None
        conclusion = conclusion or conclusion_raw


        html_text = compose_three_stage_html(prompt, meta, body, conclusion)

        slug = _slugify(prompt)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slug}_ja.html"
        output_path.write_text(html_text, encoding="utf-8")

        files = {"ja": str(output_path), "en": str(output_path)}
        html_map = {"ja": html_text, "en": html_text}

        return {
            "ok": True,
            "flag": "FLAG:FILES_SENT",
            "message": "",
            "category": None,
            "subcategory": "",
            "files": files,
            "html": html_map,
            "llm_calls": LLM_CALL_COUNT,
        }


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return slug[:64] or "blog"


def generate_blogs(prompt: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """仕様に沿って日英伊のブログを生成し、HTMLを保存してメタ情報を返す。"""
    with status_scope("generate_blogs"):
        latest_topic = handle_latest_topic_request(prompt)
        if latest_topic is not None:
            return latest_topic

        taxonomy = discover_article_taxonomy(ARTICLES_DIR)
        categories = list(taxonomy)

        if not categories:
            return {
                "ok": False,
                "flag": "NO_CATEGORY",
                "message": "articles ディレクトリにカテゴリが見つかりません。",
                "category": "",
                "subcategory": "",
                "files": {},
                "html": {},
                "llm_calls": LLM_CALL_COUNT,
            }

        # 1. カテゴリ分類（LLM + フォールバック）
        category = classify_category_with_llm(prompt, categories)
        if not category:
            category = choose_category_fallback(prompt, categories)

        if not category:
            return {
                "ok": False,
                "flag": "NO_CATEGORY",
                "message": "読者の文章に対応するカテゴリが見つかりませんでした。",
                "category": "",
                "subcategory": "",
                "files": {},
                "html": {},
                "llm_calls": LLM_CALL_COUNT,
            }

        # 2. カテゴリ配下のインデックスを読む（長すぎる場合は先頭 15,000 文字で打ち切り）
        index_text = ""
        for p in taxonomy.get(category, []):
            index_text += "\n\n" + read_snippet(p, max_chars=15000)
        cleaned_index_text = _clean_text(index_text)

        # 3. mybrain（知識ファイル）を読む（同様に 15,000 文字に制限）
        knowledge_text = read_snippet(MYBRAIN_DIR, max_chars=15000)
        cleaned_knowledge_text = _clean_text(knowledge_text)

        # 4. セクション設計（短い抜粋だけ渡す）
        index_excerpt = cleaned_index_text[:1000]
        knowledge_excerpt = cleaned_knowledge_text[:1000]
        section_plan = plan_sections(prompt, category, index_excerpt, knowledge_excerpt)

        # 5. 各セクション本文＋図解生成（7つ）
        sections: list[Section] = []
        for sec in section_plan:
            title = sec.get("title", "")
            focus = sec.get("focus", "")
            section_obj = build_section(
                prompt,
                category,
                title,
                focus,
                cleaned_index_text,
                cleaned_knowledge_text,
            )
            sections.append(section_obj)

        # 6. 総論
        conclusion = build_conclusion(prompt, category, sections, cleaned_knowledge_text)

        # 7. メタ情報（3言語まとめて）
        meta_info = build_meta(prompt, category, sections, conclusion)

        # 8. HTML組み立て＆保存（日本語・英語・イタリア語）
        slug = _slugify(prompt)
        output_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        html_map: dict[str, str] = {}
        for lang in ("en", "ja", "it"):
            meta = meta_info.get(lang, meta_info.get("ja", {}))
            html_text = compose_html(lang, meta, sections, conclusion)
            output_path = output_dir / f"{slug}_{lang}.html"
            output_path.write_text(html_text, encoding="utf-8")
            files[lang] = str(output_path)
            html_map[lang] = html_text

        return {
            "ok": True,
            "flag": "FLAG:FILES_SENT",
            "message": "",
            "category": category,
            "subcategory": "",
            "files": files,
            "html": html_map,
            "llm_calls": LLM_CALL_COUNT,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Japanese blog HTML from a prompt.")
    parser.add_argument("prompt", nargs="?", help="Client-provided theme or text. If omitted, read from stdin.")
    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the generated HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt or html.unescape(os.sys.stdin.read()).strip()
    if not prompt:
        raise SystemExit("読者からの文章が指定されていません。")

    start = time.time()
    results = generate_blogs(prompt, output_dir=args.output_dir)
    elapsed = time.time() - start

    if not results.get("ok", True):
        print(results.get("flag", "NO_CATEGORY"))
        msg = results.get("message")
        if msg:
            print(msg)
        print(f"LLM calls: {results.get('llm_calls', 0)}  elapsed: {elapsed:.1f}s")
        return

    files = results.get("files", {})
    for lang, path in files.items():
        print(f"Saved blog HTML ({lang}) to {path}")

    print(f"LLM calls: {results.get('llm_calls', 0)}  elapsed: {elapsed:.1f}s")
    print(results.get("flag", "FLAG:FILES_SENT"))


if __name__ == "__main__":
    main()