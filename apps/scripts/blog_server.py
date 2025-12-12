#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_article_generator.py

スクレイピング済みの indexes/articles 配下のインデックスを入口に、
mybrain を含む知識ファイルをコンテキストにして 日本語記事（+ 日本語図解） を自動生成するスクリプト。

フロー:
 1. キーワードを1つランダムに選択
 2. INDEX_ROOT 配下のインデックスファイルから、phi3:mini に選ばせる
 3. 選ばれたインデックスをもとに phi3:mini に
      - タイトル(50字以内, 日本語)
      - ディスクリプション(約200字, 日本語)
      - タグ6つ (日本語)
      - 概要(約500字, 日本語)
    を JSON で生成させる
 4. 概要をもとに phi3:mini に 7 セクションのアウトラインを日本語で作らせる
 5. 各セクションごとに
      - llama3:8b に約1000〜1200字の本文を書かせる（日本語のみ）
 6. すべての日本語本文をもとに、llama3:8b に約1500字の総論を書かせる（日本語のみ）
 7. 日本語本文をもとに、qwen2.5-coder:7b で日本語図解HTMLを作成する（各セクション1つ）
 8. 日本語のHTMLファイルとメタ情報(JSON)を保存する

ログ:
  - 各プロセスごとに /var/www/Meta-Project/apps/logs/Blog に
    blog_YYYYMMDD-HHMMSS_pidXXXX.log を生成する。
"""

from __future__ import annotations

import datetime as dt
import json
from html.parser import HTMLParser
import os
import random
import re
import socket
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.scripts.keyword_selector import generate_search_query, select_keywords

# ====== 設定 =========================================================

# Ollama のエンドポイント
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")

# インデックスのルート（スクレイピングしてきた記事群）
INDEX_ROOT = Path("/var/www/Meta-Project/indexes/articles")

# 知識ファイルのルート（メモ・ナレッジなど）
KNOWLEDGE_ROOT = Path("/var/www/Meta-Project/indexes/mybrain")

# 出力先ディレクトリ (共有フォルダ側)
DEFAULT_OUTPUT_DIR = Path("/mnt/hgfs/output")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# ログ出力先
LOG_DIR = Path("/var/www/Meta-Project/apps/logs/Blog")
LOG_FILE: Optional[Path] = None
LOG_START_TIME: Optional[dt.datetime] = None

# インデックスとして扱う拡張子
INDEX_EXTS = {".txt", ".html"}

# 「知識ファイル」として扱う拡張子
TEXT_EXTS = {".txt", ".md", ".json", ".html", ".rst"}

# 使うモデル名
MODEL_PHI3 = "phi3:mini"
MODEL_LLAMA3 = "llama3:8b"
MODEL_CODEGEMMA = "qwen2.5-coder:7b"

DIAGRAM_LANG_CONFIG = {
    "ja": {"label": "Japanese", "heading": "主要ポイント"},
    "en": {"label": "English", "heading": "Key Takeaways"},
    "it": {"label": "Italian", "heading": "Punti chiave"},
}

# キーワード一覧（例）
KEYWORDS = [
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
    "Broken Utopia（壊れたユートピア）",
    "DeFi Church（ディーファイ教会）",
    "World Bug Hunters（世界のバグハンター）",
    "Anti-Propaganda Art（アンチプロパガンダ・アート）",
    "Sacred Wallet（聖なるウォレット）",
    "Conflicted Believer（揺れる信仰者）",
    "Global Surveillance Age（グローバル監視時代）",
    "Girl’s Secret Intelligence Service（少女秘密情報部）",
    "Emotional Volatility Index（感情ボラティリティ指数）",
    "Rebel Asset Class（反逆者アセットクラス）",
    "Anxiety-Backed Token（不安担保トークン）",
    "Neo-Underground Culture（ネオ・アンダーグラウンド文化）",
    "Borderless Heresy（ボーダーレス異端）",
    "Quiet Panic Generation（静かなパニック世代）",
    "Spy-Fi Metaverse（スパイ×SFメタバース）",
    "Punk Girls Intelligence Agency（パンク少女情報庁）",
    "Holy FUD（聖なるFUD）",
    "Sanctuary Nodes（聖域ノード）",
    "Confession on Chain（チェーン上の告解）",
    "Emotional Exploit（感情エクスプロイト）",
    "Black Swan Religion（ブラックスワン宗教）",
    "Digital Martyrdom（デジタル殉教）",
    "Geopolitical Meme War（地政学ミーム戦争）",
    "Shadow Wallet Society（シャドウウォレット社会）",
    "Anti-Fragile Faith（反脆弱な信仰）",
    "Burnout Capitalism（バーンアウト資本主義）",
    "Silent Rage Protocol（サイレントレイジ・プロトコル）",
    "Underdog Intelligence Network（負け犬インテリジェンス網）",
    "Emotional Zero Day（感情ゼロデイ）",
    "Doomscrolling Believer（ドゥームスクロール信者）",
    "Hope Mining（希望マイニング）",
    "Punk Redemption Arc（パンクの贖いアーク）",
    "Anxious Hodler’s Club（不安なホドラーズクラブ）",
    "Broken World, Cute Girls（壊れた世界とかわいい少女たち）",
]

RAHAB_CATEGORY_URL = "https://rahabpunkaholicgirls.com/category/stories-en/nft-music-en/"

# ====== ログユーティリティ =========================================


def init_logger() -> None:
    """
    プロセスごとに別ファイルにログを書き出す。
    例: blog_20251207-120000_pid12345.log
    """
    global LOG_FILE, LOG_START_TIME
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    LOG_START_TIME = dt.datetime.now()
    ts = LOG_START_TIME.strftime("%Y%m%d-%H%M%S")
    pid = os.getpid()
    LOG_FILE = LOG_DIR / f"blog_{ts}_pid{pid}.log"


def log(msg: str) -> None:
    """
    標準出力 + ログファイルの両方に書く。
    """
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"

    # 標準出力
    print(line, flush=True)

    # ログファイル
    if LOG_FILE is not None:
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # ログ書き込み失敗で本体を落とさない
            pass


# ====== 共通ユーティリティ =========================================


def call_ollama_generate(model: str, prompt: str, temperature: float = 0.4) -> str:
    """
    Ollama /api/generate を叩いて単発プロンプトを投げる。
    重い処理なのでタイムアウトは 1800 秒に設定。
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=1800) as resp:
            body = resp.read().decode("utf-8")
        res = json.loads(body)
        return res.get("response", "").strip()

    except (TimeoutError, socket.timeout) as e:
        log(f"ERROR: Ollama request TIMEOUT for model={model}: {e}")
        return ""

    except (URLError, HTTPError) as e:
        log(f"ERROR: Ollama request failed for model={model}: {e}")
        return ""

    except json.JSONDecodeError:
        log("ERROR: Failed to decode Ollama response JSON")
        return ""


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    モデルから返ってきたテキストから JSON ブロックだけを抜き出す。
    最初の '{' から最後の '}' までを JSON とみなす簡易版。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def extract_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    JSON の配列 (list) を抜き出す版。
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        data = json.loads(snippet)
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except json.JSONDecodeError:
        return None
    return None


def tokenize_keyword(keyword: str) -> List[str]:
    """Split keywords into searchable chunks.

    The splitter is intentionally simple so that unit tests remain deterministic
    while still covering cases like "Underdog Intelligence Network（負け犬インテリジェンス網）".
    """

    tokens: List[str] = []
    for raw in re.split(r"[\s、,，。・（）()\[\]{}<>《》/\\|]+", keyword):
        token = raw.strip()
        if token:
            tokens.append(token)
    return tokens


def read_text(path: Path, max_chars: int | None = None) -> str:
    """
    テキストファイルを読み込む。JSONでもそのまま文字列として扱う。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""
    if max_chars is not None:
        return text[:max_chars]
    return text


def slugify(value: str) -> str:
    """
    ファイル名用の簡易スラッグ。
    """
    value = value.strip().lower()
    value = re.sub(r"[^\w]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "article"


def translate_text(text: str, target_lang: str) -> str:
    """
    日本語テキストを target_lang に翻訳する。
    ※現バージョンでは main() からは呼ばない（将来拡張用）。
    """
    if not text.strip():
        return text
    prompt = textwrap.dedent(
        f"""
        You are a professional translator.
        The source text is Japanese. Translate it into natural {target_lang}.
        - Preserve the meaning and nuance.
        - Use fluent, publication-quality {target_lang}.
        - Do NOT explain, do NOT repeat the Japanese.
        - Output ONLY the translated text.

        ----
        {text}
        ----
        """
    )
    return call_ollama_generate(MODEL_LLAMA3, prompt)


# ====== ステップ 1: キーワード & インデックス選択 ===================


def find_index_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in INDEX_EXTS:
            files.append(p)
    return files


def search_indexes(keyword: str, files: List[Path]) -> List[Path]:
    """
    Return index files that mention the keyword.

    A lightweight substring search keeps the function deterministic and
    testable without external services.
    """

    hits: List[Path] = []
    if not keyword.strip():
        return hits

    query = generate_search_query(keyword)
    lowered_query = query.lower()
    lowered_keyword = keyword.strip().lower()
    for path in files:
        text = read_text(path, max_chars=2000).lower()
        if (lowered_keyword and lowered_keyword in text) or (
            lowered_query and lowered_query in text
        ):
            hits.append(path)
    return hits


def search_indexes_by_parts(keyword: str, files: List[Path]) -> List[Path]:
    """Try each keyword fragment until we collect hits."""

    seen = set()
    hits: List[Path] = []
    for token in tokenize_keyword(keyword):
        for path in search_indexes(token, files):
            if path not in seen:
                hits.append(path)
                seen.add(path)
    return hits


def parse_related_terms(text: str) -> List[str]:
    """Parse llama/phi suggestions into a clean term list."""

    terms: List[str] = []
    if not text.strip():
        return terms

    for line in re.split(r"[\n,]", text):
        term = line.strip(" -•・\t")
        if term:
            terms.append(term)
    return terms


def suggest_related_terms_with_llm(keyword: str, limit: int = 3) -> List[str]:
    prompt = textwrap.dedent(
        f"""
        あなたは関連語を提案するアシスタントです。
        次のキーワードを検索するために、意味が近い語を{limit}個以内で提案してください。
        - 返答は日本語・英語の混在可。
        - 一行ごとに1単語/フレーズだけを書き、説明は不要です。
        - 略語ではなく、検索に使える語を返してください。

        キーワード: {keyword}
        """
    )

    raw = call_ollama_generate(MODEL_PHI3, prompt)
    terms = parse_related_terms(raw)[:limit]
    log(f"Related term suggestions: {terms}")
    return terms


def search_indexes_for_terms(terms: List[str], files: List[Path]) -> List[Path]:
    seen = set()
    hits: List[Path] = []
    for term in terms:
        for path in search_indexes(term, files):
            if path not in seen:
                hits.append(path)
                seen.add(path)
    return hits


def locate_relevant_indexes(keyword: str, index_files: List[Path]) -> List[Path]:
    matched_indexes = search_indexes(keyword, index_files)
    if matched_indexes:
        return matched_indexes

    log("No search hits found. Trying keyword fragments...")
    matched_indexes = search_indexes_by_parts(keyword, index_files)
    if matched_indexes:
        return matched_indexes

    log("No fragment hits. Asking LLM for related terms...")
    related_terms = suggest_related_terms_with_llm(keyword)
    if not related_terms:
        return []

    matched_indexes = search_indexes_for_terms(related_terms, index_files)
    return matched_indexes


def choose_index_with_phi3(keyword: str, files: List[Path]) -> Optional[Path]:
    """
    複数のインデックス候補から、phi3:mini に 1 つ選ばせる。
    """
    if not files:
        return None

    entries: List[str] = []
    for i, path in enumerate(files):
        snippet = read_text(path, max_chars=800)
        rel = path.relative_to(INDEX_ROOT)
        entries.append(
            f"[{i}] {rel}\n"
            f"---------\n"
            f"{snippet}\n"
        )

    joined = "\n\n".join(entries)

    prompt = textwrap.dedent(
        f"""
        あなたはインデックスファイルの選定アシスタントです。
        テーマ:「{keyword}」

        以下に、複数のインデックスファイルの候補が番号付きで並んでいます。
        テーマに最も関連が深そうなものを 1 つだけ選んでください。

        出力フォーマット:
        - 選んだ番号だけを半角数字で1つだけ出力してください。
        - 説明や他の文字は一切書かないでください。

        候補一覧:
        {joined}
        """
    )
    res = call_ollama_generate(MODEL_PHI3, prompt)
    m = re.search(r"\d+", res)
    if not m:
        log("WARN: phi3 から選択番号が取得できなかったため、先頭ファイルを採用します。")
        return files[0]
    idx = int(m.group(0))
    if idx < 0 or idx >= len(files):
        log("WARN: phi3 が範囲外の番号を返したため、先頭ファイルを採用します。")
        return files[0]
    return files[idx]


# ====== ステップ 2: メタ情報 (タイトル/説明/タグ/概要) ================


def generate_metadata_with_phi3(
    keyword: str, index_text: str, rahab_context: str = ""
) -> Dict[str, Any]:
    """
    phi3:mini にメタ情報を JSON で作らせる（すべて日本語）。
    """
    prompt = textwrap.dedent(
        f"""
        あなたは「Panchan」というペルソナに向けた
        日本語ブログの編集者です。
        テーマは「{keyword}」です。
        ペルソナ特徴: 静かな考察を好み、批判的だが希望も探る。
        ペルソナが抱えている主な問題: 情報過多と不信感により、
        Web3 や NFT に惹かれつつも心の軸や行動指針を見いだせていない。
        トーン: シリアスだが読者に寄り添う穏やかな口調。

        以下のインデックス内容と Rahab Punkaholic Girls の曲リストを読み、
        このテーマに沿ったブログ記事の
        メタ情報を**すべて日本語**で作成してください。
        インデックスに書かれていない事実は絶対に補わず、推測で埋めないでください。
        わからない場合は「情報が不足している」と簡潔に書いてください。

        タイトル・ディスクリプション・タグ・概要のすべてで、
        上記の問題に対して解決策や前進の手がかりを提示する構成にしてください。

        <INDEX>
        {index_text}
        </INDEX>

        <RAHAB_TRACKS>
        {rahab_context}
        </RAHAB_TRACKS>

        必ず次の形式の JSON だけを出力してください。
        キーも値も日本語で書き、英語文は使わないでください。

        {{
          "title": "ペルソナ特徴とトーンを踏まえ、問題解決を示唆するタイトル。日本語で50文字以内",
          "description": "約200文字のディスクリプション。ペルソナへの共感と解決策の方向性を含める。",
          "tags": ["タグ1", "タグ2", "タグ3", "タグ4", "タグ5", "タグ6"],
          "overview": "記事全体の内容をまとめた概要。約500文字。日本語のみで書き、問題解決に向けた流れを示す。"
        }}
        """
    )
    res = call_ollama_generate(MODEL_PHI3, prompt)
    meta = extract_json_block(res) or {}
    return meta

# ====== ステップ 3: 7 セクション構成 ================================


def generate_sections_with_phi3(
    keyword: str, overview: str, rahab_context: str = ""
) -> List[Dict[str, str]]:
    """
    日本語の概要から 7 つのセクション（タイトル＋要約）を日本語で作る。

    ポイント:
    - LLM が返す "title" は信用せず、summary と keyword からこちらでタイトルを決め直す。
    - 汎用的な「セクション1」「第1節」などは使わない。
    """

    # 1) まずは LLM に「summary 付きのセクション案」を考えさせる
    prompt = textwrap.dedent(
        f"""
        あなたは長文ブログの構成作家です。
        テーマ「{keyword}」の記事について、以下の概要を元に
        7つのセクション構成を**日本語**で考えてください。
        概要やインデックスに書かれていない固有名詞や事実を新規に作らないでください。

        読者となるペルソナが抱える「情報過多による不信感」と
        「行動指針の欠如」をやわらげ、具体的な解決策や前進の道筋を
        提案できるようなセクションを組み立ててください。

        概要（日本語）:
        {overview}

        Rahab Punkaholic Girls の曲が持つ世界観と主張:
        {rahab_context}

        各セクションについて、次の情報を用意してください:
        - "title": いったん仮のタイトルで構いません（後で機械的に書き換えます）
        - "summary": セクションで展開する内容の説明
                     （2〜3文、合計で約150文字、日本語）
                     ペルソナの問題解決に直結する視点・提案を含めること。

        次の形式の JSON 配列だけを出力してください:
        （JSON 以外の文字は出力しない）

        [
          {{"title": "仮タイトル1", "summary": "説明文..." }},
          ...  // 合計7つ
        ]
        """
    )

    res = call_ollama_generate(MODEL_PHI3, prompt)
    sections = extract_json_list(res) or []

    cleaned: List[Dict[str, str]] = []

    def make_label_from_summary(summary: str, idx: int, keyword: str) -> str:
        """
        summary から見出し用の短いラベルを作る。
        - 最初の文（「。」まで）を見出し候補にする
        - 長すぎたら 24 文字で切る
        - 何もなければ keyword を使ったテンプレートから決める
        """

        s = (summary or "").strip()
        if s:
            # 「。」or「．」で最初の文だけ取る
            first = re.split(r"[。．]", s, maxsplit=1)[0].strip()
            base = first or s
            if len(base) > 24:
                base = base[:24] + "…"
            # 「第1節」「セクション1」みたいなゴミタイトルなら捨てる
            if not re.fullmatch(r"(第?\d+節?|セクション\d+)", base):
                return base

        # ---- ここからフォールバック（summary が空・微妙なとき）----

        # keyword が「Faith vs. Market（信仰かマーケットか）」みたいな形の場合、
        # カッコ内の日本語だけ抜く。それもなければ keyword そのものを使う。
        m = re.search(r"（(.+?)）", keyword)
        base_kw = m.group(1) if m else keyword

        patterns = [
            "{}とは何かを整理する",
            "{}の歴史的背景",
            "{}と日常生活のあいだ",
            "{}とテクノロジー／ネット社会",
            "{}が孕むリスクと矛盾",
            "{}との付き合い方と実践",
            "これからの{}と私たち",
        ]
        idx0 = min(max(idx - 1, 0), len(patterns) - 1)
        return patterns[idx0].format(base_kw)

    # 7 セクション分きっちり作る
    for i in range(7):
        if i < len(sections):
            sec = sections[i]
            summary = str(sec.get("summary") or "")
        else:
            summary = ""

        title = make_label_from_summary(summary, i + 1, keyword)
        cleaned.append({"title": title, "summary": summary})

    return cleaned


# ====== ステップ 4: 各セクション本文 ================================


def collect_knowledge_text(chosen_index: Path, max_chars: int = 4000) -> str:
    """
    インデックスファイルと、その周辺の知識ファイルを連結して
    LLM に渡すコンテキストテキストを作る。
    """
    chunks: List[str] = []

    # インデックス自体
    chunks.append(f"[INDEX] {chosen_index}\n")
    chunks.append(read_text(chosen_index, max_chars=2000))

    # インデックスと同じディレクトリ配下のファイルを軽く参照
    for p in chosen_index.parent.rglob("*"):
        if not p.is_file():
            continue
        if p == chosen_index:
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        chunks.append(f"\n[FILE] {p}\n")
        chunks.append(read_text(p, max_chars=800))
        if sum(len(c) for c in chunks) > max_chars:
            break

    text = "\n".join(chunks)
    return text[:max_chars]


def parse_rahab_tracks(html: str) -> List[str]:
    class TrackParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.capture = False
            self.buffer: List[str] = []
            self.tracks: List[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag.lower() != "a":
                return
            href = dict(attrs).get("href", "")
            if "stories-en/nft-music-en" in href:
                self.capture = True
                self.buffer = []

        def handle_data(self, data: str) -> None:
            if self.capture:
                self.buffer.append(data)

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() == "a" and self.capture:
                title = "".join(self.buffer).strip()
                if title:
                    self.tracks.append(title)
                self.capture = False
                self.buffer = []

    parser = TrackParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    # Preserve order while removing duplicates
    seen = set()
    unique_tracks: List[str] = []
    for title in parser.tracks:
        if title not in seen:
            unique_tracks.append(title)
            seen.add(title)
    return unique_tracks


def fetch_rahab_tracks(category_url: str = RAHAB_CATEGORY_URL, max_pages: int = 1) -> List[str]:
    """
    Rahab のカテゴリページから曲タイトルを取得する。
    パフォーマンスのため、デフォルトでは 1 ページのみ見る。
    """
    tracks: List[str] = []
    seen = set()
    for page in range(1, max_pages + 1):
        url = category_url if page == 1 else urljoin(category_url, f"page/{page}/")
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001 - keep running even if network fails
            log(f"WARN: Failed to fetch Rahab tracks from {url}: {exc}")
            break

        titles = parse_rahab_tracks(html)
        for title in titles:
            if title not in seen:
                tracks.append(title)
                seen.add(title)

        if not titles:
            break

    return tracks


def build_rahab_worldview_block(tracks: List[str]) -> str:
    if not tracks:
        return ""

    lines = [
        "Rahab Punkaholic Girls のNFTミュージックから拾った曲とモチーフ:",
    ]
    for title in tracks:
        lines.append(f"- {title}")
    lines.append(
        "これらの曲が描く反抗・救済・信仰と市場の緊張感を主張に織り込み、"
        "記事テーマと共鳴させてください。"
    )
    return "\n".join(lines)


def write_section_body_with_llama3(
    keyword: str,
    persona: str,
    section: Dict[str, str],
    article_title: str,
    overview: str,
    index_text: str,
    knowledge_text: str,
    rahab_context: str = "",
) -> str:
    """
    llama3:8b に1セクション分の本文を書かせる（日本語のみ）。
    目安: 1000〜1200文字。
    """
    prompt = textwrap.dedent(
        f"""
        あなたは日本語で文章を書く哲学系ブロガーです。
        ペルソナ: {persona}
        設定: パンクロック、NFT、Web3、暗号通貨、信仰と市場の葛藤などを
        テーマにしつつ、批評的かつ文学的な文体で論じます。
        宣伝や特定サービスのPRは絶対に行わないでください。
        読者は情報過多と不信感に悩み、指針を求めています。
        本文ではその問題をほぐし、行動や思考の具体的な解決策を提示してください。

        記事のタイトル: {article_title}
        記事の大テーマ: {keyword}
        記事全体の概要:
        {overview}

        セクション情報（日本語）:
        タイトル: {section.get("title")}
        セクション概要:
        {section.get("summary")}

        インデックスファイルからの情報（原文は英語など混在可）:
        {index_text}

        知識ファイルからの抜粋:
        {knowledge_text}

        Rahab Punkaholic Girls の楽曲が示す世界観・主張:
        {rahab_context}

        上記をすべて踏まえて、このセクションの本文を**日本語だけで**
        1000〜1200文字を目安に論文風で執筆してください。

        条件:
        - 出力はすべて自然な日本語で書くこと（英語の文を混在させない）
        - セクションのタイトルと概要に沿った内容にすること
        - インデックスと知識ファイルに書かれている情報から論を組み立て、
          そこにない事実を作らないこと。情報が不足している場合は、
          「情報が不足している」と述べて補わないこと。
        - 同じ文章の繰り返しや文字数稼ぎを避け、不要なら短くまとめること
        - 特定の企業・取引所・コイン等の宣伝は一切しないこと
        - 読者に思考を促すような哲学的な語り口にすること
        - ペルソナが抱える課題の解決策や、次の一歩につながる提案を必ず含めること
        """
    )
    return call_ollama_generate(MODEL_LLAMA3, prompt)


def clean_paragraphs(text: str, max_chars: int = 1400) -> str:
    """
    - 連続した重複段落を間引く
    - 文字数が長すぎる場合は適度に切り詰める
    """
    cleaned_lines: List[str] = []
    last_line = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == last_line:
            continue
        cleaned_lines.append(line)
        last_line = line

    cleaned = "\n".join(cleaned_lines)
    if len(cleaned) > max_chars:
        trimmed = cleaned[:max_chars]
        # 文末で切れた場合は句点まで戻す
        if "。" in trimmed:
            trimmed = trimmed.rsplit("。", 1)[0] + "。"
        cleaned = trimmed
    return cleaned


def normalize_description(
    description: str, fallback: str = "", target: int = 200, tolerance: int = 40
) -> str:
    """
    ブログのディスクリプションを約200文字に収める。

    - 空なら fallback を簡易要約として利用する
    - 長すぎる場合は target 付近で切り詰める
    """

    desc = (description or "").strip()
    if not desc:
        fallback_text = (fallback or "").strip()
        if fallback_text:
            desc = fallback_text[: target + tolerance]
    if not desc:
        return ""

    min_len = max(0, target - 20)
    max_len = target + 20

    if len(desc) > max_len:
        trimmed = desc[:max_len]
        if "。" in trimmed:
            trimmed = trimmed.rsplit("。", 1)[0] + "。"
        desc = trimmed

    if len(desc) < min_len and len(desc) > 0:
        desc = desc.ljust(min_len, "。")

    return desc


def normalize_tags(tags: Any, keyword: str, desired_count: int = 6) -> List[str]:
    """
    タグをちょうど6個に揃える。
    足りない場合はキーワードベースのタグで補完する。
    """

    cleaned: List[str] = []
    if isinstance(tags, list):
        for t in tags:
            tag_str = str(t).strip()
            if tag_str and tag_str not in cleaned:
                cleaned.append(tag_str)

    base = keyword.strip() or "タグ"
    while len(cleaned) < desired_count:
        cleaned.append(f"{base} {len(cleaned) + 1}")

    return cleaned[:desired_count]


def normalize_metadata(meta: Dict[str, Any], keyword: str, description_source: str) -> Dict[str, Any]:
    """
    メタ情報の不足を補い、ディスクリプションとタグを所望の形式に整える。
    """

    normalized = dict(meta or {})
    normalized["description"] = normalize_description(
        normalized.get("description", ""), fallback=description_source
    )
    normalized["tags"] = normalize_tags(normalized.get("tags"), keyword)
    return normalized


# ====== 図解 HTML（日本語セクションのみ） ===========================


def write_section_html_with_codegemma(
    section_title: str,
    section_body: str,
    diagram_language: str,
) -> str:
    """
    codegemma に、セクション内容を可視化する
    リッチな HTML (JS + CSS 込み) を作らせる。
    """
    prompt = textwrap.dedent(
        f"""
        You are a front-end engineer and data-visualization designer.
        Based on the following section of an article, create ONE rich,
        self-contained HTML snippet that visually explains the core ideas.

        IMPORTANT LANGUAGE RULES:
        - All text INSIDE the diagram (titles, labels, legends, tooltips, etc.)
          MUST be in {diagram_language}.
        - Do NOT mix other languages inside the visual content.

        VISUAL REQUIREMENTS:
        - Do NOT just output a simple <ul> with bullet points.
        - Use a combination of semantic HTML + CSS + vanilla JavaScript.
        - Include inline <style> and <script>.
        - Implement at least ONE interactive or animated behavior, for example:
          * hover highlight of nodes,
          * click to toggle details,
          * step-by-step reveal,
          * animated progress bar,
          * simple chart that updates on click.
        - Use a responsive layout with flexbox or CSS grid.

        DIAGRAM PATTERNS (pick ONE that fits best):
        - Article roadmap / progress tracker (セクション1〜7の全体像)
        - Concept map / mind map（概念・用語の関係図）
        - Flowchart or step-by-step process（仕組みやデータの流れ）
        - Architecture / system diagram（全体構成図）
        - Comparison table styled with CSS（比較表）
        - Simple chart using <svg> (line / bar / pie)（数値・トレンド図）
        - Step-by-step / checklist layout（手順・チェックリスト）
        - Risk matrix (probability × impact)（リスク・注意点）
        - Persona / use-case card layout（ペルソナ・ユースケース）
        - Summary board of 3–5 key points（要点サマリー）

        HTML CONSTRAINTS:
        - The snippet will be injected inside:
              <div class="section-visual"> ... </div>
          so DO NOT include <html>, <head>, or <body> tags.
        - Start with:
              <section class="auto-visual">
                <h3>{section_title} – [short diagram name]</h3>
                ...
              </section>
        - Use only inline CSS (<style>) and inline JS (<script>).
        - Do NOT load external libraries (no CDN, no frameworks).
        - Do NOT wrap your answer in ``` fences.

        SECTION TITLE ({diagram_language}):
        {section_title}

        SECTION BODY ({diagram_language}):
        {section_body}

        Output ONLY the HTML snippet, with no explanation and no backticks.
        """
    )
    return call_ollama_generate(MODEL_CODEGEMMA, prompt)


def build_fallback_visual(
    section_title: str, section_body: str, heading_label: str = "Key Takeaways"
) -> str:
    """LLM が HTML を返さない場合に備えた簡易ビジュアル。"""
    sentences = [s.strip() for s in re.split(r"[。.!?]\s*", section_body) if s.strip()]
    bullets = sentences[:4] if sentences else [section_title]
    items = "".join(
        f"<li>{textwrap.shorten(b, width=140, placeholder='…')}</li>" for b in bullets
    )
    return textwrap.dedent(
        f"""
        <section class="auto-visual fallback-visual">
          <h3>{section_title} – {heading_label}</h3>
          <ul>{items}</ul>
          <style>
            .fallback-visual {{
              border: 1px solid #d0d7de;
              border-radius: 8px;
              padding: 1rem;
              background: #f8fafc;
            }}
            .fallback-visual h3 {{
              margin-top: 0;
              font-size: 1.1rem;
            }}
            .fallback-visual ul {{
              padding-left: 1.2rem;
              margin: 0.5rem 0 0;
              display: grid;
              gap: 0.3rem;
            }}
          </style>
        </section>
        """
    ).strip()


def ensure_visual_snippet(
    section_title: str, section_body: str, html_snippet: str, heading_label: str
) -> str:
    """
    図解HTMLを検証しつつ、多少の崩れは許容して採用する。
    本当に壊れている場合のみ箇条書きのフォールバックに落とす。
    """
    snippet = (html_snippet or "").strip()
    if not snippet:
        log(f"WARN: Empty visual snippet for section '{section_title}', using fallback.")
        return build_fallback_visual(section_title, section_body, heading_label)

    # ```html ... ``` 形式を剥がす
    if snippet.startswith("```"):
        snippet = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", snippet)
        snippet = re.sub(r"\s*```$", "", snippet).strip()

    lowered = snippet.lower()

    # フルHTMLドキュメントを返してきた場合は破綻しやすいのでフォールバック
    if any(tag in lowered for tag in ["<html", "<head", "<body", "<!doctype"]):
        log(
            "WARN: Visual snippet contained full document tags; "
            f"using fallback for '{section_title}'."
        )
        return build_fallback_visual(section_title, section_body, heading_label)

    # HTMLタグが全くない場合もフォールバック
    if "<" not in snippet:
        log(
            f"WARN: Visual snippet missing HTML tags for '{section_title}', using fallback."
        )
        return build_fallback_visual(section_title, section_body, heading_label)

    # auto-visual が無い場合はこちらでラップする
    if "auto-visual" not in snippet:
        log(
            "INFO: Visual snippet missing 'auto-visual' wrapper; "
            f"wrapping snippet for '{section_title}'."
        )
        snippet = textwrap.dedent(
            f"""
            <section class="auto-visual">
              <h3>{section_title} – {heading_label}</h3>
              {snippet}
            </section>
            """
        ).strip()

    return snippet


def generate_section_visuals(
    sections: List[Dict[str, Any]],
    section_bodies: List[str],
    lang_code: str,
) -> List[str]:
    lang_config = DIAGRAM_LANG_CONFIG.get(lang_code, DIAGRAM_LANG_CONFIG["ja"])
    heading_label = lang_config.get("heading", "主要ポイント")
    diagram_language = lang_config.get("label", "Japanese")

    visuals: List[str] = []
    for i, sec in enumerate(sections):
        title = str(sec.get("title") or f"Section {i + 1}")
        body = section_bodies[i] if i < len(section_bodies) else ""
        log(
            f"Generating {lang_code.upper()} HTML visual for section {i+1}: {title}"
        )
        html_snippet = write_section_html_with_codegemma(
            section_title=title,
            section_body=body,
            diagram_language=diagram_language,
        )
        visuals.append(ensure_visual_snippet(title, body, html_snippet, heading_label))
    return visuals


# ====== ステップ 5: 総論 ============================================


def write_conclusion_with_llama3(
    keyword: str,
    persona: str,
    overview: str,
    all_section_bodies: List[str],
    rahab_context: str = "",
) -> str:
    """
    日本語で総論を書く。本文もすべて日本語。
    """
    joined = "\n\n".join(all_section_bodies)
    prompt = textwrap.dedent(
        f"""
        あなたは日本語で文章を書く哲学系ブロガーです。
        ペルソナ: {persona}
        大テーマ: {keyword}

        記事の概要（日本語）:
        {overview}

        Rahab Punkaholic Girls の曲がもたらす世界観:
        {rahab_context}

        以下に、すでに執筆済みの各セクション本文があります（日本語）。
        ----
        {joined}
        ----

        これら全体を踏まえた「総論」を日本語で約1500文字書いてください。

        条件:
        - 各セクションの内容を踏まえて、全体の問題意識とメッセージを総括する
        - ペルソナが抱える不信感と迷いに対し、具体的な解決策や次の一歩を示す
        - 読者に静かな余韻と問いを残す締めくくりにする
        - 特定のサービスや商品の宣伝は一切しない
        - 同一文章の繰り返しは避ける
        - 出力はすべて自然な日本語だけで書く（英語文を混在させない）
        """
    )
    return call_ollama_generate(MODEL_LLAMA3, prompt)


# ====== HTML 出力 ====================================================


def build_html_document(
    lang: str,
    meta: Dict[str, Any],
    sections: List[Dict[str, Any]],
    section_bodies: List[str],
    section_htmls: List[str],
    conclusion: str,
) -> str:
    """
    1言語分の HTML を組み立てる。
    現バージョンでは lang="ja" のみ使用。
    """
    title = str(meta.get("title") or "自動生成記事")
    description = str(meta.get("description") or "")
    tags = meta.get("tags") or []
    overview = str(meta.get("overview") or "")

    tags_line = ", ".join(str(t) for t in tags)

    # 見出しのラベル
    if lang == "en":
        overview_heading = "Overview"
        conclusion_heading = "Conclusion"
    elif lang == "it":
        overview_heading = "Panoramica"
        conclusion_heading = "Conclusione"
    else:
        # 日本語では「概要」という見出しは使わない
        overview_heading = "イントロダクション"
        conclusion_heading = "総論"

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append(f'<html lang="{lang}">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8" />')
    parts.append(f"<title>{title}</title>")
    parts.append(f'<meta name="description" content="{description}">')
    parts.append(
        "<style>"
        "html, body { min-height: 100%; margin: 0; padding: 0; }"
        "body { font-family: 'Noto Sans JP', 'Helvetica Neue', Arial, sans-serif;"
        " line-height: 1.6; font-size: 16px; color: #111;"
        " padding: 1.5rem; box-sizing: border-box; overflow-y: auto; background: #fff; }"
        "article { max-width: 960px; margin: 0 auto; }"
        "section { margin-bottom: 2rem; }"
        "header h1 { margin-bottom: 0.3rem; }"
        "header p { margin: 0.2rem 0; }"
        ".section-body { display: grid; gap: 0.8rem; }"
        ".section-visual { background: #f7f8fa; border: 1px solid #e5e7eb;"
        " padding: 1rem; border-radius: 10px; overflow-x: auto; }"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<article>")
    parts.append("<header>")
    parts.append(f"<h1>{title}</h1>")
    if tags_line:
        parts.append(f"<p><strong>Tags:</strong> {tags_line}</p>")
    parts.append(f"<p>{description}</p>")
    parts.append("</header>")

    # Overview
    parts.append("<section>")
    parts.append(f"<h2>{overview_heading}</h2>")
    for paragraph in overview.splitlines():
        paragraph = paragraph.strip()
        if paragraph:
            parts.append(f"<p>{paragraph}</p>")
    parts.append("</section>")

    # Sections
    for i, sec in enumerate(sections):
        body = section_bodies[i] if i < len(section_bodies) else ""
        html_snippet = section_htmls[i] if i < len(section_htmls) else ""
        parts.append("<section>")
        parts.append(f"<h2>{sec.get('title')}</h2>")
        if sec.get("summary"):
            parts.append(f"<p><em>{sec.get('summary')}</em></p>")
        parts.append('<div class="section-body">')
        for paragraph in body.splitlines():
            paragraph = paragraph.strip()
            if paragraph:
                parts.append(f"<p>{paragraph}</p>")
        parts.append("</div>")
        if html_snippet.strip():
            parts.append('<div class="section-visual">')
            parts.append(html_snippet)
            parts.append("</div>")
        parts.append("</section>")

    # Conclusion
    parts.append("<section>")
    parts.append(f"<h2>{conclusion_heading}</h2>")
    for paragraph in conclusion.splitlines():
        paragraph = paragraph.strip()
        if paragraph:
            parts.append(f"<p>{paragraph}</p>")
    parts.append("</section>")

    parts.append("</article>")
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)
# ====== メイン処理 ====================================================


def main() -> None:
    global LOG_START_TIME

    # ロガー初期化（ここでファイルが作成される）
    init_logger()
    log("=== Blog generator started ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    persona = "Panchan"

    # 1) キーワードをランダムに選択（シード指定可）
    seed_env = os.environ.get("BLOG_KEYWORD_SEED")
    seed = int(seed_env) if seed_env and seed_env.isdigit() else None
    keyword_candidates = select_keywords(KEYWORDS, seed=seed)
    if not keyword_candidates:
        log("ERROR: No keywords available.")
        return
    keyword = keyword_candidates[0]
    log(f"Selected keyword: {keyword}")

    # 2) インデックスファイル一覧を取得し、検索でヒットを確認
    index_files = find_index_files(INDEX_ROOT)
    if not index_files:
        log(f"ERROR: No index files found under {INDEX_ROOT}")
        return

    matched_indexes = locate_relevant_indexes(keyword, index_files)
    log(f"Search hits for keyword '{keyword}': {len(matched_indexes)}")
    if not matched_indexes:
        log("No search hits found after fallbacks. Skipping article generation.")
        return

    chosen_index = choose_index_with_phi3(keyword, matched_indexes)
    if not chosen_index:
        log("ERROR: Failed to choose index file.")
        return

    log(f"Chosen index file: {chosen_index}")
    # コンテキスト量を抑えてパフォーマンス改善
    index_text_full = read_text(chosen_index, max_chars=2000)
    knowledge_text = collect_knowledge_text(chosen_index, max_chars=4000)

    rahab_tracks = fetch_rahab_tracks()
    rahab_context = build_rahab_worldview_block(rahab_tracks)
    if rahab_context:
        log(f"Loaded {len(rahab_tracks)} Rahab Punkaholic Girls tracks.")
    else:
        log("WARN: Failed to load Rahab Punkaholic Girls tracks; continuing.")

    # 3) メタ情報を phi3 で生成（日本語）
    log("Generating Japanese metadata with phi3...")
    meta_ja = generate_metadata_with_phi3(keyword, index_text_full, rahab_context)
    meta_ja = normalize_metadata(meta_ja, keyword, description_source=index_text_full[:240])
    if not meta_ja.get("title"):
        log("WARN: metadata title is empty; using keyword as fallback title.")
        meta_ja["title"] = keyword
    log(f"Metadata generated (JA): title={meta_ja.get('title')}")

    overview_ja = str(meta_ja.get("overview") or "")
    if not overview_ja:
        log("WARN: overview is empty, using index snippet as fallback.")
        overview_ja = index_text_full[:800]
        meta_ja["overview"] = overview_ja

    # 4) 7セクション構成を phi3 で生成（日本語）
    log("Generating 7 Japanese sections with phi3...")
    sections_ja = generate_sections_with_phi3(keyword, overview_ja, rahab_context)
    log(f"Generated {len(sections_ja)} sections (JA)")

    # 5) 各セクション本文を生成（日本語）
    section_bodies_ja: List[str] = []
    for i, sec in enumerate(sections_ja):
        log(f"Generating JA body for section {i+1}: {sec.get('title')}")
        body = write_section_body_with_llama3(
            keyword=keyword,
            persona=persona,
            section=sec,
            article_title=str(meta_ja.get("title") or keyword),
            overview=overview_ja,
            index_text=index_text_full,
            knowledge_text=knowledge_text,
            rahab_context=rahab_context,
        )
        section_bodies_ja.append(clean_paragraphs(body))

    # 6) 総論を生成（日本語）
    log("Generating JA conclusion...")
    conclusion_ja = write_conclusion_with_llama3(
        keyword=keyword,
        persona=persona,
        overview=overview_ja,
        all_section_bodies=section_bodies_ja,
        rahab_context=rahab_context,
    )
    conclusion_ja = clean_paragraphs(conclusion_ja, max_chars=1800)

    # 7) 図解 HTML は日本語のみ生成
    log("Generating Japanese section visuals...")
    section_htmls_ja = generate_section_visuals(
        sections_ja, section_bodies_ja, lang_code="ja"
    )

    # 8) HTML とメタ情報を保存（日本語のみ）
    now = dt.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    slug = slugify(keyword)

    html_ja_path = OUTPUT_DIR / f"{timestamp}_{slug}_ja.html"
    json_path = OUTPUT_DIR / f"{timestamp}_{slug}_meta.json"

    log(f"Saving HTML and metadata to {OUTPUT_DIR} ...")

    html_ja = build_html_document(
        lang="ja",
        meta=meta_ja,
        sections=sections_ja,
        section_bodies=section_bodies_ja,
        section_htmls=section_htmls_ja,
        conclusion=conclusion_ja,
    )

    html_ja_path.write_text(html_ja, encoding="utf-8")

    meta_out: Dict[str, Any] = {
        "generated_at": now.isoformat(),
        "keyword": keyword,
        "index_file": str(chosen_index),
        "meta_ja": meta_ja,
        "sections_ja": sections_ja,
    }
    json_path.write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"Saved HTML (JA): {html_ja_path}")
    log(f"Saved JSON meta: {json_path}")

    # 経過時間ログ
    if LOG_START_TIME is not None:
        elapsed = dt.datetime.now() - LOG_START_TIME
        minutes = elapsed.total_seconds() / 60.0
        log(f"=== Blog generator finished. Elapsed: {minutes:.1f} minutes ===")


if __name__ == "__main__":
    main()

