from __future__ import annotations

import argparse
import html
import json
import random
import sys
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

OUTPUT_DIR = Path("/mnt/hgfs/output")
INDEX_PATH = Path("indexes/index.json")
PERSONA_PATH = Path("indexes/mybrain/knowledge.md")
DEFAULT_MODEL = "phi3:mini"
DEFAULT_PROVIDER = "ollama"
DEFAULT_API_BASE = "http://127.0.0.1:11434"
BLOG_TRIGGER_PHRASE = "ブログ書いて"
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
BLOG_WORKFLOW_GUIDE = """
ブログ生成フロー仕様

クライアントから「ブログを書いてほしい」と要望を受けたときに、`client_instruction_responder.py` や `blog_server.py` で LLM に投げるプロンプトの作り方をまとめました。概論は 500 文字前後、各セクション本文は 1,500 文字前後で生成することを前提にしています。

【参照するデータ】
- インデックス: indexes/index.json — セクションのキーとなる文章に近いタイトルやサマリーを探します。
- 知識ベース: indexes/mybrain/knowledge.md — Rahab Punkaholic Girls/Johanne の世界観や読者像を補足します。

【生成手順】
1. 概論 (500 文字程度)
   - `index.json` 全体と `knowledge.md` の要点を読み、ブログ全体の背景と狙いを 500 文字前後の日本語で要約するように LLM に依頼します。
   - 「索引用のエントリを横断し、読者像にフィットする導入にする」と明示すると、セクション本文とのつながりが良くなります。
2. セクション本文 (各 1,500 文字程度)
   - セクション見出しやリード文をキーに `index.json` を照合し、最も近いエントリの `summary` を詳細素材として抽出します。
   - `knowledge.md` から Johanne の嗜好・不安・期待に触れる記述を併用し、読者文脈を補強します。
   - 上記 2 つの素材をひとつのプロンプトにまとめ、1,500 文字前後の本文生成を指示します。構成（主張→根拠→示唆→次節へのブリッジ）を指定すると安定します。

【プロンプト例】
以下のテンプレートを `client_instruction_responder.py` や `blog_server.py` へ渡す `prompt` として利用できます。`{section_title}` と `{section_lead}` を適宜差し替えてください。

あなたは日本語で執筆する編集ライターです。
- 概論はすでに用意済み。これから「{section_title}」セクション本文（約1,500文字）を書いてください。
- セクションの意図: {section_lead}
- 参考インデックス: indexes/index.json から近い要素を要約として活用。
- 読者像と世界観: indexes/mybrain/knowledge.md にある Johanne の嗜好・不安・期待を踏まえる。
- 構成: 主張 → 根拠/事例 → 示唆 → 次節へのブリッジ。
- トーン: パンクな反骨と冷静な分析が同居する語り口。過度な煽りや専門用語の羅列は避ける。

【運用メモ】
- 概論と各セクションを別々に生成し、最終的に一つの HTML に結合します。`client_instruction_responder.py` には `--filename` でファイル名を指定できます。
- LLM への入力が長くなる場合は、`index.json` の該当エントリのみを貼り付けるか、サマリー部分だけを引用してください。
""".strip()
SYSTEM_PROMPT = """
あなたは日本語で簡潔に回答するアシスタントです。依頼内容を整理し、必要に応じて箇条書きでわかりやすくまとめてください。
""".strip()

OPENAI_AVAILABLE = find_spec("openai") is not None
if OPENAI_AVAILABLE:
    from openai import OpenAI
else:  # pragma: no cover - optional dependency hint
    OpenAI = None  # type: ignore[misc]


def _htmlize_text(text: str) -> str:
    """Escape text for HTML and preserve line breaks."""
    escaped = html.escape(text)
    return escaped.replace("\n", "<br/>")


def _load_persona_text() -> str:
    """Read persona information from the local knowledge base if available."""

    try:
        return PERSONA_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "ペルソナ情報は利用できませんでした。"


def _load_index_text() -> str:
    """Return the raw index file content to attach as knowledge."""

    try:
        return INDEX_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _load_index_entries() -> list[dict[str, object]]:
    """Load index entries from the attached directory if present."""

    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def _parse_published(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return datetime.min


def _select_latest_entry(keyword: str) -> dict[str, object] | None:
    """Pick the latest index entry, prioritizing those matching the keyword."""

    entries = _load_index_entries()
    if not entries:
        return None

    keyword_lower = keyword.casefold()

    def matches(entry: dict[str, object]) -> bool:
        haystack = f"{entry.get('title', '')} {entry.get('summary', '')}".casefold()
        return keyword_lower in haystack

    filtered = [entry for entry in entries if matches(entry)]
    candidates = filtered or entries

    candidates.sort(key=lambda entry: _parse_published(entry.get("published")), reverse=True)
    return candidates[0] if candidates else None


def build_blog_generation_prompt(
    original_prompt: str,
    persona_text: str,
    keyword: str,
    latest_entry: dict[str, object] | None,
    index_text: str,
) -> str:
    """Compose the LLM prompt for rich HTML blog requests using persona and latest info."""

    latest_info_block = "最新情報が見つかりませんでした。"
    if latest_entry:
        latest_info_block = "\n".join(
            [
                f"- タイトル: {latest_entry.get('title', '')}",
                f"- URL: {latest_entry.get('url', '')}",
                f"- 要約: {latest_entry.get('summary', '')}",
                f"- 公開日時: {latest_entry.get('published', '')}",
            ]
        )

    index_block = index_text or "インデックスファイルが添付されていません。"

    return (
        "以下の条件でリッチHTMLのブログ記事を作成してください。\n"
        f"依頼内容: {original_prompt}\n"
        f"選択キーワード: {keyword}\n"
        "\n[ペルソナ情報]\n"
        f"{persona_text}\n"
        "\n[最新情報]\n"
        f"{latest_info_block}\n"
        "\n[添付インデックスファイル]\n"
        f"{index_block}\n"
        "\n[図解の使い方]\n"
        "- 3〜5 個の図解を HTML で埋め込み、<figure class=\"diagram\"> で囲んでください。\n"
        "- 導入で 1 つ（記事マップ／概念マップ）、中盤で 1〜2 つ（フロー図・シーケンス図・アーキテクチャ図・比較表・折れ線/棒/円グラフなど）、終盤で 1 つ（リスクマトリクス／要点サマリー／今後の展望ツリー）を配置してください。\n"
        "- 図の候補例: 記事マップ、概念マップ、マインドマップ、フロー図、シーケンス図、全体アーキテクチャ図、インフラ構成図、比較表/レーダーチャート、折れ線グラフ/棒グラフ/円グラフ、ステップ図、リスクマトリクス、Myth vs Fact 図、ペルソナカード、要点サマリー図、今後の展望ツリー。\n"
        "- 図解は純粋な HTML/CSS で簡潔に表現し、内容を短い箇条書きやテーブルで示してください (SVG/Canvas 不要)。\n"
        "\n[出力HTML構造]\n"
        "- <article> 内に収め、プレーン HTML で返してください。\n"
        "- <header> にタイトル、約 200 文字のディスクリプション、タグ 6 つ (ul > li) を配置。\n"
        "- <section class=\"summary\"> に 300〜400 文字のリード文と、重要ポイント 3〜5 件の箇条書きを置いてください。\n"
        "- セクション名を 7 つ使い、各 <section class=\"body\"> に h2 見出し＋本文 350〜500 文字＋ 3 点以内の箇条書きを含め、本文直後に対応する <figure> 図解を挿入。\n"
        "- <footer> にリスクと対処法の要約、今後の展望や ToDo を簡潔にまとめるブロックを置いてください。\n"
        "\n[執筆要件]\n"
        "- すべて日本語で、ペルソナに刺さる語り口にしてください。\n"
        "- 添付インデックスファイルと最新情報を引用し、本文で要約できるように重要トピックを整理してください。\n"
        "- 図解のキャプションに図の種類を明記し、本文の要点と整合させてください。\n"
    )


def generate_openai_completion(prompt: str, model: str) -> tuple[str | None, str | None]:
    """Send the prompt to OpenAI if available and return (content, error)."""
    if not OPENAI_AVAILABLE:
        return None, "openai パッケージが見つかりません。 'pip install openai' を実行してください。"

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
    except Exception as exc:  # pragma: no cover - network/API specific
        return None, str(exc)

    message = response.choices[0].message.content if response.choices else None
    return message.strip() if message else "", None


def generate_ollama_completion(
    prompt: str, model: str, api_base: str
) -> tuple[str | None, str | None]:
    """Send the prompt to Ollama's /api/chat endpoint and return (content, error)."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        request = Request(
            url=f"{api_base.rstrip('/')}/api/chat",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        with urlopen(request) as response:
            data = json.loads(response.read())
    except URLError as exc:  # pragma: no cover - depends on runtime environment
        return None, f"Ollama への接続に失敗しました: {exc.reason}"
    except Exception as exc:  # pragma: no cover - network/API specific
        return None, str(exc)

    message = None
    if isinstance(data, dict):
        message = data.get("message", {}).get("content") or data.get("response")

    if message is None:
        return None, "Ollama から有効な応答が得られませんでした。"

    return str(message).strip(), None


def build_html(prompt: str, completion: str, model: str, error: str | None = None) -> str:
    """Compose an HTML document that shows the prompt and LLM response."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    error_block = (
        f'<p class="error">LLM エラー: {html.escape(error)}</p>' if error else ""
    )
    return f"""<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"UTF-8\" />
  <title>LLM 応答</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 1.5rem; color: #1f2933; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    .meta {{ color: #52606d; font-size: 0.95rem; margin-bottom: 1rem; }}
    section {{ margin-bottom: 1.25rem; }}
    pre {{ background: #f5f7fa; padding: 0.75rem; border-radius: 8px; white-space: pre-wrap; word-break: break-word; }}
    .response {{ background: #f0f4ff; padding: 0.75rem; border-radius: 8px; line-height: 1.6; }}
    .error {{ color: #b42318; font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <h1>LLM 応答</h1>
    <p class=\"meta\">モデル: {html.escape(model)} / 出力時刻 (UTC): {timestamp}</p>
  </header>
  <main>
    <section>
      <h2>依頼内容</h2>
      <pre>{html.escape(prompt)}</pre>
    </section>
    <section>
      <h2>生成結果</h2>
      <div class=\"response\">{_htmlize_text(completion)}</div>
      {error_block}
    </section>
  </main>
</body>
</html>
"""


def respond_to_instruction(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    api_base: str = DEFAULT_API_BASE,
    filename: str | None = None,
) -> dict[str, object]:
    """Generate a response HTML document and save it to disk.

    Returns a dict containing success flag, HTML content, output path, and error message.
    """

    prompt_text = prompt.strip()
    if not prompt_text:
        raise ValueError("Prompt text is required.")

    final_prompt = prompt_text
    if BLOG_TRIGGER_PHRASE in prompt_text:
        keyword = random.choice(BLOG_KEYWORDS)
        persona_text = _load_persona_text()
        latest_entry = _select_latest_entry(keyword)
        index_text = _load_index_text()
        final_prompt = build_blog_generation_prompt(
            prompt_text, persona_text, keyword, latest_entry, index_text
        )

    selected_provider = provider or DEFAULT_PROVIDER
    selected_model = model or DEFAULT_MODEL
    selected_api_base = api_base or DEFAULT_API_BASE

    if selected_provider == "ollama":
        completion, error = generate_ollama_completion(
            final_prompt, selected_model, selected_api_base
        )
    else:
        completion, error = generate_openai_completion(final_prompt, selected_model)

    if completion is None:
        completion = "LLM 応答を生成できませんでした。詳細は error を確認してください。"

    html_document = build_html(final_prompt, completion, selected_model, error)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_filename = filename or f"client_response_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.html"
    output_path = OUTPUT_DIR / resolved_filename
    output_path.write_text(html_document, encoding="utf-8")

    return {
        "ok": error is None,
        "html": html_document,
        "path": str(output_path),
        "filename": resolved_filename,
        "model": selected_model,
        "provider": selected_provider,
        "error": error,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "クライアントの指示を LLM に投げ、結果を /mnt/hgfs/output 以下に HTML で保存します。"
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="クライアントの指示文。省略時は標準入力を参照します。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"使用するモデル名 (デフォルト: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=DEFAULT_PROVIDER,
        help=(
            "使用する推論エンジン。Ollama で phi3:mini (タイトル/タグ/セクション名)"
            " と llama3:8b (本文/図解/総論) を使う運用を想定しています。"
        ),
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="Ollama を利用する場合のベース URL (デフォルト: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--filename",
        help="出力 HTML ファイル名。省略時はタイムスタンプ付きで保存します。",
    )
    return parser.parse_args(argv)


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text

    raise SystemExit("プロンプトが指定されていません。引数または標準入力で渡してください。")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prompt_text = read_prompt(args)
    result = respond_to_instruction(
        prompt_text,
        model=args.model,
        provider=args.provider,
        api_base=args.api_base,
        filename=args.filename,
    )

    print(f"HTML を {result['path']} に出力しました。")
    if result.get("error"):
        print(f"LLM 呼び出しで警告が発生しました: {result['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
