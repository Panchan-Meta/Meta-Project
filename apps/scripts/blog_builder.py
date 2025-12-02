"""Generate persona-focused HTML blogs from client prompts.

This script reads domain knowledge from ``/var/www/Meta-Project/indexes`` and
categorizes a client-provided prompt into one of the predefined blog domains.
It then assembles a ~10,000 character HTML article tailored for the persona
"哲学者気取りのヨハネ", including diagrams per section and an outro informed by
shared knowledge files.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import os
import re
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import requests

BASE_DIR = Path("/var/www/Meta-Project")
INDEX_DIR = BASE_DIR / "indexes"
DEFAULT_OUTPUT_DIR = Path("/mnt/hgfs/output")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_TIMEOUT = 900
SCRIPT_CATEGORY_FILES = [
    "Web3",
    "NFT",
    "Cryptocurrency",
    "DeFi",
    "Blockchain & Cryptocurrency Security",
    "Cybersecurity",
    "Hacking",
    "Geopolitics",
    "Mental Health",
    "Dystopian",
]
COMMON_FILES = ["mybrain", "catholic", "bible", "berkshire"]


@dataclass
class LanguagePack:
    code: str
    title_template: str
    description: str
    intro: str
    themes: list[str]
    section_body_template: str
    chart_labels: dict[str, list[str]]
    chart_titles: dict[str, str]
    chart_captions: dict[str, str]
    outro_lines: list[str]
    padding_phrase: str


def get_language_packs() -> list[LanguagePack]:
    return [
        LanguagePack(
            code="ja",
            title_template="{category}で世界のバグを解体する夜",
            description="リスクと静けさを両立させたいヨハネに向けた、現実と物語の橋渡し。",
            intro="哲学者気取りのヨハネのまなざしで、与えられたテーマを批判的に分解する長文ブログ。",
            themes=[
                "市場とテクノロジーの歪さを読み解く",
                "リスクとリターンのバランス設計",
                "コミュニティと倫理の接点",
                "個人メンタルと情報ダイエット",
                "アートと物語への投資戦略",
                "未来への問いと実装ロードマップ",
            ],
            section_body_template=(
                "哲学者気取りのヨハネの視点から、{category}領域に潜む課題を具体的に検証する。{theme}という切り口で、"
                "世界のバグを見抜こうとする冷静さと、パンクな衝動が同居する。与えられたテーマ「{prompt}」"
                "を軸に、欧州の地政学リスク、ドル覇権、分散型テクノロジーの進化を重ね合わせ、盲目的な熱狂ではなく"
                "自分で考え抜くためのフレームを組み立てる。ヨハネが夜のカフェでチャートを眺めながら感じる虚無感を、"
                "データ、ストーリー、リスク設計という3本柱で受け止め、同じ違和感を持つ読者に静かな伴走を提供する。"
            ),
            chart_labels={
                "bar": ["技術", "市場", "文化"],
                "line": ["短期", "中期", "長期", "その先"],
                "pie": ["倫理", "経済", "創造"],
            },
            chart_titles={
                "bar": "関心強度の棒グラフ",
                "line": "リスク認識の推移",
                "pie": "意思決定の内訳",
            },
            chart_captions={
                "bar": "ヨハネの関心の強さを示す仮想データ。",
                "line": "時間とともに変動するリスク認識をモデル化。",
                "pie": "精神・経済・倫理のバランス感覚を擬似データで可視化。",
            },
            outro_lines=[
                "世界が揺らいでも、盲目的な信仰に逃げ込まない態度こそが次の実験を開く。",
                "Rahabのステージは、批判も迷いも歓迎するサンドボックスであるべきだ。",
                "だからこそ、疑い続けることと信じてみることの両立を手放さないでほしい。",
            ],
            padding_phrase="ヨハネの内省を深掘りし、同じ問いを持つ読者に長文で伴走する。",
        ),
        LanguagePack(
            code="en",
            title_template="Dissecting the world's glitches through {category}",
            description="A bridge between reality and narrative for Yohane, balancing risk with quiet focus.",
            intro="From Yohane's restless yet analytical gaze, this article deconstructs the client theme without blind hype.",
            themes=[
                "Reading the distortions of markets and technology",
                "Designing the balance of risk and return",
                "Where community meets ethics",
                "Mental health and information fasting",
                "Investing in art and narrative",
                "Questions for the future and a working roadmap",
            ],
            section_body_template=(
                "From Yohane's perspective, this piece examines the hidden challenges in {category}. By focusing on {theme},"
                " it holds together punk urgency and cold rationality. It layers European"
                " geopolitics, dollar hegemony, and decentralized tech progress. The goal is a frame to think for yourself, not"
                " to be swept away by hype. Late-night charts in a café, the unease of volatility, and the trio of data, story,"
                " and risk design all converge to accompany readers who share the same dissonance."
            ),
            chart_labels={
                "bar": ["Tech", "Markets", "Culture"],
                "line": ["Short", "Mid", "Long", "Beyond"],
                "pie": ["Ethics", "Economy", "Creation"],
            },
            chart_titles={
                "bar": "Interest intensity",
                "line": "Risk perception over time",
                "pie": "How decisions are weighted",
            },
            chart_captions={
                "bar": "Imaginary distribution of Yohane's focus across domains.",
                "line": "A modeled curve of how risk perception moves with time.",
                "pie": "A synthetic balance among ethics, economy, and creativity.",
            },
            outro_lines=[
                "Even when the world shakes, refusing blind faith opens the next experiment.",
                "Rahab's stage should stay a sandbox that welcomes both doubt and critique.",
                "Keep holding both skepticism and the courage to believe—at the same time.",
            ],
            padding_phrase="Yohane keeps unpacking his unease, walking readers through long-form reflection.",
        ),
        LanguagePack(
            code="it",
            title_template="Smontare i bug del mondo con {category}",
            description="Un ponte tra realtà e narrazione per Yohane, in equilibrio tra rischio e quiete.",
            intro="Con lo sguardo inquieto ma analitico di Yohane, questo articolo smonta il tema del cliente senza farsi trascinare dall'hype.",
            themes=[
                "Leggere le distorsioni di mercato e tecnologia",
                "Progettare l'equilibrio tra rischio e ritorno",
                "Dove comunità ed etica si toccano",
                "Salute mentale e dieta informativa",
                "Investire in arte e narrazione",
                "Domande sul futuro e roadmap di lavoro",
            ],
            section_body_template=(
                "Dal punto di vista di Yohane, esploriamo le sfide nascoste in {category}. Con {theme} come lente,"
                " convivono urgenza punk e razionalità fredda. Il tema del cliente '{prompt}' fa da perno per intrecciare"
                " geopolitica europea, egemonia del dollaro e progresso delle tecnologie decentralizzate. L'obiettivo è"
                " costruire un frame per pensare in autonomia, non inseguire l'entusiasmo cieco. I grafici notturni al caffè,"
                " l'inquietudine della volatilità e il trio dati-narrazione-design del rischio guidano i lettori con la stessa dissonanza."
            ),
            chart_labels={
                "bar": ["Tecnologia", "Mercati", "Cultura"],
                "line": ["Breve", "Medio", "Lungo", "Oltre"],
                "pie": ["Etica", "Economia", "Creazione"],
            },
            chart_titles={
                "bar": "Intensità dell'interesse",
                "line": "Percezione del rischio nel tempo",
                "pie": "Composizione delle decisioni",
            },
            chart_captions={
                "bar": "Distribuzione immaginaria dell'attenzione di Yohane tra i domini.",
                "line": "Una curva modellata di come cambia la percezione del rischio.",
                "pie": "Bilanciamento sintetico tra etica, economia e creatività.",
            },
            outro_lines=[
                "Anche quando il mondo trema, rifiutare la fede cieca apre il prossimo esperimento.",
                "Il palco di Rahab deve restare un sandbox che accoglie dubbio e critica.",
                "Continua a tenere insieme scetticismo e coraggio di credere, allo stesso tempo.",
            ],
            padding_phrase="Yohane approfondisce la sua inquietudine accompagnando i lettori con una lunga riflessione.",
        ),
    ]


PERSONA = textwrap.dedent(
    """
    ペルソナ: 哲学者気取りのヨハネ（25歳、北イタリア在住のFintech系エンジニア）
    - 感情の波が大きく、合理性と虚無感の間を揺れる。
    - Web3/NFT/DeFi/AIなどの前衛技術と、世界の歪さへの哲学的違和感を持つ。
    - パンクロックやサイバーパンク的アートに惹かれ、リスクとリターンのバランスを常に探っている。
    - 盲目的な信仰を避け、自分で考え抜いた上で頼れる拠りどころを模索している。
    """
)


@dataclass
class Section:
    heading: str
    body: str
    diagram_html: str
    chart_note: str


class StatusReporter:
    """Periodically records the active program/function for clients."""

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
    STATUS_REPORTER.update(program=program, function=function_name, file=file)
    try:
        yield
    finally:
        STATUS_REPORTER.update(program=program, function="idle", file=file)


def _post_llm(model: str, prompt: str, *, system: str | None = None) -> str:
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[LLM error: {exc}]"


def warm_model(model: str) -> None:
    with status_scope(f"warm_model:{model}"):
        _post_llm(model, prompt="warmup ping")


def read_snippet(path_candidates: Iterable[Path]) -> str:
    """Return concatenated snippets from available files, ignore missing ones."""

    snippets: list[str] = []
    for path in path_candidates:
        if path.is_file():
            try:
                snippets.append(path.read_text(encoding="utf-8"))
            except OSError:
                continue
    return "\n\n".join(snippets)


def discover_category_files(index_dir: Path) -> Mapping[str, Path]:
    """Map canonical category names to existing files under the index directory."""

    mapping: dict[str, Path] = {}
    if not index_dir.exists():
        return mapping

    for path in index_dir.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        for name in SCRIPT_CATEGORY_FILES:
            if stem.lower().startswith(name.lower().replace(" & ", " ")):
                mapping[name] = path
    return mapping


def choose_category(prompt: str, category_snippets: Mapping[str, str]) -> str:
    """Choose the best matching category based on keyword overlap."""

    normalized = prompt.lower()
    scores: dict[str, int] = {}
    for category, snippet in category_snippets.items():
        score = 0
        for token in category.lower().split():
            if token and token in normalized:
                score += 2
        for token in set(snippet.lower().split()):
            if token in normalized:
                score += 1
        scores[category] = score

    if not scores:
        return "Web3"
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Web3"
    return best


def classify_with_llm(prompt: str, category_snippets: Mapping[str, str]) -> str | None:
    """Use the LLM once to propose a category based on provided snippets."""

    category_list = "\n".join(f"- {name}" for name in SCRIPT_CATEGORY_FILES)
    snippets = "\n".join(
        f"{name}: {snippet[:800]}" for name, snippet in category_snippets.items()
    )
    system = (
        "あなたは分類器です。以下のカテゴリ一覧から最も近い1つを日本語で返してください。"
        "カテゴリ名のみを返し、説明文は不要です。"
    )
    prompt_body = textwrap.dedent(
        f"""
        クライアントからの文章:
        {prompt}

        利用可能なカテゴリ:
        {category_list}

        参考スニペット:
        {snippets or '（スニペットなし）'}
        """
    )
    llm_answer = _post_llm("phi3:mini", prompt_body, system=system)
    if not llm_answer:
        return None

    for name in SCRIPT_CATEGORY_FILES:
        if name.lower() in llm_answer.lower():
            return name
    return None


def _deterministic_numbers(prompt: str, count: int, scale: int) -> list[int]:
    digest = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).digest()
    numbers = [b % scale for b in digest[:count]]
    return [n + 1 for n in numbers]


def build_bar_chart(values: list[int], labels: list[str], *, title: str, caption: str, aria_label: str) -> str:
    max_val = max(values) if values else 1
    width = 500
    height = 200
    bar_width = width // max(len(values), 1)
    bars = []
    for i, value in enumerate(values):
        bar_height = int((value / max_val) * (height - 30))
        x = i * bar_width + 10
        y = height - bar_height - 20
        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_width - 20}" height="{bar_height}" fill="#7f5af0" />'
            f'<text x="{x + (bar_width - 20) / 2}" y="{height - 5}" font-size="12" text-anchor="middle" fill="#16161a">{html.escape(labels[i])}</text>'
        )
    return (
        f"<figure><svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(aria_label)}'>"
        f"<title>{html.escape(title)}</title>{''.join(bars)}</svg>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def build_line_chart(values: list[int], labels: list[str], *, title: str, caption: str, aria_label: str) -> str:
    width = 500
    height = 220
    if not values:
        values = [1, 1, 1]
    max_val = max(values)
    points = []
    step = width // max(len(values) - 1, 1)
    for i, value in enumerate(values):
        x = i * step
        y = height - 20 - int((value / max_val) * (height - 40))
        points.append(f"{x},{y}")
    polyline = " ".join(points)
    circles = [f"<circle cx='{p.split(',')[0]}' cy='{p.split(',')[1]}' r='4' fill='#2cb67d'/>" for p in points]
    texts = [
        f"<text x='{i * step}' y='{height - 5}' font-size='12' text-anchor='start' fill='#16161a'>{html.escape(label)}</text>"
        for i, label in enumerate(labels)
    ]
    return (
        f"<figure><svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(aria_label)}'>"
        f"<title>{html.escape(title)}</title><polyline fill='none' stroke='#2cb67d' stroke-width='3' points='{polyline}'/>"
        f"{''.join(circles)}{''.join(texts)}</svg>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def build_pie_chart(values: list[int], labels: list[str], *, title: str, caption: str, aria_label: str) -> str:
    total = sum(values) or 1
    radius = 80
    cx = cy = 100
    start_angle = 0.0
    slices = []
    colors = ["#7f5af0", "#2cb67d", "#ff8906", "#e53170", "#94a1b2", "#0f0"]
    for i, (value, label) in enumerate(zip(values, labels)):
        angle = (value / total) * 360
        end_angle = start_angle + angle
        x1 = cx + radius * _cos_deg(start_angle)
        y1 = cy + radius * _sin_deg(start_angle)
        x2 = cx + radius * _cos_deg(end_angle)
        y2 = cy + radius * _sin_deg(end_angle)
        large_arc = 1 if angle > 180 else 0
        path_d = (
            f"M {cx},{cy} L {x1},{y1} A {radius},{radius} 0 {large_arc} 1 {x2},{y2} Z"
        )
        color = colors[i % len(colors)]
        slices.append(
            f"<path d='{path_d}' fill='{color}' stroke='#16161a' stroke-width='1' />"
            f"<text x='{cx + radius + 20}' y='{20 + i * 18}' font-size='12' fill='#16161a'>{html.escape(label)} ({value})</text>"
        )
        start_angle = end_angle
    return (
        f"<figure><svg viewBox='0 0 260 200' role='img' aria-label='{html.escape(aria_label)}'>"
        f"<title>{html.escape(title)}</title>"
        f"{''.join(slices)}</svg>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _sin_deg(angle: float) -> float:
    import math

    return math.sin(math.radians(angle))


def _cos_deg(angle: float) -> float:
    import math

    return math.cos(math.radians(angle))


def _is_llm_error(text: str | None) -> bool:
    return not text or text.startswith("[LLM error")


def _fallback_section_summary(
    *,
    prompt: str,
    category: str,
    theme: str,
    pack: LanguagePack,
    values: list[int],
    labels: list[str],
    category_snippet: str,
    chart_title: str,
) -> str:
    formatted_values = ", ".join(f"{label}:{value}" for label, value in zip(labels, values))
    snippet_hint = category_snippet[:60] + ("…" if len(category_snippet) > 60 else "")
    if pack.code == "ja":
        return textwrap.dedent(
            f"""
            テーマ「{theme}」に沿って、{category}領域の論点「{prompt}」を再構成する。
            図「{chart_title}」のデータ（{formatted_values}）を手掛かりに、ヨハネは強弱の差からリスク配分を読み直す。
            参考スニペット: {snippet_hint or 'N/A'}。
            パンクな疑いと静かな洞察を両立させ、見出し通りの論点に引き戻す。
            """
        ).strip()

    if pack.code == "it":
        return textwrap.dedent(
            f"""
            Con il tema "{theme}" rilegge il prompt "{prompt}" nel contesto {category}.
            I dati del grafico "{chart_title}" ({formatted_values}) mostrano dove l'attenzione e il rischio cambiano intensità.
            Estratto di riferimento: {snippet_hint or 'N/A'}.
            Il tono resta sobrio e critico, così da rispettare titolo e diagramma.
            """
        ).strip()

    return textwrap.dedent(
        f"""
        Using the theme "{theme}", Yohane reframes the client idea "{prompt}" inside the {category} lens.
        The chart "{chart_title}" with values {formatted_values} anchors the section to the heading instead of repeating boilerplate.
        Reference snippet: {snippet_hint or 'N/A'}.
        The stance stays analytical and skeptical while following what the title and diagram demand.
        """
    ).strip()


def summarize_section_with_llm(
    prompt: str,
    category: str,
    theme: str,
    pack: LanguagePack,
    category_snippet: str,
    *,
    values: list[int],
    labels: list[str],
    chart_title: str,
) -> str:
    if pack.code == "ja":
        system = (
            "あなたは日本語のブログ執筆アシスタントです。出力は日本語のみで、翻訳や英訳の挿入は禁止。"
            "『クライアント』という語を使わず、一般公開向けに、冷静で批判的かつ少しパンクにまとめる。"
        )
        body = textwrap.dedent(
            f"""
            見出し「{theme}」を3〜4文で要約してください。英訳・翻訳注記は不要です。
            カテゴリ: {category}
            テーマ文: {prompt}
            参考スニペット抜粋:
            {category_snippet[:1200] or 'N/A'}
            図のタイトル: {chart_title}
            ラベルと値: {list(zip(labels, values))}
            語調: 冷静で批判的、かすかなパンクさを含める。
            """
        )
    else:
        system = (
            "You are a structured blog assistant. Keep the language aligned to the user locale."
        )
        body = textwrap.dedent(
            f"""
            Write 3-4 sentences in {pack.code} summarizing the theme for persona Yohane.
            Theme keyword: {theme}
            Chosen category: {category}
            Prompt text: {prompt}
            Category reference (trimmed):
            {category_snippet[:1200] or 'N/A'}
            Chart title: {chart_title}
            Chart labels and values: {list(zip(labels, values))}
            Keep it analytical, skeptical, and reflective.
            """
        )

    llm_text = _post_llm("llama3:8b", body, system=system)
    if _is_llm_error(llm_text):
        return _fallback_section_summary(
            prompt=prompt,
            category=category,
            theme=theme,
            pack=pack,
            values=values,
            labels=labels,
            category_snippet=category_snippet,
            chart_title=chart_title,
        )
    return llm_text


def explain_chart_with_llm(
    values: list[int],
    labels: list[str],
    theme: str,
    pack: LanguagePack,
    *,
    chart_title: str,
) -> str:
    if pack.code == "ja":
        system = (
            "あなたは日本語でグラフを簡潔に解説するアシスタントです。出力は日本語のみで翻訳注記は禁止。"
        )
        prompt_body = textwrap.dedent(
            f"""
            以下の擬似データを持つ図を2文で解説してください。英訳・翻訳の併記は禁止です。
            テーマ: {theme}
            図のタイトル: {chart_title}
            ラベルと値: {list(zip(labels, values))}
            トーン: 冷静で批判的、わずかにパンク。
            """
        )
    else:
        system = "Provide concise chart commentary matching the locale."
        prompt_body = textwrap.dedent(
            f"""
            Summarize the following synthetic chart insightfully in {pack.code}. Keep it to 2 sentences.
            Theme: {theme}
            Chart title: {chart_title}
            Labels and values: {list(zip(labels, values))}
            Persona tone: calm, critical, and slightly punk.
            """
        )
    llm_text = _post_llm("llama3:8b", prompt_body, system=system)
    if _is_llm_error(llm_text):
        if pack.code == "ja":
            return (
                f"図「{chart_title}」は{theme}の文脈で、{', '.join(labels)}のバランスを値{values}として示す。"
                " 極端な値の差を冷静に読み取り、見出しの問いに沿って解釈する。"
            )
        if pack.code == "it":
            return (
                f"Il grafico '{chart_title}' per '{theme}' mette a confronto {', '.join(labels)} con valori {values}."
                " Le differenze evidenziano dove attenzione e rischio vanno calibrati."
            )
        return (
            f"The chart '{chart_title}' ties the theme '{theme}' to the {labels} weights {values},"
            " highlighting where Yohane would lean in or step back."
        )
    return llm_text


def generate_sections(
    prompt: str, category: str, pack: LanguagePack, *, category_snippet: str
) -> list[Section]:
    with status_scope(f"generate_sections:{pack.code}"):
        base = _deterministic_numbers(prompt + category + pack.code, 12, 9)
        sections: list[Section] = []

        for idx, theme in enumerate(pack.themes):
            if idx % 3 == 0:
                values = base[idx : idx + 3]
                labels = pack.chart_labels["bar"]
                diagram = build_bar_chart(
                    values,
                    labels,
                    title=pack.chart_titles["bar"],
                    caption=pack.chart_captions["bar"],
                    aria_label=pack.chart_titles["bar"],
                )
            elif idx % 3 == 1:
                values = base[idx : idx + 4]
                labels = pack.chart_labels["line"]
                diagram = build_line_chart(
                    values,
                    labels,
                    title=pack.chart_titles["line"],
                    caption=pack.chart_captions["line"],
                    aria_label=pack.chart_titles["line"],
                )
            else:
                values = base[idx : idx + 3]
                labels = pack.chart_labels["pie"]
                diagram = build_pie_chart(
                    values,
                    labels,
                    title=pack.chart_titles["pie"],
                    caption=pack.chart_captions["pie"],
                    aria_label=pack.chart_titles["pie"],
                )

            body = textwrap.dedent(
                pack.section_body_template.format(prompt=prompt, category=category, theme=theme)
            ).strip()

            chart_title = pack.chart_titles[
                "bar" if idx % 3 == 0 else "line" if idx % 3 == 1 else "pie"
            ]
            llm_summary = summarize_section_with_llm(
                prompt,
                category,
                theme,
                pack,
                category_snippet,
                values=values,
                labels=labels,
                chart_title=chart_title,
            )
            chart_note = explain_chart_with_llm(
                values,
                labels,
                theme,
                pack,
                chart_title=chart_title,
            )

            sections.append(
                Section(
                    heading=theme,
                    body=body + "\n" + llm_summary,
                    diagram_html=diagram,
                    chart_note=chart_note,
                )
            )
        return sections


def build_outro(common_text: str, pack: LanguagePack) -> str:
    outro_parts = list(pack.outro_lines)
    if common_text:
        outro_parts.append(common_text)
    return "\n".join(outro_parts)


def compose_html(title: str, description: str, intro: str, sections: list[Section], outro: str, *, padding_phrase: str) -> str:
    body_parts = [
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='description'>{html.escape(description)}</p>",
        f"<p class='intro'>{html.escape(intro)}</p>",
    ]
    for section in sections:
        body_parts.append(f"<h2>{html.escape(section.heading)}</h2>")
        body_parts.append(f"<p>{html.escape(section.body)}</p>")
        body_parts.append(section.diagram_html)
        body_parts.append(f"<p class='chart-note'>{html.escape(section.chart_note)}</p>")
    body_parts.append(f"<h2>Outro</h2><p>{html.escape(outro)}</p>")

    html_doc = "\n".join(body_parts)
    if len(html_doc) < 10000:
        padding = "<p>" + html.escape(padding_phrase * ((10000 - len(html_doc)) // len(padding_phrase) + 1)) + "</p>"
        html_doc += padding
    return "<article>" + html_doc + "</article>"


def build_article(
    prompt: str,
    pack: LanguagePack,
    *,
    category: str,
    common_text: str,
    category_snippet: str,
) -> str:
    with status_scope(f"build_article:{pack.code}"):
        sections = generate_sections(prompt, category, pack, category_snippet=category_snippet)
        title = pack.title_template.format(category=category or "")
        outro = build_outro(common_text, pack)
        return compose_html(
            title,
            pack.description,
            pack.intro,
            sections,
            outro,
            padding_phrase=pack.padding_phrase,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate persona-focused blog HTML from a prompt.")
    parser.add_argument("prompt", nargs="?", help="Client-provided theme or text. If omitted, read from stdin.")
    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the generated HTML files for each language.",
    )
    return parser.parse_args()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return slug[:64] or "blog"


def generate_blogs(prompt: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Generate and persist multilingual blogs, returning metadata and HTML bodies."""

    with status_scope("generate_blogs"):
        warm_model("phi3:mini")
        warm_model("llama3:8b")

        category_files = discover_category_files(INDEX_DIR)
        category_snippets = {name: read_snippet([path]) for name, path in category_files.items()}
        category = classify_with_llm(prompt, category_snippets) or choose_category(prompt, category_snippets)

        common_candidates = [INDEX_DIR / name for name in COMMON_FILES] + [INDEX_DIR / (name + ".txt") for name in COMMON_FILES]
        common_text = read_snippet(common_candidates)

        slug = _slugify(prompt)
        output_dir.mkdir(parents=True, exist_ok=True)

        packs = get_language_packs()
        results: dict[str, object] = {
            "category": category,
            "flag": "FLAG:FILES_SENT",
            "slug": slug,
            "files": {},
            "html": {},
        }

        for pack in packs:
            article_html = build_article(
                prompt,
                pack,
                category=category,
                common_text=common_text,
                category_snippet=category_snippets.get(category, ""),
            )
            output_path = output_dir / f"{slug}_{pack.code}.html"
            output_path.write_text(article_html, encoding="utf-8")
            results["files"][pack.code] = str(output_path)
            results["html"][pack.code] = article_html

        return results


def main() -> None:
    args = parse_args()
    prompt = args.prompt or html.unescape(os.sys.stdin.read()).strip()
    if not prompt:
        raise SystemExit("クライアントの文章が指定されていません")

    results = generate_blogs(prompt, output_dir=args.output_dir)
    for code, path in results.get("files", {}).items():
        print(f"Saved blog HTML ({code}) to {path}")

    print(results.get("flag", "FLAG:FILES_SENT"))


if __name__ == "__main__":
    main()
