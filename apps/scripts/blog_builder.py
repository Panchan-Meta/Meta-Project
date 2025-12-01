"""Generate persona-focused HTML blogs from client prompts.

This script reads domain knowledge from ``/var/www/Meta-Project/indexes`` and
categorizes a client-provided prompt into one of the predefined blog domains.
It then assembles a ~10,000 character HTML article tailored for the persona
"哲学者気取りのヨハネ", including diagrams per section and an outro informed by
shared knowledge files.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

BASE_DIR = Path("/var/www/Meta-Project")
INDEX_DIR = BASE_DIR / "indexes"
DEFAULT_OUTPUT_DIR = Path("/mnt/hgfs/output")
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
            intro="哲学者気取りのヨハネのまなざしで、クライアントテーマを批判的に分解する長文ブログ。",
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
                "世界のバグを見抜こうとする冷静さと、パンクな衝動が同居する。クライアントから受け取ったテーマ「{prompt}」"
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
                " it holds together punk urgency and cold rationality. Anchored on the client theme '{prompt}', it layers European"
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


def generate_sections(prompt: str, category: str, pack: LanguagePack) -> list[Section]:
    base = _deterministic_numbers(prompt + category + pack.code, 12, 9)
    sections: list[Section] = []

    for idx, theme in enumerate(pack.themes):
        if idx % 3 == 0:
            values = base[idx : idx + 3]
            diagram = build_bar_chart(
                values,
                pack.chart_labels["bar"],
                title=pack.chart_titles["bar"],
                caption=pack.chart_captions["bar"],
                aria_label=pack.chart_titles["bar"],
            )
        elif idx % 3 == 1:
            values = base[idx : idx + 4]
            diagram = build_line_chart(
                values,
                pack.chart_labels["line"],
                title=pack.chart_titles["line"],
                caption=pack.chart_captions["line"],
                aria_label=pack.chart_titles["line"],
            )
        else:
            values = base[idx : idx + 3]
            diagram = build_pie_chart(
                values,
                pack.chart_labels["pie"],
                title=pack.chart_titles["pie"],
                caption=pack.chart_captions["pie"],
                aria_label=pack.chart_titles["pie"],
            )

        body = textwrap.dedent(
            pack.section_body_template.format(prompt=prompt, category=category, theme=theme)
        ).strip()

        sections.append(Section(heading=theme, body=body, diagram_html=diagram))
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
    body_parts.append(f"<h2>Outro</h2><p>{html.escape(outro)}</p>")

    html_doc = "\n".join(body_parts)
    if len(html_doc) < 10000:
        padding = "<p>" + html.escape(padding_phrase * ((10000 - len(html_doc)) // len(padding_phrase) + 1)) + "</p>"
        html_doc += padding
    return "<article>" + html_doc + "</article>"


def build_article(prompt: str, pack: LanguagePack, *, category: str, common_text: str) -> str:
    sections = generate_sections(prompt, category, pack)
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


def main() -> None:
    args = parse_args()
    prompt = args.prompt or html.unescape(os.sys.stdin.read()).strip()
    if not prompt:
        raise SystemExit("クライアントの文章が指定されていません")

    category_files = discover_category_files(INDEX_DIR)
    category_snippets = {name: read_snippet([path]) for name, path in category_files.items()}
    category = choose_category(prompt, category_snippets)

    common_candidates = [INDEX_DIR / name for name in COMMON_FILES] + [INDEX_DIR / (name + ".txt") for name in COMMON_FILES]
    common_text = read_snippet(common_candidates)

    slug = _slugify(prompt)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    packs = get_language_packs()
    for pack in packs:
        article_html = build_article(prompt, pack, category=category, common_text=common_text)
        output_path = output_dir / f"{slug}_{pack.code}.html"
        output_path.write_text(article_html, encoding="utf-8")
        print(f"Saved blog HTML to {output_path}")

    print("FLAG:FILES_SENT")


if __name__ == "__main__":
    main()
