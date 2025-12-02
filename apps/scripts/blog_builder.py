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
import json
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
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
MIN_SECTION_CHARS = int(os.environ.get("MIN_SECTION_CHARS", "750"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "900"))
SECTION_MODEL = os.environ.get("BLOG_BUILDER_SECTION_MODEL", "phi3:mini")
CLASSIFIER_MODEL = os.environ.get("BLOG_BUILDER_CLASSIFIER_MODEL", "phi3:mini")
ENABLE_CHART_LLM = os.environ.get("BLOG_BUILDER_CHART_LLM", "0") == "1"
WARM_MODELS = os.environ.get("BLOG_BUILDER_WARM_MODELS", "0") == "1"
CHART_MODEL = os.environ.get("BLOG_BUILDER_CHART_MODEL", SECTION_MODEL)
LLM_CALL_COUNT = 0
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
    context_line: str
    summary_heading: str
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
                "世界のバグを見抜こうとする冷静さと、パンクな衝動が同居する。{context}ヨハネが夜のカフェでチャートを"
                "眺めながら感じる虚無感を、データ、ストーリー、リスク設計という3本柱で受け止め、同じ違和感を持つ"
                "読者に静かな伴走を提供する。"
            ),
            context_line=(
                "与えられたテーマを軸に、欧州の地政学リスク、ドル覇権、分散型テクノロジーの進化を重ね合わせ、"
                "盲目的な熱狂ではなく自分で考え抜くためのフレームを組み立てる。"
            ),
            summary_heading="総論",
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
            intro="From Yohane's restless yet analytical gaze, this article deconstructs the theme without blind hype.",
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
                " it holds together punk urgency and cold rationality. {context} Late-night charts in a café, the unease of"
                " volatility, and the trio of data, story, and risk design all converge to accompany readers who share the"
                " same dissonance."
            ),
            context_line=(
                "It layers European geopolitics, dollar hegemony, and decentralized tech progress to offer a thinking frame"
                " that resists blind hype."
            ),
            summary_heading="Conclusion",
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
            intro="Con lo sguardo inquieto ma analitico di Yohane, questo articolo smonta il tema senza farsi trascinare dall'hype.",
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
                " convivono urgenza punk e razionalità fredda. {context} I grafici notturni al caffè, l'inquietudine della"
                " volatilità e il trio dati-narrazione-design del rischio guidano i lettori con la stessa dissonanza."
            ),
            context_line=(
                "Intreccia geopolitica europea, egemonia del dollaro e progresso delle tecnologie decentralizzate per"
                " costruire un frame di pensiero autonomo lontano dall'entusiasmo cieco."
            ),
            summary_heading="Conclusione",
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


def _post_llm(model: str, prompt: str, *, system: str | None = None, track_call: bool = True) -> str:
    global LLM_CALL_COUNT
    if track_call:
        LLM_CALL_COUNT += 1
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if LLM_MAX_TOKENS:
        payload["options"] = {"num_predict": LLM_MAX_TOKENS}
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
        _post_llm(model, prompt="warmup ping", track_call=False)


def read_snippet(path_candidates: Iterable[Path]) -> str:
    """Return concatenated snippets from available files or directories.

    The index directory may contain per-category folders (e.g., ``indexes/Python``)
    rather than flat files. To honor that layout we gather a few readable files
    from each candidate directory while still supporting direct file paths.
    Missing or unreadable files are skipped.
    """

    snippets: list[str] = []
    for path in path_candidates:
        if path.is_file():
            try:
                snippets.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            continue

        if path.is_dir():
            try:
                # Prefer a small, deterministic subset to avoid traversing
                # excessively large trees.
                for child in sorted(path.iterdir())[:5]:
                    if child.is_file():
                        try:
                            snippets.append(child.read_text(encoding="utf-8", errors="replace"))
                        except OSError:
                            continue
            except OSError:
                continue

    return _clean_text("\n\n".join(snippets))


def collect_section_snippet(index_dir: Path, theme: str, *, limit: int = 3) -> str:
    """Collect snippet text for a theme by scanning the index directory.

    The theme is split into lightweight tokens and used to pick a small number
    of matching files or directories under ``index_dir``. The function limits
    traversal to avoid expensive walks.
    """

    tokens = [token.lower() for token in re.split(r"[^\w]+", theme) if token]
    if not tokens or not index_dir.exists():
        return ""

    candidates: list[Path] = []
    for path in sorted(index_dir.iterdir()):
        name = (path.stem if path.is_file() else path.name).lower()
        if any(token in name for token in tokens):
            candidates.append(path)
        if len(candidates) >= limit:
            break

    return read_snippet(candidates)


def discover_category_files(index_dir: Path) -> Mapping[str, Path]:
    """Map canonical category names to existing files under the index directory."""

    mapping: dict[str, Path] = {}
    if not index_dir.exists():
        return mapping

    for path in index_dir.iterdir():
        candidate_name = path.stem if path.is_file() else path.name
        for name in SCRIPT_CATEGORY_FILES:
            normalized = name.lower().replace(" & ", " ")
            if candidate_name.lower().startswith(normalized):
                mapping[name] = path
                break
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
    llm_answer = _post_llm(CLASSIFIER_MODEL, prompt_body, system=system)
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


def _next_chart_id(kind: str) -> str:
    _next_chart_id.counter += 1
    return f"{kind}_chart_{_next_chart_id.counter}"


_next_chart_id.counter = 0


def build_bar_chart(values: list[int], labels: list[str], *, title: str, caption: str, aria_label: str) -> str:
    canvas_id = _next_chart_id("bar")
    data_json = json.dumps(values)
    labels_json = json.dumps(labels)
    script = f"""
<script>
(() => {{
  const canvas = document.getElementById('{canvas_id}');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const values = {data_json};
  const labels = {labels_json};
  const clampLabel = (text) => {{
    const normalized = String(text ?? '');
    return normalized.length > 12 ? normalized.slice(0, 12) + '…' : normalized;
  }};
  const width = canvas.width;
  const height = canvas.height;
  const padding = 40;
  const barWidth = (width - padding * 2) / Math.max(values.length, 1);
  const maxVal = Math.max(...values, 1);
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, '#7f5af0');
  gradient.addColorStop(1, '#2cb67d');

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0b1021';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#94a1b2';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding / 2);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding / 2, height - padding);
  ctx.stroke();

  values.forEach((value, idx) => {{
    const scaled = (value / maxVal) * (height - padding * 1.5);
    const x = padding + idx * barWidth + barWidth * 0.15;
    const y = height - padding - scaled;
    ctx.fillStyle = gradient;
    ctx.roundRect(x, y, barWidth * 0.7, scaled, 6);
    ctx.fill();
    ctx.fillStyle = '#e4e4ef';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(clampLabel(labels[idx]), x + barWidth * 0.35, height - padding + 16);
  }});
}})();
</script>
"""
    return (
        f"<figure aria-label='{html.escape(aria_label)}' style='text-align:center;'>"
        f"<figcaption>{html.escape(title)} — {html.escape(caption)}</figcaption>"
        f"<canvas id='{canvas_id}' width='520' height='260' role='img' aria-label='{html.escape(title)}' "
        f"style='display:block;margin:0 auto;'></canvas>"
        f"{script}</figure>"
    )


def build_line_chart(values: list[int], labels: list[str], *, title: str, caption: str, aria_label: str) -> str:
    canvas_id = _next_chart_id("line")
    data_json = json.dumps(values or [1, 1, 1])
    labels_json = json.dumps(labels)
    script = f"""
<script>
(() => {{
  const canvas = document.getElementById('{canvas_id}');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const values = {data_json};
  const labels = {labels_json};
  const clampLabel = (text) => {{
    const normalized = String(text ?? '');
    return normalized.length > 12 ? normalized.slice(0, 12) + '…' : normalized;
  }};
  const width = canvas.width;
  const height = canvas.height;
  const padding = 40;
  const stepX = (width - padding * 2) / Math.max(values.length - 1, 1);
  const maxVal = Math.max(...values, 1);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0b1021';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#94a1b2';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding / 2);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding / 2, height - padding);
  ctx.stroke();

  ctx.strokeStyle = '#2cb67d';
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((value, idx) => {{
    const x = padding + idx * stepX;
    const y = height - padding - (value / maxVal) * (height - padding * 1.5);
    if (idx === 0) {{
      ctx.moveTo(x, y);
    }} else {{
      ctx.lineTo(x, y);
    }}
  }});
  ctx.stroke();

  ctx.fillStyle = '#ff8906';
  values.forEach((value, idx) => {{
    const x = padding + idx * stepX;
    const y = height - padding - (value / maxVal) * (height - padding * 1.5);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e4e4ef';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(clampLabel(labels[idx]), x, height - padding + 16);
    ctx.fillStyle = '#ff8906';
  }});
}})();
</script>
"""
    return (
        f"<figure aria-label='{html.escape(aria_label)}' style='text-align:center;'>"
        f"<figcaption>{html.escape(title)} — {html.escape(caption)}</figcaption>"
        f"<canvas id='{canvas_id}' width='520' height='280' role='img' aria-label='{html.escape(title)}' "
        f"style='display:block;margin:0 auto;'></canvas>"
        f"{script}</figure>"
    )


def build_pie_chart(values: list[int], labels: list[str], *, title: str, caption: str, aria_label: str) -> str:
    canvas_id = _next_chart_id("pie")
    data_json = json.dumps(values)
    labels_json = json.dumps(labels)
    script = f"""
<script>
(() => {{
  const canvas = document.getElementById('{canvas_id}');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const values = {data_json};
  const labels = {labels_json};
  const clampLabel = (text) => {{
    const normalized = String(text ?? '');
    return normalized.length > 12 ? normalized.slice(0, 12) + '…' : normalized;
  }};
  const total = values.reduce((sum, v) => sum + v, 0) || 1;
  const radius = Math.min(canvas.width, canvas.height) / 2 - 10;
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const palette = ['#7f5af0', '#2cb67d', '#ff8906', '#e53170', '#94a1b2', '#0ea5e9'];

  let start = -Math.PI / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0b1021';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  values.forEach((value, idx) => {{
    const slice = (value / total) * Math.PI * 2;
    const end = start + slice;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, start, end);
    ctx.closePath();
    ctx.fillStyle = palette[idx % palette.length];
    ctx.fill();

    const mid = start + slice / 2;
    const lx = cx + Math.cos(mid) * (radius + 18);
    const ly = cy + Math.sin(mid) * (radius + 18);
    ctx.fillStyle = '#e4e4ef';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(clampLabel(labels[idx]), lx, ly);
    start = end;
  }});
}})();
</script>
"""
    return (
        f"<figure aria-label='{html.escape(aria_label)}' style='text-align:center;'>"
        f"<figcaption>{html.escape(title)} — {html.escape(caption)}</figcaption>"
        f"<canvas id='{canvas_id}' width='320' height='240' role='img' aria-label='{html.escape(title)}' "
        f"style='display:block;margin:0 auto;'></canvas>"
        f"{script}</figure>"
    )


def _sin_deg(angle: float) -> float:
    import math

    return math.sin(math.radians(angle))


def _cos_deg(angle: float) -> float:
    import math

    return math.cos(math.radians(angle))


def _is_llm_error(text: str | None) -> bool:
    return not text or text.startswith("[LLM error")


def _clean_text(text: str) -> str:
    """Remove garbled characters and collapse duplicate lines/sentences.

    The source knowledge files occasionally include replacement characters
    (e.g., ``�``) and repeated paragraphs. Cleaning them before and after LLM
    calls reduces visible artifacts in the rendered sections.
    """

    if not text:
        return ""

    # Strip common replacement characters or byte-order marks.
    cleaned = text.replace("\ufffd", "").replace("\ufeff", "")
    cleaned = re.sub(r"�+", "", cleaned)

    # Deduplicate consecutive sentences within a paragraph.
    paragraphs = [para.strip() for para in cleaned.split("\n")]
    normalized_paragraphs: list[str] = []
    for para in paragraphs:
        if not para:
            continue
        sentences = re.split(r"(?<=[。.!?])\s+", para)
        deduped: list[str] = []
        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue
            if deduped and stripped == deduped[-1]:
                continue
            deduped.append(stripped)
        normalized = " ".join(deduped)
        if normalized_paragraphs and normalized == normalized_paragraphs[-1]:
            continue
        normalized_paragraphs.append(normalized)

    collapsed = "\n".join(normalized_paragraphs)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _format_text_block(text: str) -> str:
    escaped = html.escape(text)
    return escaped.replace("\n", "<br />\n")


def _ensure_minimum_length(
    text: str, *, min_chars: int, padding_phrase: str, theme: str, category: str
) -> str:
    if len(text) >= min_chars:
        return text

    padding_lines = [
        f"{padding_phrase} 見出し『{theme}』を起点に、{category}の現場で揺れる感情や数字を掘り下げる。",
        f"ヨハネが夜のカフェでノートに書き殴る独白として、{theme}の違和感と実装プランを往復させる。",
        f"パンクな疑いと静かな祈りを両立させながら、{category}をめぐる地政学・技術・倫理を丁寧に編み直す。",
        f"{theme}という言葉が刺さる理由を、友人に語りかけるようなリズムで具体化し、読者の体験に重ねる。",
        f"数字の裏側にある失敗談や小さな成功例を織り込み、{category}の未来を自分ごととして描き直す。",
    ]

    idx = 0
    while len(text) < min_chars:
        padding = padding_lines[idx % len(padding_lines)]
        text = f"{text}\n{padding}"
        idx += 1
    return text


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
    snippet_hint = _clean_text(category_snippet[:200])
    formatted_values = ", ".join(f"{label}:{value}" for label, value in zip(labels, values))
    if pack.code == "ja":
        return textwrap.dedent(
            f"""
            テーマ「{theme}」に沿って、{category}領域の論点を再構成する。
            図「{chart_title}」のデータ（{formatted_values}）を手掛かりに、ヨハネは強弱の差からリスク配分を読み直す。
            {pack.context_line}
            パンクな疑いと静かな洞察を両立させ、見出し通りの論点に引き戻す。
            """
        ).strip()

    if pack.code == "it":
        return textwrap.dedent(
            f"""
            Con il tema "{theme}" rilegge il quadro nel contesto {category}.
            I dati del grafico "{chart_title}" ({formatted_values}) mostrano dove l'attenzione e il rischio cambiano intensità.
            {pack.context_line}
            Il tono resta sobrio e critico, così da rispettare titolo e diagramma。
            """
        ).strip()

    return textwrap.dedent(
        f"""
        Using the theme "{theme}", Yohane reframes the core questions inside the {category} lens.
        The chart "{chart_title}" with values {formatted_values} anchors the section to the heading instead of repeating boilerplate.
        Reference snippet: {snippet_hint or 'N/A'}.
        {pack.context_line}
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
            見出し「{theme}」に対して、約750文字で、かつ{MIN_SECTION_CHARS}文字以上の長文を作成してください。
            {category}の現場感とヨハネの個人的な体験をつなぎ、パラグラフごとに角度を変えて深掘りします。同じ主張や表現を繰り返さず、段落ごとに新しい視点や具体例を加えること。
            期待する構造:
            - 冒頭: 見出しで提示した違和感や問いを、ヨハネの視点で鮮やかに描写する。
            - 中盤: {pack.context_line} を踏まえ、事例・数字・倫理的なひっかかりを具体的に展開する。
            - 後半: 図「{chart_title}」のラベル{labels}と値{values}を手がかりに、リスク設計や意思決定のニュアンスを掘る。
            - 締め: 読者に向けて静かな伴走を示す。
            参考スニペット（引用や宣伝は禁止。本文には書かない）:
            {category_snippet[:1200] or 'N/A'}
            語調: 冷静で批判的、かすかなパンクさを含める。英訳・翻訳注記は禁止。インデックスの内容を抜粋したと明示する記述は避ける。
            """
        )
    else:
        system = "You are a structured blog assistant. Keep the language aligned to the user locale."
        body = textwrap.dedent(
            f"""
            Write a long-form section in {pack.code} with roughly 750 characters (at least {MIN_SECTION_CHARS}) for persona Yohane.
            Theme keyword: {theme}
            Chosen category: {category}
            Desired structure:
            - Opening: a vivid hook that ties Yohane's personal tension to the heading.
            - Middle: weave {pack.context_line} with concrete stories, data points, and ethical friction.
            - Later: interpret the chart '{chart_title}' using labels {labels} and values {values} to shape risk/return nuance.
            - Closing: offer a quiet companion message to readers.
            Context from the index (do NOT quote or promote it in the text; use it silently):
            {category_snippet[:1200] or 'N/A'}
            Keep it analytical, skeptical, reflective, and avoid repetitive boilerplate or citation-like phrasing.
            """
        )

    llm_text = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(llm_text):
        llm_text = _fallback_section_summary(
            prompt=prompt,
            category=category,
            theme=theme,
            pack=pack,
            values=values,
            labels=labels,
            category_snippet=category_snippet,
            chart_title=chart_title,
        )

    cleaned = _clean_text(llm_text)

    return _ensure_minimum_length(
        cleaned,
        min_chars=MIN_SECTION_CHARS,
        padding_phrase=pack.padding_phrase,
        theme=theme,
        category=category,
    )


def explain_chart_with_llm(
    values: list[int],
    labels: list[str],
    theme: str,
    pack: LanguagePack,
    *,
    chart_title: str,
) -> str:
    if not ENABLE_CHART_LLM:
        if pack.code == "ja":
            return (
                f"図「{chart_title}」は{theme}の文脈で、{', '.join(labels)}のバランスを値{values}として示す。\n"
                "極端な値の差を冷静に読み取り、見出しの問いに沿って解釈する。"
            )
        if pack.code == "it":
            return (
                f"Il grafico '{chart_title}' per '{theme}' mette a confronto {', '.join(labels)} con valori {values}.\n"
                "Le differenze evidenziano dove attenzione e rischio vanno calibrati."
            )
        return (
            f"The chart '{chart_title}' ties the theme '{theme}' to the {labels} weights {values}.\n"
            "It highlights where Yohane would lean in or step back while calibrating risk."
        )

    if pack.code == "ja":
        system = (
            "あなたは日本語でグラフを簡潔に解説するアシスタントです。出力は日本語のみで翻訳注記は禁止。"
        )
        prompt_body = textwrap.dedent(
            f"""
            以下の擬似データを持つ図を2〜3文で解説し、文と文の間に改行を入れてください。英訳・翻訳の併記は禁止です。
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
            Summarize the following synthetic chart insightfully in {pack.code}. Provide 2-3 sentences separated by line breaks.
            Theme: {theme}
            Chart title: {chart_title}
            Labels and values: {list(zip(labels, values))}
            Persona tone: calm, critical, and slightly punk.
            """
        )
    llm_text = _post_llm(CHART_MODEL, prompt_body, system=system)
    if _is_llm_error(llm_text):
        if pack.code == "ja":
            return (
                f"図「{chart_title}」は{theme}の文脈で、{', '.join(labels)}のバランスを値{values}として示す。\n"
                "極端な値の差を冷静に読み取り、見出しの問いに沿って解釈する。"
            )
        if pack.code == "it":
            return (
                f"Il grafico '{chart_title}' per '{theme}' mette a confronto {', '.join(labels)} con valori {values}.\n"
                "Le differenze evidenziano dove attenzione e rischio vanno calibrati."
            )
        return (
            f"The chart '{chart_title}' ties the theme '{theme}' to the {labels} weights {values}.\n"
            "It highlights where Yohane would lean in or step back while calibrating risk."
        )
    return _clean_text(llm_text)


def generate_sections(
    prompt: str, category: str, pack: LanguagePack, *, category_snippet: str
) -> list[Section]:
    with status_scope(f"generate_sections:{pack.code}"):
        base = _deterministic_numbers(prompt + category + pack.code, 12, 9)
        sections: list[Section] = []

        for idx, theme in enumerate(pack.themes):
            section_snippet = collect_section_snippet(INDEX_DIR, theme)
            snippet_for_llm = section_snippet or category_snippet

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
                pack.section_body_template.format(
                    category=category, theme=theme, context=pack.context_line
                )
            ).strip()

            chart_title = pack.chart_titles[
                "bar" if idx % 3 == 0 else "line" if idx % 3 == 1 else "pie"
            ]
            llm_summary = summarize_section_with_llm(
                prompt,
                category,
                theme,
                pack,
                snippet_for_llm,
                values=values,
                labels=labels,
                chart_title=chart_title,
            )
            combined_body = _clean_text(body + "\n" + llm_summary)
            combined_body = _ensure_minimum_length(
                combined_body,
                min_chars=MIN_SECTION_CHARS,
                padding_phrase=pack.padding_phrase,
                theme=theme,
                category=category,
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
                    body=combined_body,
                    diagram_html=diagram,
                    chart_note=chart_note,
                )
            )
        return sections


def summarize_index_with_llm(
    *,
    prompt: str,
    category: str,
    pack: LanguagePack,
    index_text: str,
) -> str:
    cleaned_index = _clean_text(index_text)
    if not cleaned_index:
        return ""

    snippet = cleaned_index[:1600]
    if pack.code == "ja":
        system = (
            "あなたは日本語で批判的かつ端的に要約するアシスタントです。出力は日本語のみ。"
        )
        body = textwrap.dedent(
            f"""
            以下のインデックス知識を踏まえ、テーマ「{prompt}」とカテゴリ「{category}」に沿った総論を3〜4文でまとめてください。
            同じ文章を繰り返さず、文字化けに見える記号は使わないこと。引用文や宣伝調は禁止。
            参照テキスト:
            {snippet}
            """
        )
    else:
        system = "Provide a concise, non-repetitive summary in the user's locale."
        body = textwrap.dedent(
            f"""
            Using the index notes below, craft a brief conclusion for category "{category}" and prompt "{prompt}".
            Keep it 3-4 sentences, avoid repeated lines or garbled characters, and maintain an analytical, calm tone.
            Source notes:
            {snippet}
            """
        )

    llm_text = _post_llm(SECTION_MODEL, body, system=system)
    if _is_llm_error(llm_text):
        return _clean_text(textwrap.shorten(cleaned_index, width=500, placeholder="…"))

    return _clean_text(llm_text)


def build_outro(common_text: str, pack: LanguagePack) -> str:
    outro_parts = list(pack.outro_lines)
    if common_text:
        outro_parts.append(common_text)
    return "\n".join(outro_parts)


def _build_summary_section(
    *, summary: str, outro: str, pack: LanguagePack, category: str
) -> Section:
    combined = "\n".join(part for part in (summary, outro) if part)
    combined = _clean_text(combined)
    extended_outro = _ensure_minimum_length(
        combined,
        min_chars=MIN_SECTION_CHARS,
        padding_phrase=pack.padding_phrase,
        theme=pack.summary_heading,
        category=category,
    )
    return Section(
        heading=pack.summary_heading,
        body=extended_outro,
        diagram_html="",
        chart_note="",
    )


def _shorten_phrase(text: str, *, limit: int = 28) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return squashed[:limit] + "…"


def _sanitize_tag(label: str) -> str:
    cleaned = re.sub(r"[#]+", "", label)
    cleaned = re.sub(r"\s+", "-", cleaned.strip())
    limited = cleaned[:15]
    if not limited:
        limited = "tag"
    return f"#{limited}"


def _build_persona_description(prompt: str, category: str, pack: LanguagePack) -> str:
    trimmed = _shorten_phrase(prompt, limit=36)
    if pack.code == "ja":
        return (
            f"ヨハネが{category}の歪さをかみ砕き、「{trimmed}」を夜のカフェ目線で要約する後書き。"
            "リスクと静けさのバランスを探る読者に向けた一息メモ。"
        )
    if pack.code == "it":
        return (
            f"Yohane smonta le crepe di {category} e riassume '{trimmed}' con lo sguardo notturno del caffè,"
            " per chi cerca equilibrio tra rischio e quiete."
        )
    return (
        f"Yohane distills the cracks in {category}, turning '{trimmed}' into a late-night café recap for readers"
        " hunting balance between risk and calm."
    )


def _build_persona_tags(prompt: str, category: str, pack: LanguagePack) -> str:
    focus = _shorten_phrase(prompt, limit=18) or category
    if pack.code == "ja":
        big = [f"{category}解体", "ヨハネの夜視点"]
        normal = ["静かな反逆", "リスク設計メモ"]
        small = [f"{focus}断片", "データと物語"]
    elif pack.code == "it":
        big = [f"{category} senza filtri", "Lente notturna di Yohane"]
        normal = ["punk-critico", "rischio-e-quiete"]
        small = [f"nota-{_shorten_phrase(focus, limit=12)}", "dati-e-narrazione"]
    else:
        big = [f"{category} breakdown", "Yohane-night-lens"]
        normal = ["quiet-punk", "risk-vs-calm"]
        small = [f"{focus}-memo", "data-x-story"]

    tags = big + normal + small
    return " ".join(_sanitize_tag(tag) for tag in tags)


def compose_html(
    title: str,
    description: str,
    intro: str,
    sections: list[Section],
    summary_section: Section,
    *,
    padding_phrase: str,
    persona_description: str,
    persona_tags: str,
    lang_code: str,
) -> str:
    body_parts = [
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='description'>{_format_text_block(description)}</p>",
        f"<p class='intro'>{_format_text_block(intro)}</p>",
    ]
    for section in sections:
        body_parts.append(f"<h2>{html.escape(section.heading)}</h2>")
        body_parts.append(f"<p>{_format_text_block(section.body)}</p>")
        body_parts.append(section.diagram_html)
        if section.chart_note.strip():
            body_parts.append(f"<p class='chart-note'>{_format_text_block(section.chart_note)}</p>")
    body_parts.append(f"<h2>{html.escape(summary_section.heading)}</h2>")
    body_parts.append(f"<p>{_format_text_block(summary_section.body)}</p>")

    body_parts.append("<section class='persona-meta'>")
    body_parts.append("<h3>Description</h3>")
    body_parts.append(f"<p class='persona-description'>{_format_text_block(persona_description)}</p>")
    body_parts.append("<h3>Tags</h3>")
    body_parts.append(f"<p class='persona-tags'>{html.escape(persona_tags)}</p>")
    body_parts.append("</section>")

    html_doc = "\n".join(body_parts)
    return textwrap.dedent(
        f"""
        <!DOCTYPE html>
        <html lang="{html.escape(lang_code)}">
        <head>
            <meta charset="UTF-8" />
            <title>{html.escape(title)}</title>
        </head>
        <body>
        <article>
        {html_doc}
        </article>
        </body>
        </html>
        """
    ).strip()


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
        combined_index_text = "\n\n".join(
            part for part in (category_snippet, common_text) if part
        )
        summary = summarize_index_with_llm(
            prompt=prompt,
            category=category,
            pack=pack,
            index_text=combined_index_text,
        )
        summary_section = _build_summary_section(
            summary=summary, outro=outro, pack=pack, category=category
        )
        persona_description = _build_persona_description(prompt, category, pack)
        persona_tags = _build_persona_tags(prompt, category, pack)
        return compose_html(
            title,
            pack.description,
            pack.intro,
            sections,
            summary_section,
            padding_phrase=pack.padding_phrase,
            persona_description=persona_description,
            persona_tags=persona_tags,
            lang_code=pack.code,
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
        if WARM_MODELS:
            warm_model(CLASSIFIER_MODEL)
            warm_model(SECTION_MODEL)
            if ENABLE_CHART_LLM:
                warm_model(CHART_MODEL)

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
            "llm_calls": 0,
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

        results["llm_calls"] = LLM_CALL_COUNT

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
