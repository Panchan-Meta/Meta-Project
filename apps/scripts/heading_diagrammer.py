"""Generate blog-friendly diagram HTML snippets from section headings.

This helper picks a diagram type from a constrained catalog based on the
semantics of each heading and renders minimal HTML that can be embedded in a
blog post. It also provides a utility to slice long article text into
~2,000-character sections to propose headings when none are supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

TARGET_SECTION_CHARS = 2000


@dataclass(frozen=True)
class DiagramTemplate:
    name: str
    renderer: Callable[[str], str]
    rationale: str


class HeadingDiagrammer:
    """Select diagram templates and render HTML snippets for headings."""

    def __init__(self) -> None:
        self._templates = self._build_templates()

    def choose_diagram_type(self, heading: str) -> DiagramTemplate:
        """Pick a diagram template that matches the intent of the heading."""

        normalized = heading.lower()
        keychecks = [
            ("journey" in normalized or "persona" in normalized, "Journey map"),
            ("timeline" in normalized or "history" in normalized, "Timeline diagram"),
            ("process" in normalized or "workflow" in normalized or "pipeline" in normalized, "Flowchart"),
            ("step" in normalized or "how to" in normalized or "guide" in normalized, "Step diagram"),
            ("vs" in normalized or "versus" in normalized or "comparison" in normalized, "Comparison table"),
            ("pros and cons" in normalized or "advantages" in normalized or "disadvantages" in normalized, "Pros and cons table"),
            ("framework" in normalized or "model" in normalized, "Framework diagram"),
            ("architecture" in normalized or "layer" in normalized, "Layer (hierarchical) diagram"),
            ("strategy" in normalized or "priority" in normalized, "Pyramid diagram"),
            ("roadmap" in normalized or "plan" in normalized, "Roadmap diagram"),
            ("positioning" in normalized or "quadrant" in normalized or "matrix" in normalized, "2×2 matrix diagram"),
            ("data" in normalized or "metrics" in normalized or "trend" in normalized, "Line chart"),
            ("distribution" in normalized or "share" in normalized or "mix" in normalized, "Pie chart"),
        ]

        for condition, template_name in keychecks:
            if condition:
                return self._templates[template_name]
        return self._templates["Concept map"]

    def render_heading_html(self, heading: str) -> str:
        """Return a figure HTML block for the heading and selected diagram."""

        template = self.choose_diagram_type(heading)
        body = template.renderer(heading)
        caption = html_escape(
            f"{template.name}: {template.rationale}"
        )
        fig = textwrap.dedent(
            f"""
            <figure class="diagram-block" style="border:1px solid #d8d8e0;padding:12px;border-radius:8px;margin:12px 0;">
              <div style="font-weight:bold;margin-bottom:6px;">{html_escape(heading)}</div>
              {body}
              <figcaption style="color:#555;margin-top:6px;">{caption}</figcaption>
            </figure>
            """
        ).strip()
        return fig

    def render_all(self, headings: Sequence[str]) -> List[str]:
        return [self.render_heading_html(h.strip()) for h in headings if h.strip()]

    def auto_headings_from_text(self, text: str) -> list[str]:
        """Slice text into ~2k-character sections and suggest headings.

        The heading for each slice is derived from its leading sentence.
        """

        sentences = _split_sentences(text)
        sections: list[str] = []
        buffer: list[str] = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > TARGET_SECTION_CHARS and buffer:
                sections.append(" ".join(buffer).strip())
                buffer = []
                current_len = 0
            buffer.append(sentence.strip())
            current_len += len(sentence)

        if buffer:
            sections.append(" ".join(buffer).strip())

        return [_derive_heading(section) for section in sections if section]

    def _build_templates(self) -> dict[str, DiagramTemplate]:
        return {
            "Flowchart": DiagramTemplate(
                "Flowchart",
                lambda heading: _flowchart_body(heading),
                "Step-by-step flow showing decisions and actions.",
            ),
            "Step diagram": DiagramTemplate(
                "Step diagram",
                lambda heading: _step_body(heading),
                "Sequential steps that readers can follow at a glance.",
            ),
            "Timeline diagram": DiagramTemplate(
                "Timeline diagram",
                lambda heading: _timeline_body(heading),
                "Key milestones aligned on a time axis.",
            ),
            "Journey map": DiagramTemplate(
                "Journey map",
                lambda heading: _journey_body(heading),
                "User touchpoints mapped over time and feeling.",
            ),
            "Comparison table": DiagramTemplate(
                "Comparison table",
                lambda heading: _comparison_body(heading),
                "Side-by-side criteria to weigh options.",
            ),
            "Pros and cons table": DiagramTemplate(
                "Pros and cons table",
                lambda heading: _pros_cons_body(heading),
                "Clear trade-offs between benefits and drawbacks.",
            ),
            "Layer (hierarchical) diagram": DiagramTemplate(
                "Layer (hierarchical) diagram",
                lambda heading: _layer_body(heading),
                "Stacked layers clarifying architecture depth.",
            ),
            "Pyramid diagram": DiagramTemplate(
                "Pyramid diagram",
                lambda heading: _pyramid_body(heading),
                "Priorities from foundation to peak strategy.",
            ),
            "Roadmap diagram": DiagramTemplate(
                "Roadmap diagram",
                lambda heading: _roadmap_body(heading),
                "Near- to long-term objectives on one line.",
            ),
            "2×2 matrix diagram": DiagramTemplate(
                "2×2 matrix diagram",
                lambda heading: _matrix_body(heading),
                "Quadrant positioning across two axes.",
            ),
            "Line chart": DiagramTemplate(
                "Line chart",
                lambda heading: _line_body(heading),
                "Trended values emphasizing movement over time.",
            ),
            "Pie chart": DiagramTemplate(
                "Pie chart",
                lambda heading: _pie_body(heading),
                "Share of total shown as slices of a whole.",
            ),
            "Framework diagram": DiagramTemplate(
                "Framework diagram",
                lambda heading: _framework_body(heading),
                "Business framework blocks to organize thinking.",
            ),
            "Layer diagram": DiagramTemplate(
                "Layer diagram",
                lambda heading: _layer_body(heading),
                "Layered view for stacked capabilities.",
            ),
            "Concept map": DiagramTemplate(
                "Concept map",
                lambda heading: _concept_map_body(heading),
                "Linked ideas that situate the main theme.",
            ),
        }


def _derive_heading(section: str) -> str:
    first_sentence = _split_sentences(section, limit=1)
    if not first_sentence:
        return "Section"
    words = first_sentence[0].split()
    summary = " ".join(words[:10]).strip()
    if len(words) > 10:
        summary += " …"
    return summary or "Section"


def _split_sentences(text: str, *, limit: int | None = None) -> list[str]:
    pattern = r"(?<=[。．\.\!\?！?])\s+"
    parts = re.split(pattern, text.strip()) if text.strip() else []
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences[:limit] if limit else sentences


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# =================== Diagram renderers =======================


def _flowchart_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <div style="padding:8px 12px;border:1px solid #999;border-radius:6px;">Start</div>
          <div aria-hidden="true">➜</div>
          <div style="padding:8px 12px;border:1px solid #999;border-radius:6px;">Key step</div>
          <div aria-hidden="true">➜</div>
          <div style="padding:8px 12px;border:1px solid #999;border-radius:6px;">Decision</div>
          <div aria-hidden="true">➜</div>
          <div style="padding:8px 12px;border:1px solid #999;border-radius:6px;">Outcome</div>
        </div>
        """
    ).strip()


def _step_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <ol style="display:flex;gap:12px;flex-wrap:wrap;padding-left:18px;">
          <li style="flex:1 1 140px;min-width:140px;border:1px solid #ccc;border-radius:6px;padding:8px;">Step 1<br><small>Setup</small></li>
          <li style="flex:1 1 140px;min-width:140px;border:1px solid #ccc;border-radius:6px;padding:8px;">Step 2<br><small>Execution</small></li>
          <li style="flex:1 1 140px;min-width:140px;border:1px solid #ccc;border-radius:6px;padding:8px;">Step 3<br><small>Review</small></li>
        </ol>
        """
    ).strip()


def _timeline_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;">
          <div style="text-align:center;min-width:120px;">
            <div style="font-weight:bold;">T0</div>
            <div style="width:2px;height:28px;background:#999;margin:6px auto;"></div>
            <div>Baseline</div>
          </div>
          <div style="text-align:center;min-width:120px;">
            <div style="font-weight:bold;">T1</div>
            <div style="width:2px;height:28px;background:#999;margin:6px auto;"></div>
            <div>Milestone</div>
          </div>
          <div style="text-align:center;min-width:120px;">
            <div style="font-weight:bold;">T2</div>
            <div style="width:2px;height:28px;background:#999;margin:6px auto;"></div>
            <div>Goal</div>
          </div>
        </div>
        """
    ).strip()


def _journey_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <table style="width:100%;border-collapse:collapse;">
          <tr><th style="border:1px solid #ccc;padding:6px;">Phase</th><th style="border:1px solid #ccc;padding:6px;">Actions</th><th style="border:1px solid #ccc;padding:6px;">Emotion</th></tr>
          <tr><td style="border:1px solid #ccc;padding:6px;">Discover</td><td style="border:1px solid #ccc;padding:6px;">Research &amp; trigger</td><td style="border:1px solid #ccc;padding:6px;">Curious</td></tr>
          <tr><td style="border:1px solid #ccc;padding:6px;">Evaluate</td><td style="border:1px solid #ccc;padding:6px;">Compare paths</td><td style="border:1px solid #ccc;padding:6px;">Weighing</td></tr>
          <tr><td style="border:1px solid #ccc;padding:6px;">Adopt</td><td style="border:1px solid #ccc;padding:6px;">Choose &amp; commit</td><td style="border:1px solid #ccc;padding:6px;">Confident</td></tr>
        </table>
        """
    ).strip()


def _comparison_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <table style="width:100%;border-collapse:collapse;">
          <tr><th style="border:1px solid #ccc;padding:6px;">Aspect</th><th style="border:1px solid #ccc;padding:6px;">Option A</th><th style="border:1px solid #ccc;padding:6px;">Option B</th></tr>
          <tr><td style="border:1px solid #ccc;padding:6px;">Focus</td><td style="border:1px solid #ccc;padding:6px;">Depth</td><td style="border:1px solid #ccc;padding:6px;">Speed</td></tr>
          <tr><td style="border:1px solid #ccc;padding:6px;">Cost</td><td style="border:1px solid #ccc;padding:6px;">Moderate</td><td style="border:1px solid #ccc;padding:6px;">Low</td></tr>
        </table>
        """
    ).strip()


def _pros_cons_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="flex:1 1 220px;border:1px solid #a4d4ae;background:#f2faf3;padding:8px;border-radius:6px;">
            <strong>Pros</strong>
            <ul style="margin:6px 0 0 16px;">
              <li>Clear upside</li>
              <li>Supports growth</li>
            </ul>
          </div>
          <div style="flex:1 1 220px;border:1px solid #f3b6b6;background:#fff7f7;padding:8px;border-radius:6px;">
            <strong>Cons</strong>
            <ul style="margin:6px 0 0 16px;">
              <li>Resource cost</li>
              <li>Potential risk</li>
            </ul>
          </div>
        </div>
        """
    ).strip()


def _layer_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div style="border:1px solid #ccc;padding:8px;border-radius:6px;">Experience layer</div>
          <div style="border:1px solid #ccc;padding:8px;border-radius:6px;">Service layer</div>
          <div style="border:1px solid #ccc;padding:8px;border-radius:6px;">Platform layer</div>
        </div>
        """
    ).strip()


def _pyramid_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
          <div style="border:1px solid #ccc;padding:6px 12px;border-radius:6px;width:70%;text-align:center;">Vision</div>
          <div style="border:1px solid #ccc;padding:6px 12px;border-radius:6px;width:80%;text-align:center;">Strategy</div>
          <div style="border:1px solid #ccc;padding:6px 12px;border-radius:6px;width:90%;text-align:center;">Execution</div>
        </div>
        """
    ).strip()


def _roadmap_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="flex:1 1 160px;border:1px dashed #999;padding:8px;border-radius:6px;">Now<br><small>Stabilize</small></div>
          <div style="flex:1 1 160px;border:1px dashed #999;padding:8px;border-radius:6px;">Mid-term<br><small>Scale</small></div>
          <div style="flex:1 1 160px;border:1px dashed #999;padding:8px;border-radius:6px;">Long-term<br><small>Optimize</small></div>
        </div>
        """
    ).strip()


def _matrix_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="border:1px solid #ccc;border-radius:6px;padding:8px;display:grid;grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(2,80px);gap:6px;">
          <div style="border:1px solid #ddd;padding:6px;border-radius:4px;">Quadrant 1<br><small>High / High</small></div>
          <div style="border:1px solid #ddd;padding:6px;border-radius:4px;">Quadrant 2<br><small>High / Low</small></div>
          <div style="border:1px solid #ddd;padding:6px;border-radius:4px;">Quadrant 3<br><small>Low / High</small></div>
          <div style="border:1px solid #ddd;padding:6px;border-radius:4px;">Quadrant 4<br><small>Low / Low</small></div>
        </div>
        """
    ).strip()


def _line_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <svg viewBox="0 0 220 120" role="img" aria-label="Line chart" style="width:100%;max-width:320px;">
          <polyline fill="none" stroke="#4a6cf7" stroke-width="2" points="10,100 60,70 110,60 160,40 210,30" />
          <line x1="10" y1="100" x2="210" y2="100" stroke="#ccc" stroke-width="1" />
          <line x1="10" y1="100" x2="10" y2="20" stroke="#ccc" stroke-width="1" />
        </svg>
        """
    ).strip()


def _pie_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <svg viewBox="0 0 120 120" role="img" aria-label="Pie chart" style="width:120px;height:120px;">
          <circle r="50" cx="60" cy="60" fill="#f0f0f7" />
          <path d="M60 60 L60 10 A50 50 0 0 1 110 60 Z" fill="#6c8cf7" />
          <path d="M60 60 L110 60 A50 50 0 0 1 20 80 Z" fill="#9fb7ff" />
          <path d="M60 60 L20 80 A50 50 0 0 1 60 10 Z" fill="#cfd9ff" />
        </svg>
        """
    ).strip()


def _framework_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <table style="width:100%;border-collapse:collapse;">
          <tr><th style="border:1px solid #ccc;padding:6px;">Context</th><td style="border:1px solid #ccc;padding:6px;">Problem &amp; scope</td></tr>
          <tr><th style="border:1px solid #ccc;padding:6px;">Choice</th><td style="border:1px solid #ccc;padding:6px;">Options &amp; criteria</td></tr>
          <tr><th style="border:1px solid #ccc;padding:6px;">Change</th><td style="border:1px solid #ccc;padding:6px;">Actions &amp; owners</td></tr>
        </table>
        """
    ).strip()


def _concept_map_body(heading: str) -> str:
    return textwrap.dedent(
        """
        <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
          <div style="padding:10px 12px;border:1px solid #ccc;border-radius:6px;">Main idea</div>
          <div aria-hidden="true">⟶</div>
          <div style="padding:10px 12px;border:1px solid #ccc;border-radius:6px;">Driver</div>
          <div aria-hidden="true">⟶</div>
          <div style="padding:10px 12px;border:1px solid #ccc;border-radius:6px;">Impact</div>
          <div aria-hidden="true">⟶</div>
          <div style="padding:10px 12px;border:1px solid #ccc;border-radius:6px;">Outcome</div>
        </div>
        """
    ).strip()


# =================== CLI helpers ============================


def _load_headings_from_file(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in raw.splitlines() if line.strip()]


def build_output(headings: Iterable[str]) -> str:
    hd = HeadingDiagrammer()
    blocks = hd.render_all(list(headings))
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render diagram HTML for headings.")
    parser.add_argument("--headings", nargs="*", help="Headings to render (space-delimited)")
    parser.add_argument("--headings-file", type=Path, help="Path to a file containing headings (JSON array or newline list)")
    parser.add_argument("--article-file", type=Path, help="Path to article text for auto heading generation")
    parser.add_argument("--output", type=Path, help="Optional file path to write HTML output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headings: list[str] = []

    if args.headings:
        headings.extend(args.headings)
    if args.headings_file:
        headings.extend(_load_headings_from_file(args.headings_file))
    if not headings and args.article_file:
        text = args.article_file.read_text(encoding="utf-8", errors="replace")
        headings.extend(HeadingDiagrammer().auto_headings_from_text(text))

    if not headings:
        raise SystemExit("Provide --headings, --headings-file, or --article-file with content")

    output_html = build_output(headings)
    if args.output:
        args.output.write_text(output_html, encoding="utf-8")
    else:
        print(output_html)


if __name__ == "__main__":
    main()
