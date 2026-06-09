"""Generate the portfolio architecture diagram."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "assets" / "architecture.png"
WIDTH = 1800
HEIGHT = 1200

BACKGROUND = "#f7f6f2"
BAND_FILL = "#ece8df"
BAND_OUTLINE = "#d3cabe"
BOX_FILL = "#ffffff"
BOX_OUTLINE = "#8e8175"
TEXT = "#1f2428"
MUTED = "#4f5a60"
ARROW = "#374151"
LOOP = "#7b4f2a"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(40, bold=True)
BAND_FONT = load_font(30, bold=True)
BODY_FONT = load_font(26)
SMALL_FONT = load_font(24)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = TEXT,
    max_chars: int = 18,
) -> None:
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(wrap(part, width=max_chars) or [""])

    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + (len(lines) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) / 2

    for line, line_height in zip(lines, line_heights):
        line_width, _ = text_size(draw, line, font)
        x = box[0] + ((box[2] - box[0]) - line_width) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + 8


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = ARROW) -> None:
    draw.line([start, end], fill=fill, width=4)
    x, y = end
    if end[0] >= start[0]:
        points = [(x, y), (x - 16, y - 9), (x - 16, y + 9)]
    else:
        points = [(x, y), (x + 16, y - 9), (x + 16, y + 9)]
    draw.polygon(points, fill=fill)


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill=BOX_FILL, outline=BOX_OUTLINE, width=3)
    draw_centered_text(draw, box, label, BODY_FONT)


def draw_band(
    draw: ImageDraw.ImageDraw,
    y: int,
    title: str,
    labels: list[str],
    loop: tuple[int, int] | None = None,
) -> None:
    x0, x1 = 70, WIDTH - 70
    band_h = 245
    draw.rounded_rectangle((x0, y, x1, y + band_h), radius=10, fill=BAND_FILL, outline=BAND_OUTLINE, width=3)
    draw.text((x0 + 28, y + 24), title, font=BAND_FONT, fill=TEXT)

    box_top = y + 84
    box_h = 118
    gap = 22
    usable_w = x1 - x0 - 56
    box_w = int((usable_w - gap * (len(labels) - 1)) / len(labels))
    boxes: list[tuple[int, int, int, int]] = []

    for index, label in enumerate(labels):
        left = x0 + 28 + index * (box_w + gap)
        box = (left, box_top, left + box_w, box_top + box_h)
        boxes.append(box)
        draw_box(draw, box, label)

    for left_box, right_box in zip(boxes, boxes[1:]):
        draw_arrow(
            draw,
            (left_box[2] + 2, (left_box[1] + left_box[3]) // 2),
            (right_box[0] - 2, (right_box[1] + right_box[3]) // 2),
        )

    if loop:
        from_index, to_index = loop
        source = boxes[from_index]
        target = boxes[to_index]
        top_y = box_top - 25
        source_x = (source[0] + source[2]) // 2
        target_x = (target[0] + target[2]) // 2
        draw.line(
            [(source_x, source[1]), (source_x, top_y), (target_x, top_y), (target_x, target[1] - 2)],
            fill=LOOP,
            width=4,
        )
        draw.polygon(
            [(target_x, target[1] - 2), (target_x - 9, target[1] - 18), (target_x + 9, target[1] - 18)],
            fill=LOOP,
        )

def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((70, 40), "Agentic RAG Document QA Architecture", font=TITLE_FONT, fill=TEXT)
    draw.text(
        (70, 90),
        "Current portfolio flow: deterministic ingestion, retrieval tools, LangGraph controls, provider-agnostic evaluation.",
        font=SMALL_FONT,
        fill=MUTED,
    )

    bands = [
        (
            145,
            "Ingestion",
            [
                "Gradio Upload",
                "PDF / Markdown / TXT Loader",
                "Recursive Chunker + Metadata",
                "Local Embeddings",
                "Deterministic Chroma Index",
            ],
            None,
        ),
        (
            415,
            "Retrieval",
            [
                "Query Rewrite",
                "Vector Candidate Retrieval",
                "Optional Cross-Encoder Reranker",
                "Retriever Tool",
            ],
            None,
        ),
        (
            685,
            "LangGraph",
            [
                "Grade Chunks",
                "Retry Rewrite loop",
                "Generate Answer",
                "Citation Marker Check",
                "Claim Verification",
                "Fallback",
            ],
            (1, 0),
        ),
        (
            955,
            "Providers and evaluation",
            [
                "DeepSeek / OpenAI-compatible",
                "Local Ollama",
                "Naive vs Agentic vs Agentic + Reranker",
            ],
            None,
        ),
    ]

    for band in bands:
        draw_band(draw, *band)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
