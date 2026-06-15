"""Generate the portfolio architecture diagram."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

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


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load Pillow's bundled font at an explicit size for portable rendering."""

    return ImageFont.load_default(size=size)


TITLE_FONT = load_font(42)
BAND_FONT = load_font(30)
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
) -> None:
    lines: list[str] = []
    for part in text.split("\n"):
        words = part.split()
        current_line = ""
        max_width = box[2] - box[0] - 24
        for word in words:
            candidate = f"{current_line} {word}".strip()
            if current_line and text_size(draw, candidate, font)[0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = candidate
        lines.append(current_line)

    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + (len(lines) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) / 2

    for line, line_height in zip(lines, line_heights, strict=True):
        line_width, _ = text_size(draw, line, font)
        x = box[0] + ((box[2] - box[0]) - line_width) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + 8


def draw_path_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    fill: str = ARROW,
) -> None:
    """Draw a directed line whose final segment determines the arrow direction."""

    draw.line(points, fill=fill, width=4, joint="curve")
    previous_x, previous_y = points[-2]
    x, y = points[-1]
    delta_x = x - previous_x
    delta_y = y - previous_y

    if abs(delta_x) >= abs(delta_y) and delta_x >= 0:
        arrow = [(x, y), (x - 16, y - 9), (x - 16, y + 9)]
    elif abs(delta_x) >= abs(delta_y):
        arrow = [(x, y), (x + 16, y - 9), (x + 16, y + 9)]
    elif delta_y >= 0:
        arrow = [(x, y), (x - 9, y - 16), (x + 9, y - 16)]
    else:
        arrow = [(x, y), (x - 9, y + 16), (x + 9, y + 16)]
    draw.polygon(arrow, fill=fill)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str = ARROW,
) -> None:
    draw_path_arrow(draw, [start, end], fill=fill)


def draw_branch_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    label: str,
) -> None:
    bbox = draw.textbbox(position, label, font=SMALL_FONT)
    padded = (bbox[0] - 6, bbox[1] - 3, bbox[2] + 6, bbox[3] + 3)
    draw.rectangle(padded, fill=BAND_FILL)
    draw.text(position, label, font=SMALL_FONT, fill=MUTED)


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill=BOX_FILL, outline=BOX_OUTLINE, width=3)
    draw_centered_text(draw, box, label, BODY_FONT)


def draw_band(
    draw: ImageDraw.ImageDraw,
    y: int,
    title: str,
    labels: list[str],
    band_h: int = 220,
) -> None:
    x0, x1 = 70, WIDTH - 70
    draw.rounded_rectangle((x0, y, x1, y + band_h), radius=10, fill=BAND_FILL, outline=BAND_OUTLINE, width=3)
    draw.text((x0 + 28, y + 24), title, font=BAND_FONT, fill=TEXT)

    box_top = y + 78
    box_h = 105
    gap = 22
    usable_w = x1 - x0 - 56
    box_w = int((usable_w - gap * (len(labels) - 1)) / len(labels))
    boxes: list[tuple[int, int, int, int]] = []

    for index, label in enumerate(labels):
        left = x0 + 28 + index * (box_w + gap)
        box = (left, box_top, left + box_w, box_top + box_h)
        boxes.append(box)
        draw_box(draw, box, label)

    for left_box, right_box in pairwise(boxes):
        draw_arrow(
            draw,
            (left_box[2] + 2, (left_box[1] + left_box[3]) // 2),
            (right_box[0] - 2, (right_box[1] + right_box[3]) // 2),
        )


def draw_langgraph_band(draw: ImageDraw.ImageDraw, y: int) -> None:
    """Draw LangGraph routing and the safety checks inside answer generation."""

    x0, x1 = 70, WIDTH - 70
    band_h = 300
    draw.rounded_rectangle(
        (x0, y, x1, y + band_h),
        radius=10,
        fill=BAND_FILL,
        outline=BAND_OUTLINE,
        width=3,
    )
    draw.text(
        (x0 + 28, y + 22),
        "LangGraph control + answer safety",
        font=BAND_FONT,
        fill=TEXT,
    )

    grade = (98, y + 86, 305, y + 166)
    generate = (430, y + 86, 665, y + 166)
    safety = (720, y + 62, 1370, y + 185)
    citation = (742, y + 100, 1002, y + 172)
    verification = (1075, y + 100, 1348, y + 172)
    grounded = (1480, y + 86, 1702, y + 166)
    retry = (110, y + 215, 390, y + 282)
    retry_retrieve = (475, y + 215, 755, y + 282)
    fallback = (1145, y + 215, 1450, y + 282)

    draw.rounded_rectangle(
        safety,
        radius=8,
        fill="#f7f6f2",
        outline=BAND_OUTLINE,
        width=2,
    )
    draw.text(
        (safety[0] + 18, safety[1] + 8),
        "Inside generate_answer node",
        font=SMALL_FONT,
        fill=MUTED,
    )

    for box, label in [
        (grade, "Grade Chunks"),
        (generate, "generate_answer node"),
        (citation, "Citation Marker Check"),
        (verification, "Claim Verification"),
        (grounded, "Answer / END"),
        (retry, "Retry Rewrite loop"),
        (retry_retrieve, "Retriever Tool"),
        (fallback, "Fallback / END"),
    ]:
        draw_box(draw, box, label)

    center_y = (grade[1] + grade[3]) // 2
    draw_arrow(draw, (grade[2] + 2, center_y), (generate[0] - 2, center_y))
    draw_branch_label(draw, (325, center_y - 36), "relevant")
    draw_arrow(
        draw,
        (generate[2] + 2, center_y),
        (citation[0] - 2, (citation[1] + citation[3]) // 2),
    )
    draw_arrow(
        draw,
        (citation[2] + 2, (citation[1] + citation[3]) // 2),
        (verification[0] - 2, (verification[1] + verification[3]) // 2),
    )
    draw_arrow(
        draw,
        (verification[2] + 2, (verification[1] + verification[3]) // 2),
        (grounded[0] - 2, center_y),
    )
    draw_branch_label(draw, (1360, center_y - 36), "verified")

    draw_path_arrow(
        draw,
        [
            ((grade[0] + grade[2]) // 2, grade[3] + 2),
            ((grade[0] + grade[2]) // 2, retry[1] - 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (215, y + 175), "retry available")
    draw_arrow(
        draw,
        (retry[2] + 2, (retry[1] + retry[3]) // 2),
        (retry_retrieve[0] - 2, (retry_retrieve[1] + retry_retrieve[3]) // 2),
        fill=LOOP,
    )
    draw_path_arrow(
        draw,
        [
            (retry_retrieve[0] - 2, retry_retrieve[3] + 2),
            (82, retry_retrieve[3] + 2),
            (82, (retry[1] + retry[3]) // 2),
            (82, center_y),
            (grade[0] - 2, center_y),
        ],
        fill=LOOP,
    )

    draw_path_arrow(
        draw,
        [
            (grade[2] - 10, grade[3] + 2),
            (fallback[0] - 2, (fallback[1] + fallback[3]) // 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (595, y + 175), "no retries")

    draw_path_arrow(
        draw,
        [
            ((citation[0] + citation[2]) // 2, citation[3] + 2),
            ((citation[0] + citation[2]) // 2, y + 190),
            (1195, y + 190),
            (1195, fallback[1] - 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (885, y + 187), "invalid citation")

    draw_path_arrow(
        draw,
        [
            ((verification[0] + verification[2]) // 2, verification[3] + 2),
            ((verification[0] + verification[2]) // 2, y + 200),
            (1385, y + 200),
            (1385, fallback[1] - 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (1285, y + 174), "unsupported")


def draw_provider_band(draw: ImageDraw.ImageDraw, y: int) -> None:
    """Draw alternative LLM providers and the shared evaluation interface."""

    x0, x1 = 70, WIDTH - 70
    band_h = 210
    draw.rounded_rectangle(
        (x0, y, x1, y + band_h),
        radius=10,
        fill=BAND_FILL,
        outline=BAND_OUTLINE,
        width=3,
    )
    draw.text(
        (x0 + 28, y + 24),
        "Providers and evaluation",
        font=BAND_FONT,
        fill=TEXT,
    )

    remote = (98, y + 78, 450, y + 183)
    local = (500, y + 78, 852, y + 183)
    interface = (955, y + 78, 1307, y + 183)
    evaluation = (1370, y + 78, 1702, y + 183)

    for box, label in [
        (remote, "DeepSeek / OpenAI-compatible"),
        (local, "Local Ollama"),
        (interface, "Shared Chat Model Interface"),
        (evaluation, "Naive vs Agentic vs Agentic + Reranker"),
    ]:
        draw_box(draw, box, label)

    draw_branch_label(draw, (290, y + 48), "alternative providers")
    draw_path_arrow(
        draw,
        [
            (remote[2] + 2, (remote[1] + remote[3]) // 2),
            (remote[2] + 2, y + 68),
            ((interface[0] + interface[2]) // 2, y + 68),
            ((interface[0] + interface[2]) // 2, interface[1] - 2),
        ],
    )
    draw_arrow(
        draw,
        (local[2] + 2, (local[1] + local[3]) // 2),
        (interface[0] - 2, (interface[1] + interface[3]) // 2),
    )
    draw_arrow(
        draw,
        (interface[2] + 2, (interface[1] + interface[3]) // 2),
        (evaluation[0] - 2, (evaluation[1] + evaluation[3]) // 2),
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
            135,
            "Ingestion",
            [
                "Gradio Upload",
                "PDF / Markdown / TXT Loader",
                "Recursive Chunker + Metadata",
                "Local Embeddings",
                "Deterministic Chroma Index",
            ],
        ),
        (
            375,
            "Retrieval",
            [
                "Query Rewrite",
                "Retriever Tool",
                "Vector Candidate Retrieval",
                "Optional Cross-Encoder Reranker",
            ],
        ),
    ]

    draw_band(draw, *bands[0])
    draw_band(draw, *bands[1])
    draw_langgraph_band(draw, 615)
    draw_provider_band(draw, 950)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
