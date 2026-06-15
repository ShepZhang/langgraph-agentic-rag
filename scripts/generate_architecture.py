"""Generate the portfolio architecture diagram."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "assets" / "architecture.png"
WIDTH = 1800
HEIGHT = 1280

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
    """Draw the current LangGraph nodes and their conditional routes."""

    x0, x1 = 70, WIDTH - 70
    band_h = 380
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

    grade = (92, y + 86, 302, y + 166)
    generate = (350, y + 86, 570, y + 166)
    extract = (618, y + 86, 838, y + 166)
    verification = (886, y + 86, 1116, y + 166)
    finalize = (1164, y + 86, 1384, y + 166)
    grounded = (1482, y + 86, 1708, y + 166)
    retry = (110, y + 270, 350, y + 335)
    retry_retrieve = (430, y + 270, 670, y + 335)
    revision = (790, y + 270, 1050, y + 335)
    fallback = (1260, y + 270, 1510, y + 335)

    for box, label in [
        (grade, "Grade Chunks"),
        (generate, "Generate Draft + Citation Check"),
        (extract, "Extract Claims"),
        (verification, "Verify Citations"),
        (finalize, "Finalize Answer"),
        (grounded, "Answer / END"),
        (retry, "Retry Rewrite"),
        (retry_retrieve, "Retriever Tool"),
        (revision, "Revise Answer"),
        (fallback, "Fallback / END"),
    ]:
        draw_box(draw, box, label)

    center_y = (grade[1] + grade[3]) // 2
    draw_arrow(draw, (grade[2] + 2, center_y), (generate[0] - 2, center_y))
    draw_arrow(draw, (generate[2] + 2, center_y), (extract[0] - 2, center_y))
    draw_arrow(draw, (extract[2] + 2, center_y), (verification[0] - 2, center_y))
    draw_arrow(draw, (verification[2] + 2, center_y), (finalize[0] - 2, center_y))
    draw_arrow(draw, (finalize[2] + 2, center_y), (grounded[0] - 2, center_y))

    draw_path_arrow(
        draw,
        [
            ((grade[0] + grade[2]) // 2, grade[3] + 2),
            ((grade[0] + grade[2]) // 2, retry[1] - 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (95, y + 192), "insufficient evidence")
    draw_arrow(
        draw,
        (retry[2] + 2, (retry[1] + retry[3]) // 2),
        (retry_retrieve[0] - 2, (retry_retrieve[1] + retry_retrieve[3]) // 2),
        fill=LOOP,
    )
    draw_path_arrow(
        draw,
        [
            ((retry_retrieve[0] + retry_retrieve[2]) // 2, retry_retrieve[3] + 2),
            ((retry_retrieve[0] + retry_retrieve[2]) // 2, y + 355),
            (72, y + 355),
            (72, center_y),
            (grade[0] - 2, center_y),
        ],
        fill=LOOP,
    )

    draw_path_arrow(
        draw,
        [
            ((generate[0] + generate[2]) // 2, generate[3] + 2),
            ((generate[0] + generate[2]) // 2, y + 225),
            (1385, y + 225),
            (1385, fallback[1] - 2),
        ],
        fill=LOOP,
    )
    draw_path_arrow(
        draw,
        [
            (grade[2] - 10, grade[3] + 2),
            (grade[2] - 10, y + 225),
            (fallback[0] - 2, y + 225),
            (fallback[0] - 2, fallback[1] - 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (510, y + 195), "invalid draft or retry exhausted")

    draw_path_arrow(
        draw,
        [
            ((verification[0] + verification[2]) // 2, verification[3] + 2),
            ((verification[0] + verification[2]) // 2, revision[1] - 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (880, y + 192), "unsupported claims")
    draw_path_arrow(
        draw,
        [
            (revision[0] - 2, (revision[1] + revision[3]) // 2),
            (748, (revision[1] + revision[3]) // 2),
            (748, extract[3] + 2),
        ],
        fill=LOOP,
    )
    draw_path_arrow(
        draw,
        [
            (revision[2] + 2, (revision[1] + revision[3]) // 2),
            (fallback[0] - 2, (fallback[1] + fallback[3]) // 2),
        ],
        fill=LOOP,
    )
    draw_branch_label(draw, (1065, y + 238), "unsupported after revision")


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
        (evaluation, "V0-V6 + Historical Evaluation Matrix"),
    ]:
        draw_box(draw, box, label)

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
        "Current flow: deterministic ingestion, hybrid retrieval, explicit LangGraph safety nodes, and reproducible evaluation.",
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
                "Structured Query Transformation",
                "Retriever Tool",
                "Dense + BM25 + RRF",
                "Optional Cross-Encoder Reranker",
            ],
        ),
    ]

    draw_band(draw, *bands[0])
    draw_band(draw, *bands[1])
    draw_langgraph_band(draw, 615)
    draw_provider_band(draw, 1020)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
