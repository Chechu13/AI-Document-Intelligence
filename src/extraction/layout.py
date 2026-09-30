"""Deterministic spatial grouping for OCR words."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable

from src.ocr.engine import OCRWord


@dataclass(frozen=True)
class OCRLine:
    """Words belonging to the same horizontal receipt line."""

    words: tuple[OCRWord, ...]
    text: str
    top: float
    confidence: float | None


def _vertical_overlap_ratio(
    first_top: float,
    first_height: float,
    second_top: float,
    second_height: float,
) -> float:
    """Return the vertical overlap ratio between two word boxes."""

    first_bottom = first_top + first_height
    second_bottom = second_top + second_height

    overlap = max(
        0.0,
        min(first_bottom, second_bottom) - max(first_top, second_top),
    )

    if overlap <= 0:
        return 0.0

    shortest_height = min(first_height, second_height)

    if shortest_height <= 0:
        return 0.0

    return overlap / shortest_height


def group_words_into_lines(
    words: Iterable[OCRWord],
    *,
    y_tolerance: int | None = None,
) -> list[OCRLine]:
    """
    Group OCR words into horizontal lines.

    Words are first ordered by vertical position and then assigned to the
    closest compatible line. Compatibility is determined using both vertical
    center distance and bounding-box overlap.

    When y_tolerance is None, an adaptive tolerance based on the median OCR
    word height is used.
    """

    valid_words: list[tuple[OCRWord, float, float, float, float]] = []

    for word in words:
        try:
            left, top, width, height = word.bbox

            if not str(word.text).strip():
                continue

            if width < 0 or height <= 0:
                continue

            center_y = top + height / 2

            valid_words.append(
                (
                    word,
                    float(left),
                    float(top),
                    float(height),
                    float(center_y),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue

    if not valid_words:
        return []

    if y_tolerance is not None and y_tolerance < 0:
        raise ValueError("y_tolerance must be non-negative")

    median_height = median(
        item[3]
        for item in valid_words
        if item[3] > 0
    )

    tolerance = (
        float(y_tolerance)
        if y_tolerance is not None
        else max(3.0, median_height * 0.5)
    )

    # Each entry is:
    # [words, centers, left_positions]
    lines: list[
        tuple[
            list[tuple[OCRWord, float, float, float, float]],
            list[float],
        ]
    ] = []

    sorted_words = sorted(
        valid_words,
        key=lambda item: (item[4], item[1]),
    )

    for item in sorted_words:
        word, left, top, height, center_y = item

        best_line = None
        best_distance = float("inf")

        for line in lines:
            line_items, centers = line

            line_center = mean(centers)
            center_distance = abs(center_y - line_center)

            if center_distance > tolerance:
                continue

            compatible = False

            for existing in line_items:
                _, _, existing_top, existing_height, _ = existing

                overlap = _vertical_overlap_ratio(
                    top,
                    height,
                    existing_top,
                    existing_height,
                )

                if overlap >= 0.25:
                    compatible = True
                    break

            if not compatible:
                continue

            if center_distance < best_distance:
                best_distance = center_distance
                best_line = line

        if best_line is None:
            lines.append(([item], [center_y]))
        else:
            best_line[0].append(item)
            best_line[1].append(center_y)

    result: list[OCRLine] = []

    for line_items, _ in sorted(
        lines,
        key=lambda line: min(item[2] for item in line[0]),
    ):
        ordered = tuple(
            item[0]
            for item in sorted(
                line_items,
                key=lambda item: item[1],
            )
        )

        confidences = [
            word.confidence
            for word in ordered
            if word.confidence is not None
        ]

        result.append(
            OCRLine(
                words=ordered,
                text=" ".join(
                    word.text.strip()
                    for word in ordered
                    if word.text.strip()
                ),
                top=min(
                    word.bbox[1]
                    for word in ordered
                ),
                confidence=(
                    mean(confidences)
                    if confidences
                    else None
                ),
            )
        )

    return result