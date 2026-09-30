"""Deterministic spatial grouping for OCR words."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from src.ocr.engine import OCRWord


@dataclass(frozen=True)
class OCRLine:
    """Words sharing a horizontal receipt line."""

    words: tuple[OCRWord, ...]
    text: str
    top: float
    confidence: float | None


def group_words_into_lines(
    words: Iterable[OCRWord],
    *,
    y_tolerance: int = 10,
) -> list[OCRLine]:
    """Group words by vertical center and sort each line left-to-right."""

    if y_tolerance < 0:
        raise ValueError("y_tolerance must be non-negative")
    valid_words = []
    for word in words:
        try:
            left, top, width, height = word.bbox
            if not str(word.text).strip() or width < 0 or height < 0:
                continue
            valid_words.append((word, top + height / 2, left))
        except (AttributeError, TypeError, ValueError):
            continue

    lines: list[list[tuple[OCRWord, float, int]]] = []
    for item in sorted(valid_words, key=lambda value: (value[1], value[2])):
        _, center, _ = item
        target = next((line for line in lines if abs(center - mean(entry[1] for entry in line)) <= y_tolerance), None)
        if target is None:
            lines.append([item])
        else:
            target.append(item)

    result = []
    for line in sorted(lines, key=lambda value: min(entry[1] for entry in value)):
        ordered = tuple(entry[0] for entry in sorted(line, key=lambda value: value[2]))
        confidences = [word.confidence for word in ordered if word.confidence is not None]
        result.append(
            OCRLine(
                words=ordered,
                text=" ".join(word.text.strip() for word in ordered),
                top=min(word.bbox[1] for word in ordered),
                confidence=mean(confidences) if confidences else None,
            )
        )
    return result