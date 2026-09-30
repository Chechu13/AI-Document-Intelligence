"""Transparent field-specific candidate rules for receipt OCR."""

from __future__ import annotations

import re
from statistics import mean

from src.ocr.engine import OCRResult, OCRWord

from .layout import OCRLine, group_words_into_lines
from .types import ExtractedField

DATE_PATTERN = re.compile(
    r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})(?!\d)"
)
MONEY_PATTERN = re.compile(r"(?<!\w)(?:RM\s*)?\d{1,9}(?:[.,]\d{2})(?!\d)", re.IGNORECASE)
TOTAL_LABEL_PATTERN = re.compile(
    r"\b(?:grand\s+total|total\s+amount|amount\s+due|total)\b", re.IGNORECASE
)
GENERIC_HEADER_PATTERN = re.compile(
    r"^(?:receipt|invoice|cash\s+bill|tax\s+invoice|receipt\s+no(?:\.|:)?|document\s+no)\b",
    re.IGNORECASE,
)
METADATA_PATTERN = re.compile(
    r"\b(?:date|cashier|receipt|invoice|tel|fax|member|document\s+no|operator)\b",
    re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(
    r"\b(?:no\.?|jalan|road|street|st\.?|ave(?:nue)?|taman|johor|kuala)\b|\b\d{5}\b",
    re.IGNORECASE,
)
EXCLUDED_TOTAL_PATTERN = re.compile(r"\b(?:subtotal|tax|gst|discount|cash|change)\b", re.IGNORECASE)


def _lines_for_result(ocr_result: OCRResult) -> list[OCRLine]:
    try:
        lines = group_words_into_lines(ocr_result.words)
    except (AttributeError, TypeError, ValueError):
        lines = []
    if lines:
        return lines
    text = getattr(ocr_result, "text", "") or ""
    return [OCRLine((), line.strip(), index, None) for index, line in enumerate(str(text).splitlines()) if line.strip()]


def _field(value: str, line: OCRLine, source: str | None = None) -> ExtractedField:
    return ExtractedField(value=value, confidence=line.confidence, source=source or line.text)


def extract_date_candidate(ocr_result: OCRResult) -> ExtractedField:
    """Return the first recognizable date substring."""

    for line in _lines_for_result(ocr_result):
        match = DATE_PATTERN.search(line.text)
        if match:
            return _field(match.group(0), line)
    return ExtractedField()


def extract_total_candidate(ocr_result: OCRResult) -> ExtractedField:
    """Return a money candidate tied to the strongest total label."""

    lines = _lines_for_result(ocr_result)
    ranked: list[tuple[int, int, OCRLine, re.Match[str]]] = []
    for index, line in enumerate(lines):
        if EXCLUDED_TOTAL_PATTERN.search(line.text) or not TOTAL_LABEL_PATTERN.search(line.text):
            continue
        matches = list(MONEY_PATTERN.finditer(line.text))
        if matches:
            label = TOTAL_LABEL_PATTERN.search(line.text)
            label_rank = 2 if label and "grand" in label.group(0).casefold() else 1
            ranked.append((label_rank, index, line, matches[-1]))
    if not ranked:
        return ExtractedField()
    _, _, line, match = max(ranked, key=lambda item: (item[0], -item[1]))
    return _field(match.group(0), line)


def _top_lines(lines: list[OCRLine], limit: int = 10) -> list[OCRLine]:
    return lines[:limit]


def extract_company_candidate(ocr_result: OCRResult) -> ExtractedField:
    """Choose a plausible business header line before receipt metadata."""

    candidates: list[tuple[tuple[int, int, int], OCRLine]] = []
    for index, line in enumerate(_top_lines(_lines_for_result(ocr_result))):
        text = line.text.strip()
        if not text or GENERIC_HEADER_PATTERN.match(text) or DATE_PATTERN.search(text) or METADATA_PATTERN.search(text):
            continue
        words = text.split()
        letters = sum(character.isalpha() for character in text)
        if letters < 3 or len(words) > 12:
            continue
        business = int(bool(re.search(r"\b(?:sdn|bhd|enterprise|store|mart|shop|hotel|restaurant)\b", text, re.IGNORECASE)))
        uppercase = int(sum(character.isupper() for character in text) >= max(1, letters // 2))
        candidates.append(((business, uppercase, -index), line))
    if not candidates:
        return ExtractedField()
    return _field(max(candidates, key=lambda item: item[0])[1].text, max(candidates, key=lambda item: item[0])[1])


def extract_address_candidate(ocr_result: OCRResult) -> ExtractedField:
    """Return contiguous address-like header lines before metadata."""

    lines = _top_lines(_lines_for_result(ocr_result), limit=12)
    company = extract_company_candidate(ocr_result).value
    selected: list[OCRLine] = []
    for line in lines:
        if METADATA_PATTERN.search(line.text) and selected:
            break
        if company and line.text.strip() == company.strip():
            continue
        if ADDRESS_PATTERN.search(line.text) and not GENERIC_HEADER_PATTERN.match(line.text):
            selected.append(line)
        elif selected and re.search(r"\b\d{5}\b|,", line.text):
            selected.append(line)
    if not selected:
        return ExtractedField()
    confidences = [line.confidence for line in selected if line.confidence is not None]
    confidence = mean(confidences) if confidences else None
    return ExtractedField(
        value=" ".join(line.text for line in selected),
        confidence=confidence,
        source="\n".join(line.text for line in selected),
    )