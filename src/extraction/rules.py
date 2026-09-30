"""Transparent field-specific candidate rules for receipt OCR."""

from __future__ import annotations

import re
from statistics import mean

from src.ocr.engine import OCRResult

from .layout import OCRLine, group_words_into_lines
from .types import ExtractedField


DATE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r")"
    r"(?!\d)"
)

MONEY_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:RM\s*)?"
    r"\d{1,9}(?:[.,]\d{2})"
    r"(?!\d)",
    re.IGNORECASE,
)

TOTAL_LABEL_PATTERN = re.compile(
    r"\b(?:"
    r"grand\s+total"
    r"|total\s+amount"
    r"|amount\s+due"
    r"|total"
    r")\b",
    re.IGNORECASE,
)

GENERIC_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"receipt"
    r"|invoice"
    r"|cash\s+bill"
    r"|tax\s+invoice"
    r"|receipt\s+no(?:\.|:)?"
    r"|document\s+no"
    r")\b",
    re.IGNORECASE,
)

METADATA_PATTERN = re.compile(
    r"\b(?:"
    r"date"
    r"|cashier"
    r"|receipt"
    r"|invoice"
    r"|tel"
    r"|fax"
    r"|member"
    r"|document\s+no"
    r"|operator"
    r")\b",
    re.IGNORECASE,
)

ADDRESS_PATTERN = re.compile(
    r"(?:"
    r"\bno\.?\b"
    r"|\bjalan\b"
    r"|\broad\b"
    r"|\bstreet\b"
    r"|\bst\.?\b"
    r"|\bave(?:nue)?\b"
    r"|\btaman\b"
    r"|\bjohor\b"
    r"|\bkuala\b"
    r"|\b\d{5}\b"
    r")",
    re.IGNORECASE,
)

CITY_PATTERN = re.compile(
    r"\b(?:"
    r"johor"
    r"|kuala"
    r"|selangor"
    r"|penang"
    r"|melaka"
    r"|perak"
    r"|kedah"
    r"|pahang"
    r"|sabah"
    r"|sarawak"
    r"|negeri"
    r")\b",
    re.IGNORECASE,
)

EXCLUDED_TOTAL_PATTERN = re.compile(
    r"\b(?:"
    r"subtotal"
    r"|tax"
    r"|gst"
    r"|discount"
    r"|cash"
    r"|change"
    r")\b",
    re.IGNORECASE,
)


def _lines_for_result(ocr_result: OCRResult) -> list[OCRLine]:
    """Build spatial lines, falling back to OCR text when needed."""

    try:
        lines = group_words_into_lines(ocr_result.words)
    except (AttributeError, TypeError, ValueError):
        lines = []

    if lines:
        return lines

    text = getattr(ocr_result, "text", "") or ""

    return [
        OCRLine(
            (),
            line.strip(),
            float(index),
            None,
        )
        for index, line in enumerate(str(text).splitlines())
        if line.strip()
    ]


def _field(
    value: str,
    line: OCRLine,
    source: str | None = None,
) -> ExtractedField:
    """Create an extracted field with OCR provenance."""

    return ExtractedField(
        value=value,
        confidence=line.confidence,
        source=source or line.text,
    )


def extract_date_candidate(
    ocr_result: OCRResult,
) -> ExtractedField:
    """Return the first recognizable date substring."""

    for line in _lines_for_result(ocr_result):
        match = DATE_PATTERN.search(line.text)

        if match:
            return _field(
                match.group(0),
                line,
            )

    return ExtractedField()


def extract_total_candidate(
    ocr_result: OCRResult,
) -> ExtractedField:
    """
    Extract a monetary value associated with a total label.

    The value may appear on the same line as the label or on one of the
    following nearby lines. Subtotal, tax, discount, cash and change lines
    are explicitly ignored.
    """

    lines = _lines_for_result(ocr_result)

    candidates: list[
        tuple[int, int, int, OCRLine, re.Match[str]]
    ] = []

    for index, line in enumerate(lines):
        if EXCLUDED_TOTAL_PATTERN.search(line.text):
            continue

        label = TOTAL_LABEL_PATTERN.search(line.text)

        if not label:
            continue

        # Search the label line plus the next two lines.
        for distance in range(3):
            candidate_index = index + distance

            if candidate_index >= len(lines):
                break

            candidate_line = lines[candidate_index]

            # Never cross another strong excluded monetary line.
            if (
                distance > 0
                and EXCLUDED_TOTAL_PATTERN.search(candidate_line.text)
            ):
                break

            matches = list(
                MONEY_PATTERN.finditer(candidate_line.text)
            )

            for match in matches:
                label_text = label.group(0).casefold()

                if "grand" in label_text:
                    label_rank = 3
                elif "amount due" in label_text:
                    label_rank = 2
                else:
                    label_rank = 1

                same_line_rank = 2 if distance == 0 else 1

                candidates.append(
                    (
                        label_rank,
                        same_line_rank,
                        -distance,
                        candidate_line,
                        match,
                    )
                )

    if not candidates:
        return ExtractedField()

    _, _, _, line, match = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    )

    return _field(
        match.group(0),
        line,
    )


def _top_lines(
    lines: list[OCRLine],
    limit: int = 10,
) -> list[OCRLine]:
    return lines[:limit]


def extract_company_candidate(
    ocr_result: OCRResult,
) -> ExtractedField:
    """Choose a plausible business header line."""

    candidates: list[
        tuple[tuple[int, int, int], OCRLine]
    ] = []

    for index, line in enumerate(
        _top_lines(
            _lines_for_result(ocr_result)
        )
    ):
        text = line.text.strip()

        if not text:
            continue

        if GENERIC_HEADER_PATTERN.match(text):
            continue

        if DATE_PATTERN.search(text):
            continue

        if METADATA_PATTERN.search(text):
            continue

        words = text.split()

        letters = sum(
            character.isalpha()
            for character in text
        )

        if letters < 3:
            continue

        if len(words) > 12:
            continue

        business = int(
            bool(
                re.search(
                    r"\b(?:"
                    r"sdn"
                    r"|bhd"
                    r"|enterprise"
                    r"|store"
                    r"|mart"
                    r"|shop"
                    r"|hotel"
                    r"|restaurant"
                    r")\b",
                    text,
                    re.IGNORECASE,
                )
            )
        )

        uppercase = int(
            sum(
                character.isupper()
                for character in text
            )
            >= max(1, letters // 2)
        )

        candidates.append(
            (
                (
                    business,
                    uppercase,
                    -index,
                ),
                line,
            )
        )

    if not candidates:
        return ExtractedField()

    _, best_line = max(
        candidates,
        key=lambda item: item[0],
    )

    return _field(
        best_line.text.strip(),
        best_line,
    )


def _looks_like_address_continuation(
    line: OCRLine,
) -> bool:
    """Return whether a line plausibly continues an address."""

    text = line.text.strip()

    if not text:
        return False

    if DATE_PATTERN.search(text):
        return False

    if TOTAL_LABEL_PATTERN.search(text):
        return False

    if METADATA_PATTERN.search(text):
        return False

    if ADDRESS_PATTERN.search(text):
        return True

    if CITY_PATTERN.search(text):
        return True

    # A numeric address continuation such as:
    # "80000 JOHOR BAHRU"
    if re.search(r"\b\d{5}\b", text):
        return True

    return False


def extract_address_candidate(
    ocr_result: OCRResult,
) -> ExtractedField:
    """
    Extract a conservative contiguous address block.

    The extractor starts from an address-like line and accepts only nearby
    lines that still look like part of an address. It does not append
    arbitrary comma-containing lines.
    """

    lines = _top_lines(
        _lines_for_result(ocr_result),
        limit=12,
    )

    company = extract_company_candidate(
        ocr_result
    ).value

    selected: list[OCRLine] = []
    started = False

    for line in lines:
        text = line.text.strip()

        if not text:
            continue

        if company and text == company.strip():
            continue

        if DATE_PATTERN.search(text):
            if started:
                break
            continue

        if TOTAL_LABEL_PATTERN.search(text):
            if started:
                break
            continue

        if METADATA_PATTERN.search(text):
            if started:
                break
            continue

        is_address = bool(
            ADDRESS_PATTERN.search(text)
            or CITY_PATTERN.search(text)
        )

        if not started:
            if is_address:
                selected.append(line)
                started = True

            continue

        # Once an address has started, only accept lines that still
        # look like part of the address.
        if _looks_like_address_continuation(line):
            selected.append(line)
        else:
            break

        if len(selected) >= 4:
            break

    if not selected:
        return ExtractedField()

    confidences = [
        line.confidence
        for line in selected
        if line.confidence is not None
    ]

    confidence = (
        mean(confidences)
        if confidences
        else None
    )

    return ExtractedField(
        value=" ".join(
            line.text.strip()
            for line in selected
        ),
        confidence=confidence,
        source="\n".join(
            line.text.strip()
            for line in selected
        ),
    )