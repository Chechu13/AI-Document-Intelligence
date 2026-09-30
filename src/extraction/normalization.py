"""Conservative normalization helpers for candidate matching and evaluation."""

from __future__ import annotations

import re


_MONEY_PATTERN = re.compile(r"(?<!\w)(?:rm\s*)?(\d{1,9}(?:[.,]\d{2}))(?!\d)", re.IGNORECASE)


def normalize_text(value: object) -> str:
    """Normalize case and whitespace without removing useful punctuation."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.casefold().strip()
    return re.sub(r"\s+", " ", text)


def normalize_money(value: object) -> str:
    """Return a canonical two-decimal monetary value or an empty string.

    The function only normalizes an already supplied monetary candidate. It
    does not search an entire receipt or decide which number is a total.
    """

    match = _MONEY_PATTERN.search(normalize_text(value))
    if match is None:
        return ""
    return f"{float(match.group(1).replace(',', '.')):.2f}"


def values_match(expected: object, predicted: object, *, monetary: bool = False) -> bool:
    """Compare two values using field-appropriate conservative normalization."""

    if monetary:
        expected_value = normalize_money(expected)
        predicted_value = normalize_money(predicted)
    else:
        expected_value = normalize_text(expected)
        predicted_value = normalize_text(predicted)
    return bool(expected_value) and expected_value == predicted_value