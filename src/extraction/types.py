"""Typed immutable results shared by KIE implementations and evaluators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedField:
    """One extracted field and its provenance metadata."""

    value: str | None = None
    confidence: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    """Structured extraction output independent of the extraction strategy."""

    company: ExtractedField = ExtractedField()
    date: ExtractedField = ExtractedField()
    address: ExtractedField = ExtractedField()
    total: ExtractedField = ExtractedField()

    def fields(self) -> dict[str, ExtractedField]:
        """Return the four SROIE fields in a stable mapping."""

        return {
            "company": self.company,
            "date": self.date,
            "address": self.address,
            "total": self.total,
        }