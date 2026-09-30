"""Rule-based extractor implementing the stable KIE interface."""

from __future__ import annotations

from src.ocr.engine import OCRResult

from .rules import (
    extract_address_candidate,
    extract_company_candidate,
    extract_date_candidate,
    extract_total_candidate,
)
from .types import ExtractionResult


class RuleBasedExtractor:
    """Deterministic, interpretable SROIE candidate extractor."""

    def extract(self, ocr_result: OCRResult) -> ExtractionResult:
        """Extract four fields without changing the supplied OCR result."""

        if not isinstance(ocr_result, OCRResult):
            return ExtractionResult()
        return ExtractionResult(
            company=extract_company_candidate(ocr_result),
            date=extract_date_candidate(ocr_result),
            address=extract_address_candidate(ocr_result),
            total=extract_total_candidate(ocr_result),
        )