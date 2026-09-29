"""Batch OCR utilities for validated SROIE records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .engine import OCREngine, OCRResult, TesseractOCREngine

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchOCRSummary:
    """Counts and output location for one JSONL OCR batch."""

    total: int
    success: int
    empty: int
    failure: int
    output_path: Path


def run_batch_ocr(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    engine: OCREngine | None = None,
) -> BatchOCRSummary:
    """Run OCR for records containing ``image_id`` and ``image_path``.

    Every input record produces one JSONL output line, including missing-image,
    unreadable-image, Tesseract, and empty-OCR results.
    """

    ocr_engine = engine or TesseractOCREngine()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"success": 0, "empty": 0, "failure": 0}
    total = 0

    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            total += 1
            image_id = str(record["image_id"])
            image_path = str(record["image_path"])
            try:
                result = ocr_engine.recognize_path(image_path)
            except Exception as exc:
                result = OCRResult("", (), "failure", "engine_error", str(exc))
            counts[result.status] += 1
            if result.status != "success":
                LOGGER.warning("OCR %s for %s: %s", result.status, image_id, result.error_message)
            output_record = result.to_dict(image_id=image_id, image_path=image_path)
            stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    return BatchOCRSummary(
        total=total,
        success=counts["success"],
        empty=counts["empty"],
        failure=counts["failure"],
        output_path=destination,
    )


def run_sroie_ocr(
    dataset_root: str | Path,
    output_path: str | Path = "data/processed/sroie_ocr.jsonl",
    *,
    strict: bool = False,
    engine: OCREngine | None = None,
) -> BatchOCRSummary:
    """Run OCR over valid canonical SROIE records and write JSONL results."""

    from src.data.sroie import load_sroie_dataset

    dataset = load_sroie_dataset(dataset_root, strict=strict)
    return run_batch_ocr(dataset.to_dict("records"), output_path, engine=engine)
