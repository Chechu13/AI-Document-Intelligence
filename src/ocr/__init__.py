"""OCR backends and batch-processing utilities."""

from .batch import BatchOCRSummary, run_batch_ocr, run_sroie_ocr
from .engine import OCREngine, OCRResult, OCRWord, TesseractOCREngine

__all__ = [
	"BatchOCRSummary",
	"OCREngine",
	"OCRResult",
	"OCRWord",
	"TesseractOCREngine",
	"run_batch_ocr",
	"run_sroie_ocr",
]
