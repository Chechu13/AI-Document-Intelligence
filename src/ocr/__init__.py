"""OCR backends and batch-processing utilities."""

from .batch import BatchOCRSummary, run_batch_ocr, run_sroie_ocr
from .engine import ImagePreprocessor, OCREngine, OCRResult, OCRWord, TesseractOCREngine
from .preprocessing import (
	enhance_contrast,
	grayscale_contrast_image,
	grayscale_image,
	resize_image,
	threshold_image,
)

__all__ = [
	"BatchOCRSummary",
	"ImagePreprocessor",
	"OCREngine",
	"OCRResult",
	"OCRWord",
	"TesseractOCREngine",
	"enhance_contrast",
	"grayscale_contrast_image",
	"grayscale_image",
	"resize_image",
	"run_batch_ocr",
	"run_sroie_ocr",
	"threshold_image",
]
