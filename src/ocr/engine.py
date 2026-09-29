"""Tesseract OCR engine and engine-neutral result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image, UnidentifiedImageError
import pytesseract

OCRStatus = Literal["success", "empty", "failure"]


@dataclass(frozen=True)
class OCRWord:
    """One recognized word and its location in the source image."""

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR output, including observable empty and failure states."""

    text: str
    words: tuple[OCRWord, ...]
    status: OCRStatus
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self, *, image_id: str | None = None, image_path: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "image_id": image_id,
            "image_path": image_path,
            "status": self.status,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        return result


class OCREngine(Protocol):
    """Interface implemented by OCR backends used by batch processing."""

    def recognize_path(self, image_path: str | Path) -> OCRResult:
        """Recognize text from an image path without raising routine OCR errors."""


def _confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return confidence if confidence >= 0 else None


def _line_text(data: dict[str, list[Any]], word_count: int) -> str:
    lines: dict[tuple[int, int, int], list[str]] = {}
    for index in range(word_count):
        text = str(data.get("text", [""] * word_count)[index]).strip()
        if not text:
            continue
        key = (
            int(data.get("block_num", [0] * word_count)[index]),
            int(data.get("par_num", [0] * word_count)[index]),
            int(data.get("line_num", [0] * word_count)[index]),
        )
        lines.setdefault(key, []).append(text)
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()))


class TesseractOCREngine:
    """OCR backend using the locally installed Tesseract executable."""

    def __init__(self, *, language: str = "eng", config: str = "") -> None:
        self.language = language
        self.config = config

    def recognize_path(self, image_path: str | Path) -> OCRResult:
        path = Path(image_path)
        try:
            with Image.open(path) as image:
                return self.recognize_image(image)
        except FileNotFoundError as exc:
            return OCRResult("", (), "failure", "missing_image", str(exc))
        except (OSError, UnidentifiedImageError) as exc:
            return OCRResult("", (), "failure", "unreadable_image", str(exc))
        except Exception as exc:
            return OCRResult("", (), "failure", "tesseract_error", str(exc))

    def recognize_image(self, image: Image.Image) -> OCRResult:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
            word_count = len(data.get("text", []))
            words: list[OCRWord] = []
            for index in range(word_count):
                text = str(data["text"][index]).strip()
                if not text:
                    continue
                words.append(
                    OCRWord(
                        text=text,
                        bbox=(
                            int(data["left"][index]),
                            int(data["top"][index]),
                            int(data["width"][index]),
                            int(data["height"][index]),
                        ),
                        confidence=_confidence(data.get("conf", [None] * word_count)[index]),
                    )
                )
            text = _line_text(data, word_count)
            if not text:
                return OCRResult("", tuple(words), "empty", "empty_ocr", "Tesseract returned no text")
            return OCRResult(text, tuple(words), "success")
        except Exception as exc:
            return OCRResult("", (), "failure", "tesseract_error", str(exc))
