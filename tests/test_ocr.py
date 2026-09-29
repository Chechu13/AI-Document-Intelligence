import json
from pathlib import Path

from PIL import Image

from src.ocr import OCRResult, OCRWord, TesseractOCREngine, run_batch_ocr


def test_tesseract_result_contains_text_words_boxes_and_confidence(monkeypatch) -> None:
    data = {
        "text": ["", "Total", "30.90"],
        "left": [0, 10, 80],
        "top": [0, 20, 20],
        "width": [0, 50, 40],
        "height": [0, 15, 15],
        "conf": ["-1", "95.5", "-1"],
        "block_num": [0, 1, 1],
        "par_num": [0, 1, 1],
        "line_num": [0, 1, 1],
    }
    monkeypatch.setattr("src.ocr.engine.pytesseract.image_to_data", lambda *args, **kwargs: data)

    result = TesseractOCREngine().recognize_image(Image.new("RGB", (120, 80)))

    assert result.status == "success"
    assert result.text == "Total 30.90"
    assert result.words == (
        OCRWord("Total", (10, 20, 50, 15), 95.5),
        OCRWord("30.90", (80, 20, 40, 15), None),
    )
    assert result.to_dict()["words"][0]["bbox"] == [10, 20, 50, 15]


def test_empty_ocr_is_observable(monkeypatch) -> None:
    data = {
        "text": ["", " "],
        "left": [0, 0],
        "top": [0, 0],
        "width": [0, 0],
        "height": [0, 0],
        "conf": ["-1", "-1"],
    }
    monkeypatch.setattr("src.ocr.engine.pytesseract.image_to_data", lambda *args, **kwargs: data)

    result = TesseractOCREngine().recognize_image(Image.new("RGB", (20, 20)))

    assert result.status == "empty"
    assert result.error_type == "empty_ocr"
    assert result.words == ()


def test_missing_image_is_observable_without_tesseract(tmp_path: Path) -> None:
    result = TesseractOCREngine().recognize_path(tmp_path / "missing.jpg")

    assert result.status == "failure"
    assert result.error_type == "missing_image"


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def recognize_path(self, image_path: str | Path) -> OCRResult:
        path = str(image_path)
        self.calls.append(path)
        if path.endswith("broken.jpg"):
            raise RuntimeError("synthetic engine failure")
        if path.endswith("empty.jpg"):
            return OCRResult("", (), "empty", "empty_ocr", "no text")
        return OCRResult("Receipt", (OCRWord("Receipt", (1, 2, 3, 4), 88.0),), "success")


def test_batch_keeps_failures_and_associates_image_ids(tmp_path: Path) -> None:
    output_path = tmp_path / "ocr.jsonl"
    records = [
        {"image_id": "receipt-1", "image_path": "receipt.jpg"},
        {"image_id": "receipt-empty", "image_path": "empty.jpg"},
        {"image_id": "receipt-broken", "image_path": "broken.jpg"},
    ]
    engine = FakeEngine()

    summary = run_batch_ocr(records, output_path, engine=engine)

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.total == 3
    assert summary.success == 1
    assert summary.empty == 1
    assert summary.failure == 1
    assert [line["image_id"] for line in lines] == [
        "receipt-1",
        "receipt-empty",
        "receipt-broken",
    ]
    assert lines[1]["status"] == "empty"
    assert lines[2]["status"] == "failure"
    assert lines[2]["error_type"] == "engine_error"
