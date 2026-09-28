import json
from pathlib import Path

import pytest

from src.data.sroie import SROIEValidationError, load_sroie_dataset


VALID_ANNOTATION = {
    "company": "Example Store",
    "date": "18-11-18",
    "address": "1 Example Street",
    "total": "30.90",
}


def write_sample(root: Path, image_id: str = "receipt-1", annotation: object = VALID_ANNOTATION) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{image_id}.jpg").write_bytes(b"synthetic image")
    (root / f"{image_id}.txt").write_text(json.dumps(annotation), encoding="utf-8")


def issue_kinds(root: Path) -> set[str]:
    with pytest.raises(SROIEValidationError) as error:
        load_sroie_dataset(root)
    return {issue.kind for issue in error.value.issues}


def test_loads_valid_samples(tmp_path: Path) -> None:
    write_sample(tmp_path)

    dataset = load_sroie_dataset(tmp_path)

    assert list(dataset.columns) == ["image_id", "image_path", "company", "date", "address", "total"]
    assert dataset.iloc[0]["image_id"] == "receipt-1"
    assert dataset.iloc[0]["company"] == "Example Store"


def test_reports_missing_annotation(tmp_path: Path) -> None:
    (tmp_path / "receipt-1.jpg").touch()

    assert "image without annotation" in issue_kinds(tmp_path)


def test_reports_annotation_without_image(tmp_path: Path) -> None:
    (tmp_path / "receipt-1.txt").write_text(json.dumps(VALID_ANNOTATION), encoding="utf-8")

    assert "annotation without corresponding image" in issue_kinds(tmp_path)


def test_reports_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "receipt-1.jpg").touch()
    (tmp_path / "receipt-1.txt").write_text("{not json", encoding="utf-8")

    assert "invalid JSON annotation" in issue_kinds(tmp_path)


def test_reports_missing_required_field(tmp_path: Path) -> None:
    annotation = {key: value for key, value in VALID_ANNOTATION.items() if key != "total"}
    write_sample(tmp_path, annotation=annotation)

    assert "missing required fields" in issue_kinds(tmp_path)


def test_reports_empty_field(tmp_path: Path) -> None:
    annotation = {**VALID_ANNOTATION, "address": "  "}
    write_sample(tmp_path, annotation=annotation)

    assert "empty annotation values" in issue_kinds(tmp_path)


def test_reports_duplicate_image_ids(tmp_path: Path) -> None:
    write_sample(tmp_path / "first", image_id="receipt-1")
    write_sample(tmp_path / "second", image_id="receipt-1")

    assert "duplicate image ID" in issue_kinds(tmp_path)


def test_non_strict_mode_returns_valid_samples(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="valid")
    (tmp_path / "invalid.jpg").touch()

    dataset = load_sroie_dataset(tmp_path, strict=False)

    assert dataset["image_id"].tolist() == ["valid"]