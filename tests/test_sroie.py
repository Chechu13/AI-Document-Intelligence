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

    assert list(dataset.columns) == [
        "image_id",
        "image_path",
        "annotation_path",
        "company",
        "date",
        "address",
        "total",
        "duplicate_image_count",
        "duplicate_annotation_count",
    ]
    assert dataset.iloc[0]["image_id"] == "receipt-1"
    assert dataset.iloc[0]["company"] == "Example Store"
    assert dataset.iloc[0]["duplicate_image_count"] == 1
    assert dataset.iloc[0]["duplicate_annotation_count"] == 1


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


def test_accepts_duplicate_image_ids_in_different_directories(tmp_path: Path) -> None:
    write_sample(tmp_path / "first", image_id="receipt-1")
    write_sample(tmp_path / "second", image_id="receipt-1")

    dataset = load_sroie_dataset(tmp_path)

    assert dataset["image_id"].tolist() == ["receipt-1"]
    assert dataset.iloc[0]["duplicate_image_count"] == 2
    assert dataset.iloc[0]["duplicate_annotation_count"] == 2


def test_accepts_identical_windows_duplicate_files(tmp_path: Path) -> None:
    for suffix in ("", "(1)", "(2)"):
        write_sample(tmp_path, image_id=f"X123{suffix}")

    dataset = load_sroie_dataset(tmp_path)

    assert dataset["image_id"].tolist() == ["X123"]
    assert dataset.iloc[0]["duplicate_image_count"] == 3
    assert dataset.iloc[0]["duplicate_annotation_count"] == 3
    assert dataset.iloc[0]["image_path"].endswith("X123.jpg")


def test_preserves_non_numeric_parentheses_in_receipt_id(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="X123(ABC)")

    dataset = load_sroie_dataset(tmp_path)

    assert dataset["image_id"].tolist() == ["X123(ABC)"]


def test_reports_conflicting_duplicate_image_contents(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="X123")
    (tmp_path / "X123(1).jpg").write_bytes(b"different image")
    (tmp_path / "X123(1).txt").write_text(json.dumps(VALID_ANNOTATION), encoding="utf-8")

    assert "conflicting duplicate image contents" in issue_kinds(tmp_path)


def test_reports_conflicting_duplicate_json_annotations(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="X123")
    (tmp_path / "X123(1).jpg").write_bytes(b"synthetic image")
    conflicting_annotation = {**VALID_ANNOTATION, "total": "31.00"}
    (tmp_path / "X123(1).txt").write_text(
        json.dumps(conflicting_annotation), encoding="utf-8"
    )

    assert "conflicting duplicate annotation contents" in issue_kinds(tmp_path)


def test_different_ids_with_identical_contents_remain_separate(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="X123")
    write_sample(tmp_path, image_id="X124")

    dataset = load_sroie_dataset(tmp_path)

    assert dataset["image_id"].tolist() == ["X123", "X124"]


def test_non_strict_mode_returns_valid_samples(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="valid")
    (tmp_path / "invalid.jpg").touch()

    dataset = load_sroie_dataset(tmp_path, strict=False)

    assert dataset["image_id"].tolist() == ["valid"]


def test_non_strict_mode_excludes_missing_and_empty_fields(tmp_path: Path) -> None:
    write_sample(tmp_path, image_id="valid")
    write_sample(
        tmp_path,
        image_id="missing-address",
        annotation={key: value for key, value in VALID_ANNOTATION.items() if key != "address"},
    )
    write_sample(
        tmp_path,
        image_id="empty-total",
        annotation={**VALID_ANNOTATION, "total": "  "},
    )

    dataset = load_sroie_dataset(tmp_path, strict=False)

    assert dataset["image_id"].tolist() == ["valid"]