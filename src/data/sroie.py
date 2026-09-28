"""Loading and validation utilities for the SROIE receipt dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FIELDS = ("company", "date", "address", "total")
DEFAULT_DATASET_ROOT = Path("data/raw/SROIE")
DATASET_COLUMNS = ("image_id", "image_path", *REQUIRED_FIELDS)


@dataclass(frozen=True)
class ValidationIssue:
    """A validation problem associated with one dataset path or image ID."""

    kind: str
    path: Path | None = None
    image_id: str | None = None
    detail: str = ""

    def __str__(self) -> str:
        location = str(self.path or self.image_id or "dataset")
        suffix = f": {self.detail}" if self.detail else ""
        return f"{self.kind} ({location}){suffix}"


class SROIEValidationError(ValueError):
    """Raised when one or more SROIE samples fail validation."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        message = "SROIE dataset validation failed:\n" + "\n".join(
            f"- {issue}" for issue in self.issues
        )
        super().__init__(message)


def load_sroie_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Load SROIE images and JSON annotations into a DataFrame.

    Images and annotations are matched by filename stem in the same directory.
    By default, any validation issue raises :class:`SROIEValidationError`.
    With ``strict=False``, invalid samples are excluded after all issues have
    been collected, which is useful for exploratory analysis.
    """

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"SROIE dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"SROIE dataset root is not a directory: {root}")

    image_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".jpg"
    )
    annotation_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"
    )
    issues: list[ValidationIssue] = []
    records: list[dict[str, Any]] = []

    image_ids: dict[str, list[Path]] = {}
    for image_path in image_paths:
        image_ids.setdefault(image_path.stem, []).append(image_path)
    for image_id, paths in image_ids.items():
        if len(paths) > 1:
            issues.append(
                ValidationIssue(
                    kind="duplicate image ID",
                    image_id=image_id,
                    detail=", ".join(str(path) for path in paths),
                )
            )

    image_path_set = {path for path in image_paths}
    for annotation_path in annotation_paths:
        matching_image = annotation_path.with_suffix(".jpg")
        matching_image_upper = annotation_path.with_suffix(".JPG")
        if matching_image not in image_path_set and matching_image_upper not in image_path_set:
            issues.append(
                ValidationIssue(
                    kind="annotation without corresponding image", path=annotation_path
                )
            )

    for image_path in image_paths:
        matching_annotations = [
            path
            for path in (image_path.with_suffix(".txt"), image_path.with_suffix(".TXT"))
            if path.exists()
        ]
        if not matching_annotations:
            issues.append(
                ValidationIssue(
                    kind="image without annotation", path=image_path, image_id=image_path.stem
                )
            )
            continue

        annotation_path = matching_annotations[0]
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(
                ValidationIssue(
                    kind="invalid JSON annotation",
                    path=annotation_path,
                    image_id=image_path.stem,
                    detail=str(exc),
                )
            )
            continue

        if not isinstance(annotation, dict):
            issues.append(
                ValidationIssue(
                    kind="invalid JSON annotation",
                    path=annotation_path,
                    image_id=image_path.stem,
                    detail="annotation must contain a JSON object",
                )
            )
            continue

        missing_fields = [field for field in REQUIRED_FIELDS if field not in annotation]
        if missing_fields:
            issues.append(
                ValidationIssue(
                    kind="missing required fields",
                    path=annotation_path,
                    image_id=image_path.stem,
                    detail=", ".join(missing_fields),
                )
            )

        empty_fields = [
            field
            for field in REQUIRED_FIELDS
            if field in annotation
            and (annotation[field] is None or not str(annotation[field]).strip())
        ]
        if empty_fields:
            issues.append(
                ValidationIssue(
                    kind="empty annotation values",
                    path=annotation_path,
                    image_id=image_path.stem,
                    detail=", ".join(empty_fields),
                )
            )

        if missing_fields or empty_fields:
            continue

        records.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                **{field: str(annotation[field]) for field in REQUIRED_FIELDS},
            }
        )

    if issues and strict:
        raise SROIEValidationError(issues)

    return pd.DataFrame.from_records(records, columns=DATASET_COLUMNS)