"""Loading and validation utilities for the SROIE receipt dataset."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FIELDS = ("company", "date", "address", "total")
DEFAULT_DATASET_ROOT = Path("data/raw/SROIE")
DATASET_COLUMNS = (
    "image_id",
    "image_path",
    "annotation_path",
    *REQUIRED_FIELDS,
    "duplicate_image_count",
    "duplicate_annotation_count",
)
WINDOWS_DUPLICATE_SUFFIX = re.compile(r"^(?P<base>.+?)(?:\(\d+\))?$")


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


def _canonical_id(path: Path) -> str:
    match = WINDOWS_DUPLICATE_SUFFIX.fullmatch(path.stem)
    return match.group("base") if match else path.stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_by_canonical_id(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_canonical_id(path), []).append(path)
    return groups


def _select_canonical_path(paths: list[Path]) -> Path:
    return min(
        paths,
        key=lambda path: (
            path.stem != _canonical_id(path),
            path.as_posix(),
        ),
    )


def _json_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_sroie_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Load SROIE images and JSON annotations into a DataFrame.

    Images and annotations are grouped by canonical filename stem. A trailing
    Windows duplicate suffix such as ``(1)`` is removed for grouping only.
    Identical duplicate copies are represented by one canonical record, while
    conflicting copies are reported as validation issues. By default, any
    validation issue raises :class:`SROIEValidationError`. With ``strict=False``,
    invalid canonical samples are excluded after all issues have been collected.
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

    image_groups = _group_by_canonical_id(image_paths)
    annotation_groups = _group_by_canonical_id(annotation_paths)

    for image_id in sorted(set(image_groups) | set(annotation_groups)):
        candidate_images = image_groups.get(image_id, [])
        candidate_annotations = annotation_groups.get(image_id, [])
        if not candidate_images:
            issues.append(
                ValidationIssue(
                    kind="annotation without corresponding image",
                    image_id=image_id,
                    detail=", ".join(str(path) for path in candidate_annotations),
                )
            )
            continue
        if not candidate_annotations:
            issues.append(
                ValidationIssue(
                    kind="image without annotation",
                    image_id=image_id,
                    detail=", ".join(str(path) for path in candidate_images),
                )
            )
            continue

        image_hashes = {_sha256(path) for path in candidate_images}
        image_conflict = len(image_hashes) > 1
        if image_conflict:
            issues.append(
                ValidationIssue(
                    kind="conflicting duplicate image contents",
                    image_id=image_id,
                    detail=", ".join(str(path) for path in candidate_images),
                )
            )

        parsed_annotations: list[tuple[Path, Any]] = []
        invalid_annotation = False
        for annotation_path in candidate_annotations:
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                issues.append(
                    ValidationIssue(
                        kind="invalid JSON annotation",
                        path=annotation_path,
                        image_id=image_id,
                        detail=str(exc),
                    )
                )
                invalid_annotation = True
                continue
            if not isinstance(annotation, dict):
                issues.append(
                    ValidationIssue(
                        kind="invalid JSON annotation",
                        path=annotation_path,
                        image_id=image_id,
                        detail="annotation must contain a JSON object",
                    )
                )
                invalid_annotation = True
                continue
            parsed_annotations.append((annotation_path, annotation))

        if invalid_annotation or image_conflict or not parsed_annotations:
            continue

        annotation_hashes = {_sha256(path) for path in candidate_annotations}
        annotation_signatures = {_json_signature(annotation) for _, annotation in parsed_annotations}
        if len(annotation_hashes) > 1 or len(annotation_signatures) > 1:
            issues.append(
                ValidationIssue(
                    kind="conflicting duplicate annotation contents",
                    image_id=image_id,
                    detail=", ".join(str(path) for path in candidate_annotations),
                )
            )
            continue

        image_path = _select_canonical_path(candidate_images)
        annotation_path, annotation = min(
            parsed_annotations,
            key=lambda item: (
                item[0].stem != _canonical_id(item[0]),
                item[0].as_posix(),
            ),
        )

        missing_fields = [field for field in REQUIRED_FIELDS if field not in annotation]
        if missing_fields:
            issues.append(
                ValidationIssue(
                    kind="missing required fields",
                    path=annotation_path,
                    image_id=image_id,
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
                    image_id=image_id,
                    detail=", ".join(empty_fields),
                )
            )

        if missing_fields or empty_fields:
            continue

        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                **{field: str(annotation[field]) for field in REQUIRED_FIELDS},
                "duplicate_image_count": len(candidate_images),
                "duplicate_annotation_count": len(candidate_annotations),
            }
        )

    if issues and strict:
        raise SROIEValidationError(issues)

    return pd.DataFrame.from_records(records, columns=DATASET_COLUMNS)