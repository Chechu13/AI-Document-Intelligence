"""Read-only inventory of Windows-suffixed SROIE files.

This intentionally does not import or modify the production loader. It reports
filename groups, SHA-256 content identity, and JSON annotation differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DUPLICATE_SUFFIX = re.compile(r"^(?P<base>.+?)(?:\((?P<index>\d+)\))?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(path: Path) -> str:
    match = DUPLICATE_SUFFIX.fullmatch(path.stem)
    if match is None:
        return path.stem
    return match.group("base")


def group_paths(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[canonical_id(path)].append(path)
    return dict(sorted(groups.items()))


def relative_names(paths: list[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(paths)]


def content_groups(paths: list[Path], root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[sha256(path)].append(path.relative_to(root).as_posix())
    return dict(sorted(groups.items()))


def annotation_signature(path: Path) -> tuple[str, Any]:
    try:
        return ("json", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ("error", str(exc))


def print_content_summary(label: str, paths: list[Path], root: Path) -> None:
    groups = content_groups(paths, root)
    print(f"unique {label} contents: {len(groups)}")
    duplicate_hash_groups = [files for files in groups.values() if len(files) > 1]
    print(f"byte-identical {label} hash groups: {len(duplicate_hash_groups)}")
    for files in duplicate_hash_groups[:10]:
        print(f"  {label}: {files}")


def print_duplicate_groups(label: str, groups: dict[str, list[Path]], root: Path) -> None:
    duplicate_groups = {key: paths for key, paths in groups.items() if len(paths) > 1}
    duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups.values())
    print(f"duplicate {label}s: {duplicate_files}")
    print(f"duplicate-looking {label} groups: {len(duplicate_groups)}")
    for key, paths in list(duplicate_groups.items())[:10]:
        hashes = {sha256(path) for path in paths}
        print(f"  {key}: {relative_names(paths, root)}")
        print(f"    unique hashes: {len(hashes)}")


def print_annotation_comparisons(groups: dict[str, list[Path]], root: Path) -> None:
    print("annotation JSON comparisons for duplicate-looking TXT groups:")
    shown = 0
    for key, paths in groups.items():
        if len(paths) < 2:
            continue
        signatures = {repr(annotation_signature(path)) for path in paths}
        if len(signatures) > 1:
            print(f"  {key}: {len(signatures)} semantic contents")
            for path in sorted(paths):
                print(f"    {path.relative_to(root).as_posix()}: {annotation_signature(path)[1]!r}")
            shown += 1
            if shown == 10:
                break
    if shown == 0:
        print("  all duplicate-looking TXT groups have equivalent JSON contents")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {root}")

    jpg_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".jpg")
    txt_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")
    jpg_groups = group_paths(jpg_paths)
    txt_groups = group_paths(txt_paths)
    canonical_ids = set(jpg_groups) | set(txt_groups)

    print(f"root: {root}")
    print(f"JPG files: {len(jpg_paths)}")
    print(f"TXT files: {len(txt_paths)}")
    print(f"unique canonical receipt IDs: {len(canonical_ids)}")
    print_duplicate_groups("JPG", jpg_groups, root)
    print_duplicate_groups("TXT", txt_groups, root)
    print_content_summary("JPG", jpg_paths, root)
    print_content_summary("TXT", txt_paths, root)
    print_annotation_comparisons(txt_groups, root)


if __name__ == "__main__":
    main()