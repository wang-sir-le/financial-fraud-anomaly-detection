"""Verify the current public inventory without regenerating scientific results.

Usage: python scripts/verify_publication.py --output /tmp/publication-check.json

The release CSV excludes itself and checksums.sha256. The checksum file includes
the release CSV and excludes itself. The figure manifest excludes itself.
Git tracked files plus nonignored untracked files define the current inventory.
Historical source hashes are provenance references, never current-file hashes.
This checks recorded bytes and publication paths, not scientific reproduction or
the contents of every compact table. Reports must be outside the public inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
RELEASE = "PUBLIC_RELEASE_MANIFEST.csv"
CHECKSUMS = "checksums.sha256"
FIGURES = "figures/MANIFEST.json"
PROVENANCE = "docs/SOURCE_PROVENANCE.csv"
RELEASE_FIELDS = ("path", "category", "size_bytes", "sha256", "upload_status")
PROVENANCE_FIELDS = (
    "public_path", "source_path", "source_sha256", "public_sha256", "transformation"
)


def failure(check: dict[str, Any], code: str, **details: Any) -> None:
    check["errors"].append({"code": code, **details})


def safe_path(root: Path, value: Any, check: dict[str, Any]) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value:
        failure(check, "invalid_relative_path", path=str(value))
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in value or path.as_posix() != value or ".." in path.parts:
        failure(check, "invalid_relative_path", path=value)
        return None
    target = root / value
    if target.is_symlink() or any(
        parent.is_symlink() for parent in target.parents if parent != root
    ):
        failure(check, "symlink_not_allowed", path=value)
        return None
    if not target.resolve().is_relative_to(root):
        failure(check, "path_outside_root", path=value)
        return None
    return target


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_file(
    root: Path, name: Any, expected_hash: Any, check: dict[str, Any], size: Any = None
) -> None:
    path = safe_path(root, name, check)
    if not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash):
        failure(check, "invalid_sha256", path=str(name))
        return
    if path is None:
        return
    if not path.is_file():
        failure(check, "missing_file", path=str(name))
        return
    try:
        actual_hash = digest(path)
    except OSError as exc:
        failure(check, "file_unreadable", path=name, reason=str(exc))
        return
    if actual_hash != expected_hash.lower():
        failure(check, "sha256_mismatch", path=name, expected=expected_hash, actual=actual_hash)
    if size is not None:
        try:
            expected_size = int(size)
            if expected_size < 0 or isinstance(size, bool) or str(expected_size) != str(size):
                raise ValueError("invalid nonnegative integer")
        except (ValueError, TypeError):
            failure(check, "invalid_byte_count", path=name, value=size)
            return
        if path.stat().st_size != expected_size:
            failure(check, "byte_count_mismatch", path=name,
                    expected=expected_size, actual=path.stat().st_size)


def read_csv(
    root: Path, name: str, fields: tuple[str, ...], check: dict[str, Any]
) -> list[dict[str, str]]:
    try:
        with (root / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(fields):
                failure(check, "invalid_csv_header", path=name,
                        expected=list(fields), actual=reader.fieldnames)
                return []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        failure(check, "csv_unreadable", path=name, reason=str(exc))
        return []
    valid = []
    for line, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            failure(check, "malformed_csv_row", path=name, line=line)
        else:
            valid.append(row)
    check["records"] = len(valid)
    return valid


def coverage(check: dict[str, Any], recorded: set[str], expected: set[str]) -> None:
    check["recorded_files"] = len(recorded)
    check["expected_files"] = len(expected)
    if expected - recorded:
        failure(check, "unmanifested_files", paths=sorted(expected - recorded))
    if recorded - expected:
        failure(check, "unexpected_manifest_entries", paths=sorted(recorded - expected))


def unique_name(check: dict[str, Any], names: set[str], name: Any) -> bool:
    if not isinstance(name, str):
        failure(check, "invalid_relative_path", path=str(name))
        return False
    if name in names:
        failure(check, "duplicate_path", path=name)
        return False
    names.add(name)
    return True


def current_inventory(root: Path, check: dict[str, Any]) -> set[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            check=True, capture_output=True,
        )
        names = set(result.stdout.decode("utf-8").rstrip("\0").split("\0")) - {""}
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        failure(check, "git_inventory_unavailable", reason=str(exc))
        return None
    for name in names:
        path = safe_path(root, name, check)
        if path is not None and not path.is_file():
            failure(check, "inventory_file_missing", path=name)
    check["files"] = len(names)
    check["basis"] = (
        "Git tracked and nonignored untracked files; ignored local files are outside scope"
    )
    return names


def verify_figures(root: Path, inventory: set[str] | None, check: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    try:
        document = json.loads((root / FIGURES).read_text(encoding="utf-8"))
        rows = document["files"]
        if not isinstance(rows, list):
            raise ValueError("files must be an array")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        failure(check, "figure_manifest_unreadable", reason=str(exc))
        return names
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
            failure(check, "invalid_figure_record", index=index)
            continue
        if unique_name(check, names, row["path"]):
            verify_file(root / "figures", row["path"], row["sha256"], check, row["bytes"])
    if inventory is not None:
        expected = {name.removeprefix("figures/") for name in inventory
                    if name.startswith("figures/") and name != FIGURES}
        coverage(check, names, expected)
    return {"figures/" + name for name in names}


def verify_release(root: Path, inventory: set[str] | None, check: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for row in read_csv(root, RELEASE, RELEASE_FIELDS, check):
        if row["upload_status"] != "include" or not row["category"]:
            failure(check, "invalid_release_metadata", path=row["path"])
        if unique_name(check, names, row["path"]):
            verify_file(root, row["path"], row["sha256"], check, row["size_bytes"])
    if inventory is not None:
        coverage(check, names, inventory - {RELEASE, CHECKSUMS})
    check["self_exclusions"] = [RELEASE, CHECKSUMS]
    return names


def verify_checksums(root: Path, inventory: set[str] | None, check: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    try:
        lines = (root / CHECKSUMS).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        failure(check, "checksums_unreadable", reason=str(exc))
        return names
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            failure(check, "malformed_checksum_line", line=line_number)
            continue
        expected_hash, name = match.groups()
        if unique_name(check, names, name):
            verify_file(root, name, expected_hash, check)
    if inventory is not None:
        coverage(check, names, inventory - {CHECKSUMS})
    check["self_exclusions"] = [CHECKSUMS]
    return names


def verify_provenance(root: Path, release: set[str], check: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for row in read_csv(root, PROVENANCE, PROVENANCE_FIELDS, check):
        name = row["public_path"]
        if (
            not row["source_path"] or not row["transformation"]
            or not SHA256.fullmatch(row["source_sha256"])
        ):
            failure(check, "invalid_provenance_metadata", path=name)
        if unique_name(check, names, name):
            verify_file(root, name, row["public_sha256"], check)
        if name not in release:
            failure(check, "provenance_target_not_in_release", path=name)
    check["scope"] = (
        "Validate listed public targets and public_sha256 only; source_path and source_sha256 "
        "describe original or historical sources and are not opened or compared with "
        "current bytes. "
        "Not every current release file requires a copied-source provenance row."
    )
    return names


def verify_publication_paths(root: Path, names: set[str], check: dict[str, Any]) -> None:
    binary_data = {".parquet", ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".model",
                   ".bin", ".db", ".sqlite", ".sqlite3", ".duckdb"}
    document_suffixes = {".doc", ".docx", ".docm", ".dot", ".dotx", ".odt"}
    media_suffixes = {
        ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".svg"
    }
    transaction_names = {"train_transaction.csv", "train_identity.csv", "test_transaction.csv",
                         "test_identity.csv", "ps_20174392719_1491204439457_log.csv",
                         "transactions.csv", "raw_transactions.csv", "labels.csv", "scores.csv",
                         "predictions.csv", "features.csv", "bootstrap_replicates.csv"}
    for name in sorted(names):
        path = safe_path(root, name, check)
        if path is None:
            continue
        lower = name.lower()
        relative = PurePosixPath(lower)
        reason = None
        if relative.suffix in document_suffixes:
            reason = "full manuscript/office documents are excluded"
        elif relative.suffix in binary_data:
            reason = "transaction caches, fitted states and binary data are excluded"
        elif relative.name in transaction_names:
            reason = "source transaction files are excluded"
        elif (
            lower.startswith(("data/raw/", "data/processed/", "outputs/"))
            and relative.name != "readme.md"
        ):
            reason = "raw/processed data and local run outputs are excluded"
        elif relative.suffix in media_suffixes and (
            not lower.startswith(("figures/main/", "figures/supplement/"))
            or not re.match(r"fig(?:s)?\d", relative.name)
            or re.search(
                r"(?:^|[/_ -])(?:pages?|screenshots?|rendered_pages|page_images)"
                r"(?:[/_ -]|\d|\.)", lower,
            )
        ):
            reason = (
                "only individual figure media are published; manuscript pages/screenshots "
                "are excluded"
            )
        elif re.search(
            r"(?:^|/)(?:row_level|transaction_level|predictions|prediction_ledgers|score_cache)"
            r"(?:/|[_.-])", lower,
        ):
            reason = "transaction-level prediction/score paths are excluded"
        if reason:
            failure(check, "prohibited_publication_path", path=name, reason=reason)
    check["files_checked"] = len(names)
    check["scope"] = (
        "Path and extension guard only; does not certify table contents or inspect manuscript text"
    )


def verify(root: Path) -> dict[str, Any]:
    checks = {name: {"errors": []} for name in
              ("inventory", "figures", "release", "checksums", "provenance", "publication_paths")}
    inventory = current_inventory(root, checks["inventory"])
    figures = verify_figures(root, inventory, checks["figures"])
    release = verify_release(root, inventory, checks["release"])
    checksums = verify_checksums(root, inventory, checks["checksums"])
    provenance = verify_provenance(root, release, checks["provenance"])
    verify_publication_paths(
        root, (inventory or set()) | figures | release | checksums | provenance,
        checks["publication_paths"],
    )
    for check in checks.values():
        check["status"] = "FAIL" if check["errors"] else "PASS"
    return {
        "status": "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL",
        "scope": (
            "Current public-file integrity and publication paths; no raw-to-model reproduction"
        ),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path,
        help="JSON report outside the repository or in an ignored local directory",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output and args.output.resolve().is_relative_to(root):
        relative = args.output.resolve().relative_to(root).as_posix()
        ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", relative],
                                 check=False, capture_output=True)
        if ignored.returncode != 0 or relative.startswith(".git/"):
            parser.error(
                "--output must not change the public inventory; "
                "use /tmp or an ignored outputs/ path"
            )
    report = verify(root)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
