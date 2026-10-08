"""Hash-verified public PaySim acquisition without redistributing source records."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from fraudx.paysim_deterministic.preprocess import file_hash
from fraudx.q2_extension.common import require, write_json

URL = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1"
CSV_NAME = "PS_20174392719_1491204439457_log.csv"
CSV_SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
CSV_BYTES = 493534783


def acquire(destination: Path, archive: Path | None = None) -> dict[str, Any]:
    """Use a public download or an explicitly supplied ZIP; never overwrite a run."""
    require(not destination.exists(), "Acquisition destination must be new")
    destination.mkdir(parents=True)
    downloaded = archive is None
    if archive is None:
        archive = destination / "paysim1.zip"
        request = urllib.request.Request(URL, headers={"User-Agent": "PaySim-reproduction/2"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as output:  # nosec B310
            shutil.copyfileobj(response, output, length=1024 * 1024)
    output_csv = destination / CSV_NAME
    with zipfile.ZipFile(archive) as package:
        require(package.testzip() is None, "Corrupt ZIP: retry to a new destination")
        members = [m for m in package.infolist() if Path(m.filename).name == CSV_NAME]
        require(len(members) == 1, "Expected one exact CSV member")
        member = members[0]
        require(member.file_size == CSV_BYTES, "Unexpected raw member length")
        # Stream the selected member to a fixed local filename; never trust ZIP paths.
        with package.open(member) as source, output_csv.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    digest = file_hash(output_csv)
    require(digest == CSV_SHA256, "Source version differs: do not silently substitute")
    report = {"status": "PASS_SOURCE_HASH", "network_download_in_this_call": downloaded,
              "public_url": URL, "archive": str(archive.resolve()),
              "archive_sha256": file_hash(archive), "csv": str(output_csv.resolve()),
              "csv_sha256": digest, "csv_bytes": output_csv.stat().st_size}
    write_json(destination / "ACQUISITION_REPORT.json", report)
    return report
