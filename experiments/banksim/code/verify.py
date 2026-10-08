"""Run static checks and artificial-data tests; never call scientific training."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed local validation commands only
import sys
import xml.etree.ElementTree as ET  # nosec B405 - reads locally generated pytest XML only
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from banksim.common import write_json  # noqa: E402
from banksim.workflow import source_records  # noqa: E402


def main() -> int:
    sequence = len(list((ROOT / "audit").glob("validation_attempt_*"))) + 1
    folder = ROOT / "audit" / f"validation_attempt_{sequence:02d}"
    folder.mkdir()
    baseline = source_records(ROOT)
    junit = folder / "pytest.xml"
    commands = [
        ("dependencies", ["-m", "pip", "check"]),
        ("ruff", ["-m", "ruff", "check", "src", "tests", "run.py", "verify.py"]),
        ("mypy", ["-m", "mypy", "src", "run.py", "verify.py"]),
        ("bandit", ["-m", "bandit", "-r", "src", "run.py", "verify.py", "-ll"]),
        ("pytest", ["-m", "pytest", "--junitxml", str(junit), "--basetemp", str(folder / "temporary"), "-q"]),
    ]
    results = []
    for name, args in commands:
        completed = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)  # nosec B603 - static allowlist; shell=False
        (folder / f"{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        results.append({"name": name, "exit_code": completed.returncode, "log": str((folder / f"{name}.log").relative_to(ROOT))})
        print(name, "PASS" if completed.returncode == 0 else "FAIL", flush=True)
    counts = {}
    if junit.exists():
        tree = ET.parse(junit)  # nosec B314 - produced by the immediately preceding local pytest run
        counts = {k: sum(int(node.get(k, "0")) for node in tree.findall("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    unchanged = baseline == source_records(ROOT)
    passed = all(r["exit_code"] == 0 for r in results) and unchanged and counts.get("tests", 0) > 0 and counts.get("skipped", 1) == 0
    report = {"status": "PASS" if passed else "FAIL", "checks": results, "pytest_counts": counts, "source_records": baseline, "source_unchanged_during_validation": unchanged, "data_scope": "hand-constructed artificial fixtures only; real estimator fit methods blocked", "real_training_performed": False}
    write_json(folder / "summary.json", report)
    # Rechecking a frozen run must not replace the evidence sealed in FREEZE.json.
    if not (ROOT / "protocol/FREEZE.json").exists():
        write_json(ROOT / "audit/VALIDATION.json", report)
    print(json.dumps({"status": report["status"], "counts": counts, "report": str(folder / "summary.json")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
