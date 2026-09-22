"""Host-recorded command execution and deterministic result parsing."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import uuid
from typing import Iterable

from .storage import require_safe_id, sha256_file, write_json


FAIL_MARKERS = (
    re.compile(r"\bFAILED\b"),
    re.compile(r"\bERRORS?\b"),
    re.compile(r"\b[1-9][0-9]* failed\b", re.IGNORECASE),
    re.compile(r"\bnot ok\b", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)"),
)
PASS_MARKERS = (
    re.compile(r"\bOK\b"),
    re.compile(r"\b[1-9][0-9]* passed\b", re.IGNORECASE),
    re.compile(r"\bok\b", re.IGNORECASE),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_test_output(text: str, exit_status: int, timed_out: bool = False) -> str:
    if timed_out:
        return "incomplete"
    if exit_status != 0 or any(marker.search(text) for marker in FAIL_MARKERS):
        return "fail"
    if any(marker.search(text) for marker in PASS_MARKERS):
        return "pass"
    return "uncertain"


def run_command(
    home: Path,
    objective_id: str,
    command: list[str],
    cwd: Path,
    *,
    artifact_paths: Iterable[Path] = (),
    acceptance_check_ids: Iterable[str] = (),
    timeout: float | None = None,
) -> Path:
    require_safe_id(objective_id, "objective id")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("command must be a non-empty argument list")
    receipt_id = uuid.uuid4().hex
    evidence_dir = Path(home) / "evidence" / objective_id / receipt_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = evidence_dir / "stdout.log"
    stderr_path = evidence_dir / "stderr.log"
    started_at = _now()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, env=os.environ.copy())
        try:
            exit_status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                exit_status = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_status = process.wait()
    finished_at = _now()
    combined = stdout_path.read_text(encoding="utf-8", errors="replace") + "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
    artifact_hashes = {
        str(path): sha256_file(path) for path in artifact_paths if Path(path).is_file()
    }
    receipt = {
        "schema_version": "mavis.evidence-receipt/v1",
        "receipt_id": receipt_id,
        "objective_id": objective_id,
        "command": command,
        "cwd": str(Path(cwd).resolve()),
        "exit_status": exit_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "raw_output": {
            "path": str(evidence_dir.resolve()),
            "sha256": sha256_file(stdout_path) + ":" + sha256_file(stderr_path),
            "bytes": stdout_path.stat().st_size + stderr_path.stat().st_size,
        },
        "changed_revision": git_revision(Path(cwd)),
        "artifact_hashes": artifact_hashes,
        "acceptance_check_ids": sorted(
            {require_safe_id(item, "acceptance check id") for item in acceptance_check_ids}
        ),
        "producer": "mavis-host-command/v1",
        "verdict": parse_test_output(combined, exit_status, timed_out),
    }
    receipt_path = evidence_dir / "receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path
