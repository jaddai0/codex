"""Evidence boundaries for the librarian and output reader helpers.

These interfaces validate helper output without running a model. The host keeps
the source archive and command receipt authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .storage import read_json, sha256_file
from .transcripts import TranscriptArchive


class LibrarianEvidence:
    """Expose only hash-verified archive lines and validate cited answers."""

    def __init__(self, archive: TranscriptArchive):
        self.archive = archive

    def search(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be 1..200")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be nonnegative")
        return self.archive.search(query, limit=limit, offset=offset)

    def validate_answer(
        self, answer: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reject citations outside the retrieved, verified evidence packet.

        This proves source identity, not that a model's interpretation is true.
        """
        if not isinstance(answer, dict):
            raise ValueError("librarian answer must be an object")
        text = answer.get("answer")
        uncertainty = answer.get("uncertainty")
        citations = answer.get("citations")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("librarian answer must contain text")
        if not isinstance(uncertainty, str) or not uncertainty.strip():
            raise ValueError("librarian answer must state uncertainty")
        if not isinstance(citations, list) or not citations:
            raise ValueError("librarian answer must cite source lines")
        if not self.archive.manifest_path.is_file():
            raise ValueError("transcript archive has no manifest")
        manifest = read_json(self.archive.manifest_path)
        archived = set()
        for item in manifest["segments"]:
            try:
                resolved = self.archive.resolve_segment_file(item["path"])
            except ValueError:
                continue
            archived.add((str(resolved), item["sha256"]))
        allowed = set()
        for item in evidence:
            if (
                not isinstance(item, dict)
                or not {"path", "line", "sha256", "text"} <= item.keys()
            ):
                continue
            if (item["path"], item["sha256"]) not in archived:
                continue
            try:
                path = self.archive.resolve_segment_file(item["path"])
            except ValueError:
                continue
            if sha256_file(path) != item["sha256"]:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            line = item["line"]
            if (
                isinstance(line, int)
                and 1 <= line <= len(lines)
                and lines[line - 1] == item["text"]
            ):
                allowed.add((item["path"], line, item["sha256"]))
        for item in citations:
            if not isinstance(item, dict) or set(item) != {"path", "line", "sha256"}:
                raise ValueError("citation shape is invalid")
            line = item["line"]
            if isinstance(line, bool):
                raise ValueError("citation line must be an integer")
            key = (item["path"], line, item["sha256"])
            if key not in allowed:
                raise ValueError("citation is outside the verified evidence packet")
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError("cited source changed after retrieval")
        return answer


FAILURE_LINE = re.compile(r"\b(?:FAILED|ERROR|FAIL|not ok|Traceback)\b", re.IGNORECASE)
ZERO_FAILURE_COUNT = re.compile(r"\b0\s+(?:failed|errors?)\b", re.IGNORECASE)
POSITIVE_FAILURE_COUNT = re.compile(r"\b[1-9]\d*\s+(?:failed|errors?)\b", re.IGNORECASE)
COUNT = re.compile(
    r"\b(?:\d+\s+(?:passed|failed|errors?|tests?|skipped)|Ran\s+\d+\s+tests?)\b",
    re.IGNORECASE,
)


def _is_failure_line(line: str) -> bool:
    # Known test summaries may report "0 failed" beside a passing count.
    # Strip only those zero counts; an explicit marker or positive count on
    # the same line still signals failure.
    return bool(
        POSITIVE_FAILURE_COUNT.search(line)
        or FAILURE_LINE.search(ZERO_FAILURE_COUNT.sub("", line))
    )


class OutputReader:
    """Build a lossless safety envelope around an optional model summary."""

    @staticmethod
    def read(
        receipt_path: Path, proposed: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        receipt_path = Path(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "mavis.evidence-receipt/v1":
            raise ValueError("unsupported evidence receipt")
        raw = receipt.get("raw_output")
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ValueError("receipt has no raw output reference")
        directory = Path(raw["path"])
        stdout, stderr = directory / "stdout.log", directory / "stderr.log"
        if not stdout.is_file() or not stderr.is_file():
            raise ValueError("raw output is missing")
        actual_hash = sha256_file(stdout) + ":" + sha256_file(stderr)
        if actual_hash != raw.get("sha256"):
            raise ValueError("raw output changed after receipt")
        if stdout.stat().st_size + stderr.stat().st_size != raw.get("bytes"):
            raise ValueError("raw output size differs from receipt")
        lines = (
            stdout.read_text(encoding="utf-8", errors="replace")
            + "\n"
            + stderr.read_text(encoding="utf-8", errors="replace")
        ).splitlines()
        failures = [
            {"line": index, "text": line}
            for index, line in enumerate(lines, 1)
            if _is_failure_line(line)
        ]
        counts = [
            {"line": index, "text": line}
            for index, line in enumerate(lines, 1)
            if COUNT.search(line)
        ]
        verdict = receipt.get("verdict")
        if verdict not in {"pass", "fail", "uncertain", "incomplete"}:
            raise ValueError("receipt has no valid host verdict")
        if receipt.get("timed_out"):
            verdict = "incomplete"
        elif receipt.get("exit_status") != 0 or failures:
            verdict = "fail"
        if proposed is not None:
            if not isinstance(proposed, dict) or proposed.get("verdict") != verdict:
                raise ValueError("output reader cannot change the host verdict")
            if not isinstance(proposed.get("summary"), str):
                raise ValueError("output reader summary must be text")
            if verdict != "pass" and proposed["summary"]:
                raise ValueError(
                    "output reader cannot replace a non-passing result with prose"
                )
        return {
            "verdict": verdict,
            "exit_status": receipt["exit_status"],
            "timed_out": receipt["timed_out"],
            "failure_lines": failures,
            "count_lines": counts,
            "raw_output": raw,
            "summary": proposed["summary"] if proposed is not None else None,
        }
