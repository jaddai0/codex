"""Repeatable E0 repository fixtures with a retained failing baseline."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import uuid

from .storage import sha256_file, write_json


def small_repository_review_prompt(manifest_path: Path) -> str:
    return (
        "Read-only independent review of the installed Mavis repair in this repository. "
        f"Read {manifest_path} and its baseline log. Check git status and diff, run "
        "python3 -m unittest discover -s tests -q, verify user-notes.txt still matches "
        "the manifest hash, and assess whether package/pricing.py correctly fixes the "
        "seeded failure without hiding it. Do not edit files. Begin your final response "
        "with ACCEPT or REJECT, followed by concise evidence."
    )


def prepare_small_repository(home: Path) -> Path:
    task_root = Path(home) / "evaluations" / "e0" / "tasks" / uuid.uuid4().hex
    repo = task_root / "repo"
    (repo / "package").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "package" / "__init__.py").write_text("")
    (repo / "package" / "pricing.py").write_text(
        "def apply_discount(subtotal: int, discount: int) -> int:\n"
        "    return subtotal + discount\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_pricing.py").write_text(
        "import unittest\n"
        "from package.pricing import apply_discount\n\n"
        "class PricingTests(unittest.TestCase):\n"
        "    def test_discount_reduces_price(self):\n"
        "        self.assertEqual(apply_discount(10, 2), 8)\n\n"
        "    def test_zero_discount(self):\n"
        "        self.assertEqual(apply_discount(5, 0), 5)\n",
        encoding="utf-8",
    )
    notes = repo / "user-notes.txt"
    notes.write_text("User notes: preserve local edits.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", "package", "tests", "user-notes.txt"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Mavis E0",
            "-c",
            "user.email=mavis@local.invalid",
            "commit",
            "-qm",
            "Seed failing pricing test",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    notes.write_text(
        notes.read_text() + "Private working note: keep this line exactly.\n",
        encoding="utf-8",
    )
    test_command = ["python3", "-m", "unittest", "discover", "-s", "tests", "-q"]
    baseline = subprocess.run(
        test_command, cwd=repo, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if baseline.returncode == 0 or "test_discount_reduces_price" not in baseline.stderr:
        raise RuntimeError(
            "E0 small repository does not have the expected failing baseline"
        )
    baseline_path = task_root / "baseline-test.log"
    baseline_path.write_text(baseline.stdout + baseline.stderr, encoding="utf-8")
    manifest = task_root / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mavis.e0-small-repository/v1",
            "repo": str(repo.resolve()),
            "starting_revision": revision,
            "owned_paths": ["package/pricing.py"],
            "protected_dirty_file": str(notes.resolve()),
            "protected_dirty_sha256": sha256_file(notes),
            "baseline_exit_status": baseline.returncode,
            "baseline_log": str(baseline_path.resolve()),
            "baseline_log_sha256": sha256_file(baseline_path),
            "test_command": test_command,
        },
    )
    return manifest
