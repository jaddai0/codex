import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from mavis.cli import build_parser, main
from mavis.project_memory import ProjectMemory
from mavis.retrieval import ProjectIndex


class ProjectMemoryTests(unittest.TestCase):
    def make_repo(self, root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Mavis Test"], cwd=root, check=True
        )
        (root / "decision.md").write_text("The team chose a stable release.\n")
        subprocess.run(["git", "add", "decision.md"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    def test_author_all_kinds_with_append_only_versions_and_ignored_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            memory = ProjectMemory(root)
            for kind in ("decision", "failure", "tool-recipe", "history"):
                record = memory.add(kind, kind, f"Remember {kind}", ["decision.md"])
                self.assertEqual(record["record_version"], 1)
                self.assertEqual(record["verification_state"], "unverified")
            revised = memory.add(
                "decision", "decision", "Remember revised choice", ["decision.md"]
            )
            self.assertEqual(revised["record_version"], 2)
            self.assertEqual(memory.search("Remember decision"), [])
            self.assertEqual(memory.search("revised choice")[0]["record_version"], 2)
            self.assertEqual(
                len(list((root / ".mavis/memory/records").rglob("v*.json"))), 5
            )
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", ".mavis/memory/records"],
                cwd=root,
                check=False,
            )
            self.assertEqual(ignored.returncode, 0)

    def test_project_memory_precedes_verified_global_with_paging_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            self.make_repo(root)
            shared = Path(directory) / "shared"
            memory = ProjectMemory(root)
            for number in range(3):
                memory.add(
                    f"memory-{number}",
                    "decision",
                    f"needle decision {number}",
                    ["decision.md"],
                )
            source = Path(directory) / "global-source.txt"
            source.write_text("global source")
            knowledge = shared / "knowledge"
            knowledge.mkdir(parents=True)
            (knowledge / "global.json").write_text(
                json.dumps(
                    {
                        "schema_version": "mavis.memory-record/v1",
                        "scope": "global-coding",
                        "verification_state": "verified",
                        "claim": "needle global",
                        "source_references": [
                            {
                                "path": str(source),
                                "sha256": hashlib.sha256(
                                    source.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                )
            )
            index = ProjectIndex(root, shared)
            hits = index.search("needle", limit=2, offset=0) + index.search(
                "needle", limit=2, offset=2
            )
            self.assertEqual(
                [hit["source"] for hit in hits],
                ["project-memory"] * 3 + ["verified-global"],
            )
            self.assertEqual(hits[0]["source_references"][0]["path"], "decision.md")
            self.assertEqual(hits[0]["verification_state"], "unverified")
            self.assertEqual(hits[0]["branch"], memory.branch())
            self.assertIn("observed_at", hits[0]["freshness"])

    def test_search_citation_is_the_exact_claim_line_in_saved_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            claim = 'Decision: use "stable" café and retry.'
            ProjectMemory(root).add("quoted", "decision", claim, ["decision.md"])
            hit = ProjectIndex(root, root / "shared").search("café")[0]
            self.assertEqual(hit["claim"], claim)
            saved_lines = Path(hit["path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(saved_lines[hit["line"] - 1], hit["text"])
            self.assertIn('"claim":', hit["text"])
            self.assertEqual(json.loads(Path(hit["path"]).read_text())["claim"], claim)

    def test_equal_claims_collapse_before_paging_with_project_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            self.make_repo(root)
            shared = Path(directory) / "shared"
            memory = ProjectMemory(root)
            memory.add("z-record", "decision", "Needle stable choice", ["decision.md"])
            memory.add(
                "a-record", "history", " needle  STABLE choice ", ["decision.md"]
            )
            memory.add("other", "failure", "Needle separate failure", ["decision.md"])
            source = Path(directory) / "global-source.txt"
            source.write_text("source")
            knowledge = shared / "knowledge"
            knowledge.mkdir(parents=True)
            for name, claim in (
                ("duplicate", "Needle stable choice"),
                ("unique", "Needle global fact"),
            ):
                (knowledge / f"{name}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "mavis.memory-record/v1",
                            "scope": "global-coding",
                            "verification_state": "verified",
                            "claim": claim,
                            "source_references": [
                                {
                                    "path": str(source),
                                    "sha256": hashlib.sha256(
                                        source.read_bytes()
                                    ).hexdigest(),
                                }
                            ],
                        }
                    )
                )
            index = ProjectIndex(root, shared)
            hits = index.search("needle", limit=2) + index.search(
                "needle", limit=2, offset=2
            )
            self.assertEqual(
                [hit["source"] for hit in hits],
                ["project-memory", "project-memory", "verified-global"],
            )
            self.assertEqual(
                [hit.get("record_id") for hit in hits[:2]], ["a-record", "other"]
            )
            self.assertEqual(hits[-1]["text"], "Needle global fact")

    def test_changed_deleted_source_and_branch_switch_invalidate_without_resurrection(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            memory = ProjectMemory(root)
            memory.add("choice", "decision", "retain safe choice", ["decision.md"])
            self.assertEqual(len(memory.search("safe choice")), 1)
            original_branch = memory.branch()
            subprocess.run(["git", "checkout", "-qb", "other"], cwd=root, check=True)
            self.assertEqual(memory.search("safe choice"), [])
            memory.add("choice", "decision", "other branch choice", ["decision.md"])
            self.assertEqual(len(memory.search("other branch choice")), 1)
            subprocess.run(
                ["git", "checkout", "-q", original_branch], cwd=root, check=True
            )
            self.assertEqual(len(memory.search("safe choice")), 1)
            (root / "decision.md").write_text("changed decision\n")
            self.assertEqual(memory.search("safe choice"), [])
            (root / "decision.md").unlink()
            self.assertEqual(memory.search("safe choice"), [])

    def test_rewritten_branch_history_invalidates_older_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            memory = ProjectMemory(root)
            memory.add("choice", "decision", "old lineage choice", ["decision.md"])
            first = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            subprocess.run(
                ["git", "commit", "--allow-empty", "-qm", "later"], cwd=root, check=True
            )
            memory.add("second", "history", "later lineage fact", ["decision.md"])
            self.assertEqual(len(memory.search("lineage")), 2)
            subprocess.run(
                ["git", "reset", "--hard", "-q", first], cwd=root, check=True
            )
            self.assertEqual(
                [hit["record_id"] for hit in memory.search("lineage")], ["choice"]
            )

    def test_rejects_unsafe_sources_and_forged_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            self.make_repo(root)
            memory = ProjectMemory(root)
            outside = Path(directory) / "outside.txt"
            outside.write_text("private")
            (root / "linked.md").symlink_to(outside)
            (root / ".gitignore").write_text("private.txt\n")
            (root / "private.txt").write_text("private")
            for source in (
                "../outside.txt",
                str(outside),
                "linked.md",
                ".mavis/.gitignore",
                ".git/config",
                "private.txt",
                "",
            ):
                with self.assertRaises(ValueError):
                    memory.add("unsafe", "decision", "private claim", [source])
            record = memory.add("safe", "decision", "supported claim", ["decision.md"])
            file = next((root / ".mavis/memory/records").rglob("v1.json"))
            forged = {
                **record,
                "source_references": [{"path": "decision.md", "sha256": "0" * 64}],
            }
            file.write_text(json.dumps(forged))
            self.assertEqual(memory.search("supported claim"), [])
            forged["source_references"] = record["source_references"]
            forged["verification_state"] = "verified"
            file.write_text(json.dumps(forged))
            self.assertEqual(memory.search("supported claim"), [])
            forged["verification_state"] = "unverified"
            forged["source_references"] = [
                {
                    "path": "../outside.txt",
                    "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                }
            ]
            file.write_text(json.dumps(forged))
            self.assertEqual(memory.search("supported claim"), [])
            file.unlink()
            file.symlink_to(outside)
            self.assertEqual(memory.search("supported claim"), [])

    def test_cli_add_exposes_authoring_with_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            args = build_parser().parse_args(
                [
                    "project-memory",
                    "--project",
                    str(root),
                    "add",
                    "runbook",
                    "tool-recipe",
                    "Use the stable test command",
                    "--source",
                    "decision.md",
                ]
            )
            self.assertEqual(
                (args.memory_command, args.kind, args.source),
                ("add", "tool-recipe", ["decision.md"]),
            )
            self.assertEqual(
                main(
                    [
                        "project-memory",
                        "--project",
                        str(root),
                        "add",
                        "runbook",
                        "tool-recipe",
                        "Use the stable test command",
                        "--source",
                        "decision.md",
                    ]
                ),
                0,
            )
            self.assertEqual(
                ProjectIndex(root, root / "shared").search("stable test")[0][
                    "record_id"
                ],
                "runbook",
            )


if __name__ == "__main__":
    unittest.main()
