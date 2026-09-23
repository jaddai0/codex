from pathlib import Path
import hashlib
import json
import copy
import importlib.util
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis import e1_bootstrap, e1_bootstrap_gateway
from mavis.e1 import E1Runner
from mavis.evaluations import E0_CASES
from mavis.experiments import (
    candidate_assignment_requirements,
    review_assignment_requirements,
)
from mavis.evidence import run_command
from mavis.package_provenance import package_tree_sha256
from mavis.storage import read_json, sha256_file, write_json


class E1RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "source"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test"], check=True
        )
        (self.repo / "result.txt").write_text("fail\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "result.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True
        )
        revision = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        check = {
            "id": "accept",
            "argv": [
                "python3",
                "-c",
                "from pathlib import Path; import sys; good=Path('result.txt').read_text().strip()=='pass'; print('OK' if good else 'FAILED'); sys.exit(0 if good else 1)",
            ],
            "timeout_seconds": 10,
        }
        failed_receipt = run_command(
            self.root / "failure-home",
            "original-failure",
            check["argv"],
            self.repo,
            acceptance_check_ids=["accept"],
            timeout=10,
        )
        cases = [
            {
                "id": id_,
                "source": str(self.repo),
                "revision": revision,
                "task": f"Repair {id_}",
                "checks": [check],
            }
            for id_ in ("regression", "held-a", "held-b")
        ]
        self.manifest = self.root / "cases.json"
        write_json(
            self.manifest,
            {
                "schema_version": "mavis.e1-cases/v1",
                "scoring": "held_out_pass_fraction/v1",
                "regression": "regression",
                "failure_receipt": {
                    "path": str(failed_receipt),
                    "sha256": sha256_file(failed_receipt),
                },
                "held_out": ["held-a", "held-b"],
                "minimum_gain": 0.5,
                "cases": cases,
            },
        )
        self.home = self.root / "home"
        self.runner = E1Runner(self.home)
        base = {"prompts": {"system": "A"}, "tool_settings": {}, "retrieval": {}}
        self.runner.store.seed_active("main", base)
        base["prompts"]["system"] = "B"
        self.candidate = self.root / "candidate.json"
        write_json(self.candidate, base)

    def _freeze(self):
        return self.runner.freeze(
            "repair",
            "main",
            "prompts",
            self.candidate,
            self.manifest,
            "Fix a real omission",
        )

    def _pass_candidate(self, case_id):
        checkout = self.home / "e1" / "repair" / "checkouts" / "candidate" / case_id
        (checkout / "result.txt").write_text("pass\n")
        subprocess.run(["git", "-C", str(checkout), "add", "result.txt"], check=True)
        subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test",
                        "-c", "user.email=test@example.invalid", "commit", "-qm", "repair"], check=True)
        return checkout

    def _trial_fixture(self, arm, case_id):
        """Construct a host-shaped receipt without invoking a core or model."""
        record = self.runner.store.load("repair")
        case = next(case for case in read_json(self.manifest)["cases"] if case["id"] == case_id)
        root = self.home / "e1" / "repair"
        checkout = root / "checkouts" / arm / case_id
        result = read_json(root / "results" / arm / f"{case_id}.json")
        runtime = root / "runtime" / arm / case_id
        runtime.mkdir(parents=True, exist_ok=True)
        share = self.root / "installed-share"
        share.mkdir(exist_ok=True)
        core = share / "local-codex-core"
        core.write_text("fixture core")
        template = share / "base-instructions.md"
        template.write_text("Base instructions\n")
        trial_source = share / "trial_runtime.py"
        trial_source.write_text("fixture runtime")
        for name in ("launch_core.py", "prepare_runtime.py", "generation_lease.py", "persona.toml"):
            (share / name).write_text("fixture " + name)
        package_dir = share / "mavis"
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").write_text("# fixture package\n")
        committed_package = self.repo / "local-codex/mavis/mavis/__init__.py"
        if not committed_package.exists():
            committed_package.parent.mkdir(parents=True)
            committed_package.write_bytes((package_dir / "__init__.py").read_bytes())
            subprocess.run(["git", "-C", str(self.repo), "add", "local-codex/mavis/mavis/__init__.py"], check=True)
            subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "fixture package"], check=True)
        source_revision = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        launcher = self.root / "installed-mavis-command"
        launcher.write_text("fixture launcher")
        package_path = share / "install-manifest.json"
        write_json(package_path, {
            "schema_version": "mavis.installed-core/v1",
            "core_binary": str(core), "core_sha256": sha256_file(core),
            "launcher": str(launcher), "launcher_sha256": sha256_file(launcher),
            "trial_runtime_sha256": sha256_file(trial_source),
            "launch_core_sha256": sha256_file(share / "launch_core.py"),
            "prepare_runtime_sha256": sha256_file(share / "prepare_runtime.py"),
            "generation_lease_sha256": sha256_file(share / "generation_lease.py"),
            "base_instructions_sha256": sha256_file(template),
            "persona_sha256": sha256_file(share / "persona.toml"),
            "mavis_package_sha256": package_tree_sha256(package_dir),
            "source_repository": str(self.repo), "source_revision": source_revision,
        })
        profile = self.home / "profiles" / "main" / "v1.json"
        write_json(profile, {"profile_id": "accepted", "role": "main", "status": "active",
                             "model_identity": {"model_id": "exact-model"}})
        write_json(profile.parent / "active.json", {"version": 1, "path": str(profile)})
        prompt = read_json(Path(record[arm]["path"]))["prompts"]["system"]
        instructions = runtime / "accepted-model-instructions.md"
        instructions.write_text("Base instructions\n\n" + prompt + "\n")
        catalog = runtime / "omlx-models.json"
        catalog.write_text("{}\n")
        config = runtime / "config.toml"
        config.write_text('model = "exact-model"\n[model_providers.omlx]\nbase_url = "http://127.0.0.1:8001/v1"\n')
        stdout = runtime / "stdout.log"
        stderr = runtime / "stderr.log"
        stdout.write_text("ran\n")
        stderr.write_text("")
        transcript = runtime / "rollout.jsonl"
        session = f"session-{arm}-{case_id}"
        transcript.write_text(json.dumps({"type": "session_meta", "payload": {"id": session}}) + "\n")
        trial_id = f"repair-{arm}-{case_id}"
        observation = runtime / "effective-config.json"
        write_json(observation, {
            "schema_version": "mavis.e1-effective-config/v1", "trial_id": trial_id,
            "model": "exact-model", "model_provider": "omlx",
            "catalog_sha256": sha256_file(catalog),
            "base_instructions_sha256": sha256_file(instructions),
            "session_id": session, "rollout_path": str(transcript),
        })
        argv = [str(core), "exec", "-C", str(checkout.resolve()), "--", case["task"]]
        overrides = [f"{key}={json.dumps(value)}" for key, value in {
            "model": "exact-model", "model_provider": "omlx",
            "model_catalog_json": str(catalog), "model_instructions_file": str(instructions),
        }.items()]
        effective_argv = [argv[0], *(item for override in overrides for item in ("-c", override)), *argv[1:]]
        digest_argv = lambda args: hashlib.sha256(json.dumps(args, separators=(",", ":")).encode()).hexdigest()
        receipt = {
            "schema_version": "mavis.e1-trial-launch/v1", "state": "exited", "core_exit_code": 0,
            "termination_signal": None, "observation_status": "matched_startup_config",
            "profile_source": "accepted-main",
            "experiment_id": "repair", "arm": arm, "case_id": case_id, "trial_id": trial_id,
            "task": case["task"], "task_sha256": hashlib.sha256(case["task"].encode()).hexdigest(),
            "core_argv": argv, "core_argv_sha256": digest_argv(argv),
            "binding_overrides": overrides, "effective_argv": effective_argv,
            "effective_argv_sha256": digest_argv(effective_argv),
            "manifest_sha256": record["workload"]["manifest_sha256"],
            "snapshot_path": record[arm]["path"], "snapshot_sha256": record[arm]["sha256"],
            "accepted_profile_id": "accepted", "accepted_profile_path": str(profile),
            "accepted_profile_sha256": sha256_file(profile),
            "checkout": str(checkout.resolve()), "starting_revision": case["revision"],
            "resulting_revision": result["checked_revision"], "checkout_dirty": False,
            "runtime_home": str(runtime.resolve()), "mavis_home": str(self.home.resolve()),
            "core_binary": str(core), "core_sha256": sha256_file(core),
            "package_manifest": str(package_path), "package_manifest_sha256": sha256_file(package_path),
            "core_provenance": "installed-package", "selected_model": "exact-model", "model_provider": "omlx",
            "model_endpoint": "http://127.0.0.1:8001/v1",
            "config_path": str(config), "config_sha256": sha256_file(config),
            "catalog_path": str(catalog), "catalog_sha256": sha256_file(catalog),
            "instructions_path": str(instructions), "instructions_sha256": sha256_file(instructions),
            "effective_system_prompt_sha256": sha256_file(instructions),
            "observation_path": str(observation), "observation_sha256": sha256_file(observation),
            "stdout_path": str(stdout), "stdout_sha256": sha256_file(stdout),
            "stderr_path": str(stderr), "stderr_sha256": sha256_file(stderr),
            "transcript_path": str(transcript), "transcript_sha256": sha256_file(transcript),
        }
        path = root / "trials" / arm / f"{case_id}.json"
        write_json(path, receipt)
        return path

    def test_complete_comparison_retains_raw_receipts_and_coverage(self):
        frozen = self._freeze()
        self.assertEqual(len(frozen["candidate_requirements"]), 1)
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.runner.compare("repair", "native-job", self.manifest)
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            self._pass_candidate(case)
            result = self.runner.check("repair", "candidate", case)
            self.assertTrue(result["passed"])
            receipt = read_json(Path(result["checks"][0]["receipt"]))
            self.assertTrue(
                (Path(receipt["raw_output"]["path"]) / "stdout.log").is_file()
            )
        self.assertEqual(self.runner.coverage("repair")["state"], "complete")
        report = self.root / "native-report.json"
        report.write_text('{"native":"placeholder requiring gateway review"}')
        record = self.runner.compare("repair", "native-job", report)
        self.assertEqual(record["state"], "compared")
        self.assertEqual(record["comparison"]["baseline"]["target_score"], 0)
        self.assertEqual(record["comparison"]["candidate"]["target_score"], 1)
        self.assertEqual(
            self.runner.store.active("main")["configuration"], record["baseline"]
        )

    def test_changed_manifest_fails_closed(self):
        self._freeze()
        frozen = self.home / "e1" / "repair" / "cases.json"
        frozen.write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.runner.coverage("repair")

    def test_changed_original_failure_output_fails_closed(self):
        failure = read_json(self.manifest)["failure_receipt"]
        receipt = read_json(Path(failure["path"]))
        (Path(receipt["raw_output"]["path"]) / "stdout.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "host receipt"):
            self._freeze()

    def test_original_failure_must_be_from_case_source(self):
        manifest = read_json(self.manifest)
        path = Path(manifest["failure_receipt"]["path"])
        receipt = read_json(path)
        receipt["cwd"] = str(self.root)
        write_json(path, receipt)
        manifest["failure_receipt"]["sha256"] = sha256_file(path)
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "host receipt"):
            self._freeze()

    def test_review_stage_and_promotion_recheck_underlying_raw_output(self):
        self._freeze()
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            self._pass_candidate(case)
            self.runner.check("repair", "candidate", case)
        report = self.root / "native-report.json"
        report.write_text('{"report":"native fixture"}')
        record = self.runner.compare("repair", "candidate-job", report)
        receipt_path = self.home / "verifications" / "experiments" / "repair.json"
        write_json(
            receipt_path,
            {
                "schema_version": "mavis.experiment-review/v1",
                "experiment_id": "repair",
                "comparison_digest": record["comparison"]["comparison_digest"],
                "baseline_sha256": record["baseline"]["sha256"],
                "candidate_sha256": record["candidate"]["sha256"],
                "candidate_job_id": "candidate-job",
                "gateway_worker_job_id": "review-job",
                "verifier_job_id": "terra-review",
                "verdict": "accepted",
            },
        )

        def gateway_status(job_id):
            candidate = job_id == "candidate-job"
            return {
                "job_id": job_id,
                "state": "completed",
                "exit_code": 0,
                "accepted": True,
                "receipt": {"job_id": job_id, "exit_code": 0},
                "acceptance": {
                    "accepted": True,
                    "job_id": job_id,
                    "verifier": "terra",
                    "verifier_job_id": "terra-candidate"
                    if candidate
                    else "terra-review",
                    "target_sha256": "a" * 64,
                    "evidence_sha256": "b" * 64,
                    "report_sha256_on_disk": "c" * 64,
                    "verifier_verdict_sha256": "d" * 64,
                },
                "mavis_binding": {
                    "objective_id": "repair",
                    "requirements": candidate_assignment_requirements(record)
                    if candidate
                    else review_assignment_requirements(record),
                    "report_sha256": record["comparison"]["candidate"]["evidence"][
                        "sha256"
                    ]
                    if candidate
                    else sha256_file(receipt_path),
                },
            }

        self.runner.store.gateway_status_reader = gateway_status
        self.runner.store.review("repair", receipt_path)
        case_result = read_json(
            self.home / "e1" / "repair" / "results" / "baseline" / "held-a.json"
        )
        host_receipt = Path(case_result["checks"][0]["receipt"])
        stdout = host_receipt.parent / "stdout.log"
        original = stdout.read_bytes()
        stdout.write_bytes(b"tampered after review")
        with self.assertRaisesRegex(
            ValueError, "host receipt|underlying E1|raw output"
        ):
            self.runner.store.stage("repair")
        stdout.write_bytes(original)
        self.runner.store.stage("repair")
        stdout.write_bytes(b"tampered after staging")
        with self.assertRaisesRegex(
            ValueError, "host receipt|underlying E1|raw output"
        ):
            self.runner.store.promote("repair", between_objectives=True)

    def test_manifest_rejects_overlapping_split(self):
        manifest = read_json(self.manifest)
        manifest["held_out"] = ["regression", "held-a"]
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "separate"):
            self._freeze()

    def test_native_candidate_dispatch_and_compare_use_gateway_report(self):
        self._freeze()
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        job_dir = self.root / "gateway-job"
        job_dir.mkdir()
        report = job_dir / "report.md"
        report.write_text("Native worker report\n")
        assignments = []

        def start(arguments):
            assignments.append(arguments)
            write_json(job_dir / "assignment.json", {key: value for key, value in arguments.items() if key != "job_id"})
            return {"success": True, "started": True, "accepted": False,
                    "job_id": arguments["job_id"], "job_dir": str(job_dir),
                    "report_path": str(report), "assignment_path": str(job_dir / "assignment.json")}

        self.runner.gateway_assignment_starter = start
        dispatch = self.runner.dispatch_candidate(
            "repair", "regression", job_id="candidate-native", lane="minimax",
            model="minimax/MiniMax-M3", task="Repair the frozen case")
        self.assertEqual(assignments[0]["mavis_requirements"],
                         candidate_assignment_requirements(self.runner.store.load("repair")))
        self.assertEqual(assignments[0]["cwd"], dispatch["checkout"])
        self.assertEqual(assignments[0]["mavis_owner"],
                         {"provider": "minimax", "model": "minimax/MiniMax-M3", "harness": "opencode"})
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            self._pass_candidate(case)
            self.runner.check("repair", "candidate", case)
        binding = {"objective_id": "repair", "starting_revision": dispatch["starting_revision"],
                   "cwd": dispatch["checkout"], "requirements": assignments[0]["mavis_requirements"],
                   "report_sha256": sha256_file(report),
                   "assignment_sha256": dispatch["assignment_sha256"],
                   "changed_revision": dispatch["starting_revision"],
                   "owned_paths": [dispatch["checkout"]],
                   "owner": assignments[0]["mavis_owner"],
                   "required_checks": ["accept"]}
        self.runner.store.gateway_status_reader = lambda job_id: {
            "job_id": job_id, "state": "completed", "exit_code": 0,
            "accepted": True, "mavis_binding": binding,
            "receipt": {"job_id": job_id, "exit_code": 0},
            "acceptance": {"accepted": True, "job_id": job_id, "verifier": "terra",
                           "verifier_job_id": "terra-job", "target_sha256": "a" * 64,
                           "evidence_sha256": "b" * 64, "report_sha256_on_disk": "c" * 64,
                           "verifier_verdict_sha256": "d" * 64}}
        with self.assertRaises(FileNotFoundError):
            self.runner.compare_native("repair", "regression")
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")
        self.assertFalse((self.home / "e1" / "repair" / "candidate-worker-report.json").exists())

    def _paired_trials(self):
        self._freeze()
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            self._pass_candidate(case)
            self.runner.check("repair", "candidate", case)
            self._trial_fixture("baseline", case)
            self._trial_fixture("candidate", case)

    def _paired_bootstrap_trials(self):
        """Use the real read-only bootstrap validator with local evidence fixtures."""
        self._paired_trials()
        (self.home / "profiles/main/active.json").unlink()
        package = self.root / "installed-share/install-manifest.json"
        fingerprint = {
            "core_sha256": read_json(package)["core_sha256"],
            "service_sha256": "b" * 64,
        }
        e0 = self.home / "evaluations/e0"
        case_hashes = {}
        for case in E0_CASES:
            path = e0 / f"{case}.json"
            write_json(
                path,
                {
                    "schema_version": "mavis.evaluation-case/v1",
                    "suite": "e0",
                    "case": case,
                    "status": "pass",
                    "evidence": ["host fixture"],
                },
            )
            case_hashes[case] = sha256_file(path)
        summary = e0 / "summary.json"
        write_json(
            summary,
            {
                "schema_version": "mavis.evaluation-suite/v1",
                "suite": "e0",
                "status": "pass",
                "mandatory_cases": list(E0_CASES),
                "results": [{"case": case, "status": "pass"} for case in E0_CASES],
                "installed_candidate": fingerprint,
                "model_id": "exact-model",
                "case_receipts": case_hashes,
            },
        )
        model_root = self.root / "models"
        model_path = model_root / "exact-model"
        model_path.mkdir(parents=True)
        write_json(
            model_path / "config.json",
            {
                "architectures": ["QwenFixture"],
                "quantization": {"bits": 4},
            },
        )
        write_json(model_path / "tokenizer.json", {"version": "fixture"})
        write_json(model_path / "tokenizer_config.json", {"bos_token": "<s>"})
        (model_path / "chat_template.jinja").write_text("{{ prompt }}")
        (model_path / "model-00001.safetensors").write_bytes(b"weights-one")
        owner = {"provider": "minimax", "model": "minimax/MiniMax-M3", "harness": "opencode"}
        assignment_path = self.home / "e1/bootstrap/review-assignment.json"
        review = self.home / "verifications/e1-bootstrap/main.json"
        gateway_dir = self.root / "bootstrap-gateway-job"
        gateway_dir.mkdir()
        gateway_assignment = gateway_dir / "assignment.json"
        gateway_report = gateway_dir / "report.md"

        def start(arguments):
            saved = {key: value for key, value in arguments.items() if key != "job_id"}
            saved.update(report_dir="", verifier_required=True)
            write_json(gateway_assignment, saved)
            return {
                "success": True,
                "started": True,
                "accepted": False,
                "job_id": arguments["job_id"],
                "job_dir": str(gateway_dir),
                "assignment_path": str(gateway_assignment),
                "report_path": str(gateway_report),
            }

        def status(job_id):
            target = "c" * 64
            return {
                "job_id": job_id,
                "state": "completed",
                "exit_code": 0,
                "accepted": True,
                "receipt": {"job_id": job_id, "exit_code": 0},
                "acceptance": {
                    "accepted": True,
                    "job_id": job_id,
                    "verifier": "terra",
                    "verifier_job_id": "review-job-terra",
                    "target_sha256": target,
                    "evidence_sha256": "b" * 64,
                    "report_sha256_on_disk": sha256_file(gateway_report),
                    "verifier_verdict_sha256": "d" * 64,
                },
                "mavis_binding": {
                    "schema_version": "model-gateway-mavis-objective-binding/v1",
                    "objective_id": "e1-bootstrap-main",
                    "cwd": assignment["cwd"],
                    "owner": owner,
                    "starting_revision": assignment["starting_revision"],
                    "changed_revision": assignment["starting_revision"],
                    "owned_paths": assignment["owned_paths"],
                    "requirements": assignment["requirements"],
                    "required_checks": assignment["required_checks"],
                    "assignment_sha256": sha256_file(gateway_assignment),
                    "report_sha256": sha256_file(gateway_report),
                    "target_sha256": target,
                },
            }

        for name, value in (
            ("installed_candidate_fingerprint", lambda: fingerprint),
            ("_package_manifest_path", lambda: package),
            ("_model_root_path", lambda: model_root),
            (
                "inventory",
                lambda _endpoint: [
                    {"id": "exact-model", "model_path": str(model_path.resolve())}
                ],
            ),
            ("harness_job_status", status),
        ):
            active_patch = patch.object(e1_bootstrap, name, value)
            active_patch.start()
            self.addCleanup(active_patch.stop)
        assignment = e1_bootstrap.prepare_bootstrap_review(
            self.home,
            owner,
            inventory_reader=lambda _: [
                {"id": "exact-model", "model_path": str(model_path.resolve())}
            ],
        )
        e1_bootstrap.dispatch_bootstrap_review(self.home, "review-job", starter=start)
        baseline = self.runner.store.active("main")["configuration"]
        write_json(
            gateway_report,
            {
                "schema_version": "mavis.e1-bootstrap-review/v1",
                "verdict": "accepted",
                "e0_summary_sha256": sha256_file(summary),
                "baseline_sha256": baseline["sha256"],
                "package_manifest_sha256": sha256_file(package),
                "installed_candidate_digest": e1_bootstrap._digest(fingerprint),
                "model_identity_digest": e1_bootstrap._digest(
                    assignment["model_identity"]
                ),
                "assignment_sha256": sha256_file(assignment_path),
                "gateway_worker_job_id": "review-job",
                "verifier_job_id": "review-job-terra",
            },
        )
        e1_bootstrap.check_bootstrap_review(self.home, gateway_report)
        e1_bootstrap.import_bootstrap_review_report(self.home, status_reader=status)
        bootstrap = e1_bootstrap.create_bootstrap(self.home, review)
        for arm in ("baseline", "candidate"):
            for case_id in ("regression", "held-a", "held-b"):
                path = self.home / "e1" / "repair" / "trials" / arm / f"{case_id}.json"
                receipt = read_json(path)
                receipt.update(
                    profile_source="e0-bootstrap",
                    accepted_profile_id=None,
                    accepted_profile_path=None,
                    accepted_profile_sha256=None,
                    bootstrap_receipt_path=bootstrap["_source_path"],
                    bootstrap_receipt_sha256=bootstrap["_source_sha256"],
                )
                write_json(path, receipt)
        return bootstrap, summary, review, package

    def test_native_trials_compare_from_host_checks_but_cannot_promote(self):
        self._paired_trials()
        record = self.runner.compare_native("repair", "regression")
        self.assertEqual(record["state"], "compared")
        self.assertEqual(record["comparison"]["baseline"]["target_score"], 0)
        self.assertEqual(record["comparison"]["candidate"]["target_score"], 1)
        self.assertTrue(record["comparison"]["candidate"]["e1_native_trial"])
        self.assertEqual(self.runner.store.active("main")["configuration"], record["baseline"])
        review = self.home / "verifications" / "experiments" / "repair.json"
        write_json(review, {"schema_version": "mavis.experiment-review/v1"})
        with self.assertRaises(FileNotFoundError):
            self.runner.store.review("repair", review)
        self.assertEqual(self.runner.store.load("repair")["state"], "compared")

    def test_native_trial_missing_or_swapped_receipt_fails_closed(self):
        self._paired_trials()
        path = self.home / "e1" / "repair" / "trials" / "candidate" / "held-a.json"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.runner.compare_native("repair", "regression")
        self._trial_fixture("candidate", "held-a")
        receipt = read_json(path)
        receipt["case_id"] = "held-b"
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "frozen command or checkout"):
            self.runner.compare_native("repair", "regression")
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")

    def test_native_trial_prompt_command_and_raw_tampering_fail_closed(self):
        self._paired_trials()
        for field, value, expected in (
            ("selected_model", "other-model", "frozen command or checkout"),
            ("core_argv", ["wrong"], "frozen command or checkout"),
            ("observation_status", "inconclusive", "frozen command or checkout"),
            ("profile_source", "unknown", "unsupported profile source"),
            ("bootstrap_receipt_path", "/wrong", "accepted profile has a bootstrap binding"),
        ):
            with self.subTest(field=field):
                path = self.home / "e1" / "repair" / "trials" / "candidate" / "held-a.json"
                original = read_json(path)
                receipt = dict(original)
                receipt[field] = value
                write_json(path, receipt)
                with self.assertRaisesRegex(ValueError, expected):
                    self.runner.compare_native("repair", "regression")
                write_json(path, original)
        runtime = self.home / "e1" / "repair" / "runtime" / "candidate" / "held-a"
        instructions = runtime / "accepted-model-instructions.md"
        original_instructions = instructions.read_text()
        instructions.write_text("wrong prompt")
        with self.assertRaisesRegex(ValueError, "instructions hash changed"):
            self.runner.compare_native("repair", "regression")
        instructions.write_text(original_instructions)
        (runtime / "stdout.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "stdout hash changed"):
            self.runner.compare_native("repair", "regression")
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")

    def test_native_trial_bundle_rechecks_transcript_after_compare(self):
        self._paired_trials()
        self.runner.compare_native("repair", "regression")
        transcript = self.home / "e1" / "repair" / "runtime" / "candidate" / "held-a" / "rollout.jsonl"
        transcript.write_text("changed")
        with self.assertRaisesRegex(ValueError, "transcript identity changed"):
            self.runner.store._check_comparison(self.runner.store.load("repair"))

    def test_native_trial_rejects_changed_active_profile_pointer(self):
        self._paired_trials()
        pointer = self.home / "profiles" / "main" / "active.json"
        write_json(pointer, {"version": 2, "path": str(pointer.parent / "v2.json")})
        with self.assertRaisesRegex(ValueError, "accepted main profile pointer changed"):
            self.runner.compare_native("repair", "regression")
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")

    def test_native_bootstrap_pair_compares_without_active_profile(self):
        bootstrap, _, _, _ = self._paired_bootstrap_trials()
        self.assertFalse((self.home / "profiles/main/active.json").exists())
        with (patch.object(e1_bootstrap, "inventory", wraps=e1_bootstrap.inventory) as inventory_reader,
              patch.object(e1_bootstrap, "validate_bootstrap", wraps=e1_bootstrap.validate_bootstrap) as validator):
            record = self.runner.compare_native("repair", "regression")
            inventory_reader.assert_called_once_with("http://127.0.0.1:8001/v1")
            validator.assert_called_once_with(self.home)
        self.assertEqual(record["state"], "compared")
        self.assertEqual(record["comparison"]["candidate"]["target_score"], 1)
        self.assertEqual(record["comparison"]["baseline"]["target_score"], 0)
        self.assertEqual(bootstrap["model_identity"]["model_id"], "exact-model")
        review = self.home / "verifications/experiments/repair.json"
        write_json(review, {"schema_version": "mavis.experiment-review/v1"})
        def fail_on_alarm(_signum, _frame):
            raise TimeoutError("nested bootstrap lock")

        previous = signal.signal(signal.SIGALRM, fail_on_alarm)
        signal.setitimer(signal.ITIMER_REAL, 5)
        try:
            with self.runner.store._locked():
                self.runner.store._check_comparison(self.runner.store._load("repair"))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        with self.assertRaises(FileNotFoundError):
            self.runner.store.review("repair", review)

    def _native_review(self, *, bootstrap=False):
        if bootstrap:
            self._paired_bootstrap_trials()
        else:
            self._paired_trials()
        record = self.runner.compare_native("repair", "regression")
        owner = {"provider": "zai", "model": "review-model", "harness": "zcode"}
        prepared = self.runner.prepare_review("repair", owner)
        assignment_path = Path(prepared["assignment_path"])
        assignment = read_json(assignment_path)
        job_dir = self.root / "review-gateway-job"
        job_dir.mkdir()
        gateway_assignment = job_dir / "assignment.json"
        report = job_dir / "report.md"
        def start(arguments):
            write_json(gateway_assignment, {key: value for key, value in arguments.items() if key != "job_id"})
            return {"success": True, "started": True, "accepted": False,
                    "job_id": arguments["job_id"], "job_dir": str(job_dir),
                    "report_path": str(report), "assignment_path": str(gateway_assignment)}
        self.runner.gateway_assignment_starter = start
        dispatch = self.runner.dispatch_review("repair", "independent-review-worker")
        self.assertEqual(dispatch["gateway_assignment_sha256"], sha256_file(gateway_assignment))
        review = Path(prepared["review_receipt_path"])
        write_json(review, {
            "schema_version": "mavis.e1-review/v1", "experiment_id": "repair",
            "verdict": "accepted", "assignment_sha256": sha256_file(assignment_path),
            "facts_digest": e1_bootstrap._digest(assignment["facts"]),
            "gateway_worker_job_id": "independent-review-worker",
            "case_findings": {case_id: {
                "finding": f"Reviewed both raw trial and check receipts for {case_id}",
                "evidence": evidence,
            } for case_id, evidence in assignment["facts"]["case_evidence"].items()},
        })
        report.write_bytes(review.read_bytes())
        def status(job_id):
            binding = {
                "schema_version": "model-gateway-mavis-objective-binding/v1",
                "objective_id": "repair", "cwd": assignment["cwd"],
                "owner": owner, "starting_revision": assignment["starting_revision"],
                "changed_revision": assignment["starting_revision"],
                "owned_paths": assignment["owned_paths"],
                "requirements": assignment["requirements"],
                "required_checks": assignment["required_checks"],
                "assignment_sha256": sha256_file(gateway_assignment),
                "report_sha256": sha256_file(report),
                "target_sha256": "a" * 64,
            }
            return {
                "job_id": job_id, "state": "completed", "exit_code": 0, "accepted": True,
                "receipt": {"job_id": job_id, "exit_code": 0},
                "acceptance": {"accepted": True, "job_id": job_id, "verifier": "terra",
                               "verifier_job_id": "independent-terra-verifier",
                               "target_sha256": "a" * 64,
                               "evidence_sha256": "b" * 64,
                               "report_sha256_on_disk": sha256_file(report),
                               "verifier_verdict_sha256": "d" * 64},
                "mavis_binding": binding,
            }
        self.runner.store.gateway_status_reader = status
        return record, assignment_path, review, status

    def test_native_review_stage_promote_and_rollback(self):
        record, assignment_path, review, _ = self._native_review()
        original = self.runner.store.active("main")
        self.assertEqual(self.runner.store.review("repair", review)["state"], "reviewed")
        self.assertEqual(self.runner.store.stage("repair")["state"], "staged")
        self.assertEqual(self.runner.store.active("main"), original)
        with self.assertRaisesRegex(ValueError, "between objectives"):
            self.runner.store.promote("repair", between_objectives=False)
        self.assertEqual(self.runner.store.promote("repair", between_objectives=True)["state"], "promoted")
        self.assertEqual(self.runner.store.active("main")["configuration"], record["candidate"])
        self.assertEqual(self.runner.store.rollback("repair", reason="observed regression")["state"], "rolled-back")
        self.assertEqual(self.runner.store.active("main"), original)

    def test_native_review_imports_exact_gateway_report(self):
        _, _, review, _ = self._native_review()
        review.unlink()
        self.assertEqual(self.runner.import_review("repair")["state"], "reviewed")
        self.assertTrue(review.is_file())
        self.assertEqual(self.runner.store.stage("repair")["state"], "staged")

    def test_native_reviewer_cannot_match_recorded_candidate_worker(self):
        self._paired_trials()
        record = self.runner.compare_native("repair", "regression")
        root = self.home / "e1" / "repair"
        owner = {"provider": "zai", "model": "review-model", "harness": "zcode"}
        checkout = root / "checkouts" / "candidate" / "regression"
        assignment = self.root / "candidate-assignment.json"
        write_json(assignment, {
            "mavis_owner": owner, "mavis_objective_id": "repair",
            "mavis_requirements": [f"experiment-candidate-snapshot:{record['candidate']['sha256']}"],
            "cwd": str(checkout),
        })
        write_json(root / "dispatch" / "regression.json", {
            "schema_version": "mavis.e1-native-dispatch/v1",
            "experiment_id": "repair", "case_id": "regression", "job_id": "candidate-worker",
            "candidate_sha256": record["candidate"]["sha256"],
            "assignment_path": str(assignment), "assignment_sha256": sha256_file(assignment),
            "checkout": str(checkout),
        })
        with self.assertRaisesRegex(ValueError, "independent from candidate workers"):
            self.runner.prepare_review("repair", owner)

    def test_native_bootstrap_review_revalidates_model_artifacts(self):
        _, _, review, _ = self._native_review(bootstrap=True)
        self.runner.store.review("repair", review)
        self.assertEqual(self.runner.store.stage("repair")["state"], "staged")
        model = self.root / "models/exact-model/model-00001.safetensors"
        model.write_bytes(b"different-weight-bytes")
        with self.assertRaisesRegex(ValueError, "model artifact bytes changed"):
            self.runner.store.promote("repair", between_objectives=True)

    def test_native_review_rejects_wrong_facts_and_gateway_assignment(self):
        _, assignment_path, review, status = self._native_review()
        original = read_json(review)
        for field, value in (("facts_digest", "0" * 64),
                             ("assignment_sha256", "1" * 64),
                             ("gateway_worker_job_id", "e1trial-repair")):
            with self.subTest(field=field):
                modified = dict(original)
                modified[field] = value
                write_json(review, modified)
                with self.assertRaises(ValueError):
                    self.runner.store.review("repair", review)
        write_json(review, original)
        missing_findings = dict(original)
        missing_findings.pop("case_findings")
        write_json(review, missing_findings)
        report = self.root / "review-gateway-job" / "report.md"
        report.write_bytes(review.read_bytes())
        with self.assertRaisesRegex(ValueError, "case-specific findings"):
            self.runner.store.review("repair", review)
        write_json(review, original)
        report.write_bytes(review.read_bytes())
        def wrong_status(job_id):
            result = status(job_id)
            result["mavis_binding"]["owner"] = {"provider": "forged"}
            return result
        self.runner.store.gateway_status_reader = wrong_status
        with self.assertRaisesRegex(ValueError, "exact independent gateway acceptance"):
            self.runner.store.review("repair", review)
        self.runner.store.gateway_status_reader = status
        self.runner.store.review("repair", review)
        assignment = read_json(assignment_path)
        assignment["facts"]["held_out"] = ["regression"]
        write_json(assignment_path, assignment)
        with self.assertRaisesRegex(ValueError, "assignment differs"):
            self.runner.store.stage("repair")

    def test_native_review_rechecks_raw_trial_and_status_revocation(self):
        _, _, review, status = self._native_review()
        self.runner.store.review("repair", review)
        runtime = self.home / "e1/repair/runtime/candidate/held-a"
        (runtime / "stdout.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "stdout hash changed"):
            self.runner.store.stage("repair")
        (runtime / "stdout.log").write_text("ran\n")
        def revoked(job_id):
            result = status(job_id)
            result["accepted"] = False
            return result
        self.runner.store.gateway_status_reader = revoked
        with self.assertRaisesRegex(ValueError, "not accepted"):
            self.runner.store.stage("repair")

    def test_native_bootstrap_missing_changed_or_wrong_model_fails_closed(self):
        bootstrap, summary, _, _ = self._paired_bootstrap_trials()
        path = Path(bootstrap["_source_path"])
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.runner.compare_native("repair", "regression")
        path.write_bytes(original)
        summary.write_bytes(summary.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.runner.compare_native("repair", "regression")
        summary.write_bytes(summary.read_bytes()[:-1])
        wrong = read_json(path)
        wrong["model_identity"]["model_id"] = "wrong-model"
        write_json(path, wrong)
        for arm in ("baseline", "candidate"):
            for case_id in ("regression", "held-a", "held-b"):
                trial_path = self.home / "e1" / "repair" / "trials" / arm / f"{case_id}.json"
                trial = read_json(trial_path)
                trial["bootstrap_receipt_sha256"] = sha256_file(path)
                write_json(trial_path, trial)
        with self.assertRaisesRegex(ValueError, "bootstrap receipt is invalid or stale"):
            self.runner.compare_native("repair", "regression")
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")

    def test_native_bootstrap_requires_same_binding_in_both_arms(self):
        self._paired_bootstrap_trials()
        path = self.home / "e1/repair/trials/candidate/held-b.json"
        trial = read_json(path)
        trial["bootstrap_receipt_sha256"] = "0" * 64
        write_json(path, trial)
        with self.assertRaisesRegex(ValueError, "bootstrap receipt changed"):
            self.runner.compare_native("repair", "regression")

    def test_native_bootstrap_review_change_invalidates_recorded_comparison(self):
        _, _, review, _ = self._paired_bootstrap_trials()
        self.runner.compare_native("repair", "regression")
        review.write_bytes(review.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.runner.store._check_comparison(self.runner.store.load("repair"))

    def test_bootstrap_gateway_status_rejects_forged_or_revoked_acceptance(self):
        self._paired_bootstrap_trials()
        original = e1_bootstrap.harness_job_status("review-job")
        for name, mutate in (
            ("revoked", lambda value: value.update(accepted=False)),
            (
                "wrong target",
                lambda value: value["acceptance"].update(target_sha256="0" * 64),
            ),
            (
                "wrong gateway assignment",
                lambda value: value["mavis_binding"].update(assignment_sha256="0" * 64),
            ),
            (
                "wrong owner",
                lambda value: value["mavis_binding"].update(
                    owner={"provider": "forged"}
                ),
            ),
            (
                "stale revision",
                lambda value: value["mavis_binding"].update(changed_revision="0" * 40),
            ),
        ):
            with self.subTest(name=name):
                forged = copy.deepcopy(original)
                mutate(forged)
                with patch.object(
                    e1_bootstrap, "harness_job_status", return_value=forged
                ):
                    with self.assertRaises(ValueError):
                        e1_bootstrap.validate_bootstrap(self.home)

    def test_bootstrap_gateway_completion_and_terra_contract(self):
        """Exercise the canonical gateway's real receipt and acceptance rules."""
        self._paired_bootstrap_trials()
        source = (Path(__file__).resolve().parents[4] / "ai-skills-dev-mavis-gateway"
                  / "marketplace/plugins/model-gateway/lib/harness_runner.py")
        if not source.is_file():
            self.skipTest("canonical model gateway source is not checked out")
        spec = importlib.util.spec_from_file_location("mavis_test_gateway_runner", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = module.HarnessJobRunner(self.root / "gateway-jobs", tmux_binary="/bin/false")
        worker = self.root / "bootstrap-gateway-job"
        verifier_job = "review-job-terra"
        verifier_dir = self.root / "bootstrap-terra-job"
        verifier_dir.mkdir()
        runner._job_dir = lambda job_id: worker if job_id == "review-job" else verifier_dir
        self.assertEqual(module.validate_assignment(read_json(worker / "assignment.json")), [])
        command = module.build_native_command(
            "minimax", prompt="Review frozen E1 evidence", model="minimax/MiniMax-M3")
        self.assertEqual(command[command.index("-m") + 1], "minimax/MiniMax-M3")
        self.assertNotIn("-m", module.build_native_command(
            "zcode", prompt="Review frozen E1 evidence", model="zcode-reviewer"))
        packet = read_json(self.home / "e1/bootstrap/review-assignment.json")
        with self.assertRaisesRegex(ValueError, "model-selected MiniMax"):
            e1_bootstrap._gateway_arguments(
                {**packet, "owner": {"provider": "zai", "model": "zcode-reviewer", "harness": "zcode"}},
                self.home / "e1/bootstrap/review-assignment.json", "review-job")

        write_json(worker / "meta.json", {
            "job_id": "review-job", "lane": "minimax", "command": command})
        status = runner.status
        complete = lambda job_id, checks: {
            "success": True, **runner.record_completion(job_id, check_results=checks)}
        with self.assertRaisesRegex(ValueError, "terminal receipt"):
            e1_bootstrap_gateway.complete_bootstrap_review(
                self.home, status_reader=status, completer=complete)
        write_json(worker / "receipt.json", {"job_id": "review-job", "exit_code": 0})
        completion = e1_bootstrap_gateway.complete_bootstrap_review(
            self.home, status_reader=status, completer=complete)
        self.assertEqual(completion["gateway_completion"]["check_results"][
            "independent-bootstrap-review"]["exit_code"], 0)
        self.assertIsNotNone(runner._target_binding("review-job"))
        self.assertFalse(status("review-job")["accepted"])

        def start_terra(job_id, verifier_job_id):
            binding = runner._target_binding(job_id)
            self.assertEqual(verifier_job_id, verifier_job)
            write_json(verifier_dir / "assignment.json", {
                "lane": "terra", "verification_target": binding})
            write_json(verifier_dir / "meta.json", {
                "job_id": verifier_job_id, "lane": "terra", "command": ["codex", "exec"]})
            return {"success": True, "started": True, "accepted": False,
                    "job_id": verifier_job_id}

        e1_bootstrap_gateway.start_bootstrap_verifier(self.home, starter=start_terra)
        write_json(verifier_dir / "receipt.json", {
            "job_id": verifier_job, "exit_code": 0})
        write_json(verifier_dir / "report.md", {
            "target_sha256": "0" * 64, "verdict": "accepted", "findings": "wrong target"})
        verify = lambda job_id, verifier_job_id, report_sha: {
            "success": True, **runner.record_verification(
                job_id, verifier="terra", verdict="accepted",
                evidence_sha256=report_sha, verifier_job_id=verifier_job_id)}
        with self.assertRaisesRegex(ValueError, "not accepted"):
            e1_bootstrap_gateway.verify_and_import_bootstrap_review(
                self.home, status_reader=status, verifier=verify)
        self.assertFalse(status("review-job")["accepted"])
        write_json(verifier_dir / "report.md", {
            "target_sha256": runner._target_binding("review-job")["target_sha256"],
            "verdict": "accepted", "findings": "Reviewed frozen E1 evidence."})
        result = e1_bootstrap_gateway.verify_and_import_bootstrap_review(
            self.home, status_reader=status, verifier=verify)
        self.assertEqual(result["review_sha256"], sha256_file(worker / "report.md"))
        self.assertTrue(status("review-job")["accepted"])
        output = worker / "independent-bootstrap-review.json"
        output.write_text("tampered\n")
        self.assertFalse(status("review-job")["accepted"])
        with patch.object(e1_bootstrap, "harness_job_status", status):
            with self.assertRaises(ValueError):
                e1_bootstrap.validate_bootstrap(self.home)


    def test_bootstrap_review_checkout_rejects_stale_installed_source(self):
        self._paired_bootstrap_trials()
        checkout = self.home / "e1/bootstrap/review-checkout"
        (checkout / "local-codex/mavis/mavis/__init__.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "checkout has changed files"):
            e1_bootstrap.validate_bootstrap(self.home)


    def test_bootstrap_prepare_reuses_only_exact_orphan_checkout(self):
        _, _, review, _ = self._paired_bootstrap_trials()
        packet = self.home / "e1/bootstrap/review-assignment.json"
        original = read_json(packet)
        for path in (
            packet,
            self.home / "e1/bootstrap/review-dispatch.json",
            self.home / "e1/bootstrap/main.json",
            review,
        ):
            path.unlink()
        owner = original["owner"]
        inventory_reader = lambda _: [
            {
                "id": "exact-model",
                "model_path": str((self.root / "models/exact-model").resolve()),
            }
        ]
        recovered = e1_bootstrap.prepare_bootstrap_review(
            self.home, owner, inventory_reader=inventory_reader
        )
        self.assertEqual(recovered["cwd"], original["cwd"])
        self.assertEqual(recovered["starting_revision"], original["starting_revision"])
        packet.unlink()
        (Path(original["cwd"]) / "stray.txt").write_text("unreviewed")
        with self.assertRaisesRegex(ValueError, "checkout has changed files"):
            e1_bootstrap.prepare_bootstrap_review(
                self.home, owner, inventory_reader=inventory_reader
            )


    def test_native_bootstrap_model_bytes_invalidate_recorded_comparison(self):
        self._paired_bootstrap_trials()
        self.runner.compare_native("repair", "regression")
        shard = self.root / "models/exact-model/model-00001.safetensors"
        shard.write_bytes(b"different-weights")
        with self.assertRaisesRegex(ValueError, "model artifact bytes changed"):
            self.runner.store._check_comparison(self.runner.store.load("repair"))

    def test_native_bootstrap_current_inventory_mapping_blocks_comparison(self):
        self._paired_bootstrap_trials()
        wrong = self.root / "wrong-model"
        wrong.mkdir()
        with patch.object(e1_bootstrap, "inventory", return_value=[
            {"id": "exact-model", "model_path": str(wrong.resolve())}
        ]) as inventory_reader:
            with self.assertRaisesRegex(ValueError, "current endpoint model_path differs"):
                self.runner.compare_native("repair", "regression")
            inventory_reader.assert_called_once_with("http://127.0.0.1:8001/v1")
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")
        self.assertFalse((self.home / "e1/repair/native-candidate-trial-summary.json").exists())

    def test_native_bootstrap_current_inventory_mapping_rechecked_after_compare(self):
        self._paired_bootstrap_trials()
        self.runner.compare_native("repair", "regression")
        wrong = self.root / "wrong-model"
        wrong.mkdir()
        with patch.object(e1_bootstrap, "inventory", return_value=[
            {"id": "exact-model", "model_path": str(wrong.resolve())}
        ]) as inventory_reader:
            with self.assertRaisesRegex(ValueError, "current endpoint model_path differs"):
                self.runner.store._check_comparison(self.runner.store.load("repair"))
            inventory_reader.assert_called_once_with("http://127.0.0.1:8001/v1")

    def test_native_dispatch_rejects_unbound_start_receipt(self):
        self._freeze()
        self.runner.prepare("repair", "candidate")
        self.runner.gateway_assignment_starter = lambda arguments: {
            "success": True, "started": True, "accepted": False,
            "job_id": "different-job", "job_dir": str(self.root / "job"),
            "report_path": str(self.root / "job" / "report.md"),
            "assignment_path": str(self.root / "job" / "assignment.json")}
        with self.assertRaisesRegex(ValueError, "exact Mavis candidate job"):
            self.runner.dispatch_candidate("repair", "regression", job_id="expected-job",
                                           lane="minimax", model="minimax/MiniMax-M3", task="Fix")

    def test_candidate_check_rejects_uncommitted_or_ignored_content(self):
        self._freeze()
        self.runner.prepare("repair", "candidate")
        checkout = self.home / "e1" / "repair" / "checkouts" / "candidate" / "regression"
        (checkout / "result.txt").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "checkout must be clean"):
            self.runner.check("repair", "candidate", "regression")
        self._pass_candidate("regression")
        (checkout / ".gitignore").write_text("cache.bin\n")
        subprocess.run(["git", "-C", str(checkout), "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test",
                        "-c", "user.email=test@example.invalid", "commit", "-qm", "ignore cache"], check=True)
        (checkout / "cache.bin").write_bytes(b"hidden input")
        with self.assertRaisesRegex(ValueError, "checkout must be clean"):
            self.runner.check("repair", "candidate", "regression")

    def test_comparison_rejects_checkout_change_after_host_check(self):
        self._freeze()
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            self._pass_candidate(case)
            self.runner.check("repair", "candidate", case)
        checkout = self.home / "e1" / "repair" / "checkouts" / "candidate" / "held-a"
        (checkout / "result.txt").write_text("different dirty content\n")
        report = self.root / "native-report.json"
        report.write_text('{"worker":"fixture"}')
        with self.assertRaisesRegex(ValueError, "checkout must be clean"):
            self.runner.compare("repair", "candidate-job", report)
        self.assertEqual(self.runner.store.load("repair")["state"], "candidate")


if __name__ == "__main__":
    unittest.main()
