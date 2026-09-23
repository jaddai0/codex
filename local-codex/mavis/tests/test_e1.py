from pathlib import Path
import hashlib
import json
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis import e1_bootstrap
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
        fingerprint = {"core_sha256": read_json(package)["core_sha256"], "service_sha256": "b" * 64}
        e0 = self.home / "evaluations/e0"
        case_hashes = {}
        for case in E0_CASES:
            path = e0 / f"{case}.json"
            write_json(path, {"schema_version": "mavis.evaluation-case/v1", "suite": "e0",
                              "case": case, "status": "pass", "evidence": ["host fixture"]})
            case_hashes[case] = sha256_file(path)
        summary = e0 / "summary.json"
        write_json(summary, {
            "schema_version": "mavis.evaluation-suite/v1", "suite": "e0", "status": "pass",
            "mandatory_cases": list(E0_CASES),
            "results": [{"case": case, "status": "pass"} for case in E0_CASES],
            "installed_candidate": fingerprint, "model_id": "exact-model",
            "case_receipts": case_hashes,
        })
        model_root = self.root / "models"
        model_path = model_root / "exact-model"
        model_path.mkdir(parents=True)
        write_json(model_path / "config.json", {
            "architectures": ["QwenFixture"], "quantization": {"bits": 4},
        })
        write_json(model_path / "tokenizer.json", {"version": "fixture"})
        write_json(model_path / "tokenizer_config.json", {"bos_token": "<s>"})
        (model_path / "chat_template.jinja").write_text("{{ prompt }}")
        (model_path / "model-00001.safetensors").write_bytes(b"weights-one")
        owner = {"provider": "zcode", "model": "zcode-reviewer", "harness": "zcode"}
        assignment_path = self.home / "e1/bootstrap/review-assignment.json"
        review = self.home / "verifications/e1-bootstrap/main.json"

        def status(job_id):
            return {
                "job_id": job_id, "state": "completed", "exit_code": 0, "accepted": True,
                "receipt": {"job_id": job_id, "exit_code": 0},
                "acceptance": {"accepted": True, "job_id": job_id, "verifier": "terra",
                               "verifier_job_id": "terra-job",
                               "target_sha256": sha256_file(assignment_path),
                               "evidence_sha256": "b" * 64,
                               "report_sha256_on_disk": sha256_file(review),
                               "verifier_verdict_sha256": "d" * 64},
                "mavis_binding": {
                    "schema_version": "mavis.e1-bootstrap-gateway-binding/v1",
                    "objective_id": "e1-bootstrap-main", "scope": "main",
                    "cwd": assignment["cwd"], "owner": owner,
                    "starting_revision": assignment["starting_revision"],
                    "changed_revision": assignment["starting_revision"],
                    "owned_paths": assignment["owned_paths"],
                    "requirements": assignment["requirements"],
                    "required_checks": assignment["required_checks"],
                    "assignment_sha256": sha256_file(assignment_path),
                    "report_sha256": sha256_file(review),
                    "target_sha256": sha256_file(assignment_path),
                },
            }
        for name, value in (
            ("installed_candidate_fingerprint", lambda: fingerprint),
            ("_package_manifest_path", lambda: package),
            ("_model_root_path", lambda: model_root),
            ("harness_job_status", status),
        ):
            active_patch = patch.object(e1_bootstrap, name, value)
            active_patch.start()
            self.addCleanup(active_patch.stop)
        assignment = e1_bootstrap.prepare_bootstrap_review(
            self.home, owner,
            inventory_reader=lambda _: [{"id": "exact-model", "model_path": str(model_path.resolve())}],
        )
        baseline = self.runner.store.active("main")["configuration"]
        write_json(review, {
            "schema_version": "mavis.e1-bootstrap-review/v1", "verdict": "accepted",
            "e0_summary_sha256": sha256_file(summary), "baseline_sha256": baseline["sha256"],
            "package_manifest_sha256": sha256_file(package),
            "installed_candidate_digest": e1_bootstrap._digest(fingerprint),
            "model_identity_digest": e1_bootstrap._digest(assignment["model_identity"]),
            "assignment_sha256": sha256_file(assignment_path),
            "gateway_worker_job_id": "review-job", "verifier_job_id": "terra-job",
        })
        bootstrap = e1_bootstrap.create_bootstrap(self.home, review)
        for arm in ("baseline", "candidate"):
            for case_id in ("regression", "held-a", "held-b"):
                path = self.home / "e1" / "repair" / "trials" / arm / f"{case_id}.json"
                receipt = read_json(path)
                receipt.update(profile_source="e0-bootstrap", accepted_profile_id=None,
                               accepted_profile_path=None, accepted_profile_sha256=None,
                               bootstrap_receipt_path=bootstrap["_source_path"],
                               bootstrap_receipt_sha256=bootstrap["_source_sha256"])
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
        with self.assertRaisesRegex(ValueError, "separate installed-trial review adapter"):
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
        record = self.runner.compare_native("repair", "regression")
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
        with self.assertRaisesRegex(ValueError, "separate installed-trial review adapter"):
            self.runner.store.review("repair", review)

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

    def test_native_bootstrap_model_bytes_invalidate_recorded_comparison(self):
        self._paired_bootstrap_trials()
        self.runner.compare_native("repair", "regression")
        shard = self.root / "models/exact-model/model-00001.safetensors"
        shard.write_bytes(b"different-weights")
        with self.assertRaisesRegex(ValueError, "model artifact bytes changed"):
            self.runner.store._check_comparison(self.runner.store.load("repair"))

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
