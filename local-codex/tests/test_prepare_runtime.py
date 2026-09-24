import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import hashlib
import sys
import subprocess
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


MODULE_PATH = Path(__file__).parents[1] / "prepare_runtime.py"
sys.path.insert(0, str(MODULE_PATH.parent / "mavis"))
SPEC = importlib.util.spec_from_file_location("prepare_runtime", MODULE_PATH)
prepare_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_runtime)


class PrepareRuntimeTests(unittest.TestCase):
    def test_launcher_passes_active_model_to_runtime_and_spawns_core(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mavis_home = root / "service"
            self._accepted_profile(mavis_home, 1, "model-b", "Accepted prompt")
            share = root / "share"
            share.mkdir()
            shutil.copy(MODULE_PATH, share / "prepare_runtime.py")
            shutil.copy(MODULE_PATH.parent / "launch_core.py", share / "launch_core.py")
            shutil.copy(MODULE_PATH.parent / "generation_lease.py", share / "generation_lease.py")
            (share / "persona.toml").write_text('name = "Mavis"\n')
            (share / "base-instructions.md").write_text("Base instructions\n")
            stub = share / "mavis"
            shutil.copytree(MODULE_PATH.parent / "mavis" / "mavis", stub)
            (stub / "__main__.py").write_text(
                "import json, os, sys\n"
                "open(os.environ['STUB_RUNTIME_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n"
            )
            workspace = root / "workspace"
            (workspace / ".codex").mkdir(parents=True)
            (workspace / ".codex" / "wrong.md").write_text("Wrong project prompt\n")
            (workspace / ".codex" / "config.toml").write_text(
                'model = "wrong-project-model"\n'
                'model_provider = "wrong-provider"\n'
                'model_catalog_json = "wrong-catalog.json"\n'
                'model_instructions_file = "wrong.md"\n'
            )
            core = root / "core"
            core.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys, tomllib\n"
                "args = sys.argv[1:]\n"
                "work = pathlib.Path(args[args.index('-C') + 1])\n"
                "with (work / '.codex/config.toml').open('rb') as handle: cfg = tomllib.load(handle)\n"
                "for index, argument in enumerate(args):\n"
                "    if argument == '-c':\n"
                "        key, value = args[index + 1].split('=', 1)\n"
                "        cfg[key] = tomllib.loads('value = ' + value)['value']\n"
                "prompt = pathlib.Path(cfg['model_instructions_file']).read_text()\n"
                "pathlib.Path(os.environ['STUB_CORE_OBSERVED']).write_text(json.dumps(\n"
                "    {'config': cfg, 'prompt': prompt, 'args': args}))\n"
            )
            core.chmod(0o755)
            gateway = root / "gateway" / "bin" / "mcp-server.sh"
            gateway.parent.mkdir(parents=True)
            gateway.write_text("#!/bin/sh\nexit 0\n")
            gateway.chmod(0o755)

            class Inventory(BaseHTTPRequestHandler):
                def do_GET(self):
                    payload = json.dumps({"models": [{"id": "model-b", "model_type": "llm", "loaded": True}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *_args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Inventory)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                env = os.environ.copy()
                env.update({"HOME": str(root), "LOCAL_CODEX_SHARE_DIR": str(share),
                            "LOCAL_CODEX_HOME": str(root / "codex-home"), "MAVIS_HOME": str(mavis_home),
                            "MAVIS_MEMORY_MCP_ENABLED": "0",
                            "LOCAL_CODEX_BIN": str(core), "MAVIS_GATEWAY_ROOT": str(gateway.parent.parent),
                            "OMLX_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                            "STUB_RUNTIME_ARGS": str(root / "runtime-args.json"),
                            "STUB_CORE_OBSERVED": str(root / "core-observed.json")})
                result = subprocess.run(["zsh", str(MODULE_PATH.parent / "bin/local-codex"),
                                         "-C", str(workspace)],
                                        env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads((root / "runtime-args.json").read_text())
                self.assertEqual(args[args.index("--model") + 1], "model-b")
                receipt = json.loads(next((mavis_home / "launches").glob("*.json")).read_text())
                self.assertEqual(receipt["state"], "exited")
                self.assertEqual(receipt["selected_model"], "model-b")
                self.assertEqual(receipt["core_exit_code"], 0)
                observed = json.loads((root / "core-observed.json").read_text())
                self.assertEqual(observed["config"]["model"], "model-b")
                self.assertEqual(observed["config"]["model_provider"], "omlx")
                self.assertEqual(observed["config"]["model_catalog_json"], receipt["catalog_path"])
                self.assertEqual(observed["config"]["model_instructions_file"], receipt["instructions_path"])
                self.assertEqual(hashlib.sha256(observed["prompt"].encode()).hexdigest(),
                                 receipt["effective_system_prompt_sha256"])
                self.assertEqual(observed["args"][:8],
                                 [part for override in receipt["binding_overrides"] for part in ("-c", override)])
                (root / "runtime-args.json").unlink()
                blocked = subprocess.run(["zsh", str(MODULE_PATH.parent / "bin/local-codex"),
                                          "-c", 'model_provider="other"'],
                                         env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(blocked.returncode, 2)
                self.assertFalse((root / "runtime-args.json").exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def _accepted_profile(self, mavis_home, version, model_id, prompt, previous_version=None):
        from mavis.experiments import ExperimentStore
        from mavis.storage import read_json, write_json, sha256_file

        role = mavis_home / "profiles" / "main"
        role.mkdir(parents=True, exist_ok=True)
        experiment_store = ExperimentStore(mavis_home)
        active_path = experiment_store._active_path("main")
        previous = read_json(active_path) if active_path.exists() else None
        configuration = {"prompts": {"system": prompt}, "tool_settings": {}, "retrieval": {}}
        candidate = experiment_store._snapshot(configuration)
        experiment_id = f"exp-v{version}"
        verifier_path = mavis_home / "verifications" / "experiments" / f"{experiment_id}.json"
        write_json(verifier_path, {"verdict": "accepted", "experiment_id": experiment_id})
        experiment_path = experiment_store._record_path(experiment_id)
        write_json(experiment_path, {"schema_version": "mavis.experiment-lifecycle/v1",
                                     "experiment_id": experiment_id, "scope": "main", "state": "promoted",
                                     "candidate": candidate,
                                     "baseline": previous["configuration"] if previous else candidate,
                                     "history": [],
                                     "review": {"path": str(verifier_path.resolve()),
                                                "sha256": sha256_file(verifier_path), "verdict": "accepted"}})
        write_json(active_path, {"schema_version": "mavis.experiment-active/v1", "scope": "main",
                                 "configuration": candidate, "experiment_id": experiment_id,
                                 "previous": previous})
        retained = {"accepted_experiment": {"path": str(experiment_path.resolve()),
                                             "sha256": sha256_file(experiment_path)},
                    "verifier_receipt": {"path": str(verifier_path.resolve()),
                                         "sha256": sha256_file(verifier_path)}}
        profile_path = role / f"v{version}.json"
        profile_path.write_text(json.dumps({
            "schema_version": "mavis.model-profile/v1", "profile_id": f"profile-{version}",
            "role": "main", "version": version, "previous_version": previous_version,
            "status": "active", "model_identity": {
                "model_id": model_id, "architecture": "qwen", "weights_fingerprint": "weights",
                "tokenizer_fingerprint": "tokenizer", "chat_template_fingerprint": "template",
                "quantization": "4bit"},
            "runtime": {"name": "omlx", "version": "1"}, "prompts": {"system": prompt},
            "tool_settings": {}, "context_policy": {"retrieval": {}},
            "experiments": [experiment_id], **retained,
        }))
        (role / "active.json").write_text(json.dumps({"version": version, "path": str(profile_path.resolve())}))
        return profile_path

    def test_accepted_profile_config_receipt_and_rollback_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mavis_home, home = root / "service", root / "codex"
            persona = root / "persona.toml"
            persona.write_text('name = "Mavis"\n')
            instructions = root / "base.md"
            instructions.write_text("Base instructions\n")
            records = [{"id": model, "model_type": "llm", "loaded": True}
                       for model in ("model-a", "model-b")]
            for version, model, prompt, previous in ((1, "model-a", "First", None),
                                                    (2, "model-b", "Second", 1),
                                                    (1, "model-a", "First", None)):
                self._accepted_profile(mavis_home, version, model, prompt, previous)
                argv = ["prepare_runtime.py", "--home", str(home), "--mavis-home", str(mavis_home),
                        "--base-url", "http://127.0.0.1:8001/v1", "--persona-template", str(persona),
                        "--instructions-template", str(instructions)]
                with patch.object(sys, "argv", argv), patch.object(prepare_runtime, "read_json", return_value=records):
                    self.assertEqual(prepare_runtime.main(), 0)
                parsed = tomllib.loads((home / "config.toml").read_text())
                catalog = json.loads((home / "omlx-models.json").read_text())
                receipt = json.loads(sorted((mavis_home / "launches").glob("*.json"),
                                            key=lambda path: path.stat().st_mtime_ns)[-1].read_text())
                self.assertEqual(parsed["model"], model)
                self.assertEqual(catalog["models"][0]["base_instructions"], f"Base instructions\n\n{prompt}\n")
                self.assertEqual(receipt["selected_model"], model)
                self.assertEqual(receipt["profile_version"], version)
                self.assertEqual(receipt["config_sha256"], hashlib.sha256((home / "config.toml").read_bytes()).hexdigest())
                receipt_path = next(path for path in (mavis_home / "launches").glob("*.json")
                                    if json.loads(path.read_text())["profile_sha256"] ==
                                    hashlib.sha256((mavis_home / "profiles/main" / f"v{version}.json").read_bytes()).hexdigest()
                                    and json.loads(path.read_text())["state"] == "prepared")
                stub_core = root / "stub-core"
                stub_core.write_text("#!/bin/sh\nexit 0\n")
                stub_core.chmod(0o755)
                if version == 1:
                    accepted_prompt = Path(receipt["instructions_path"])
                    original_prompt = accepted_prompt.read_text()
                    accepted_prompt.write_text("changed after preparation\n")
                    rejected = subprocess.run([sys.executable, str(MODULE_PATH.parent / "launch_core.py"),
                                               "--receipt", str(receipt_path), "--", str(stub_core)],
                                              capture_output=True, text=True,
                                              env={**os.environ, "PYTHONPATH": str(MODULE_PATH.parent / "mavis")})
                    self.assertEqual(rejected.returncode, 2)
                    self.assertEqual(json.loads(receipt_path.read_text())["state"], "prepared")
                    accepted_prompt.write_text(original_prompt)
                run = subprocess.run([sys.executable, str(MODULE_PATH.parent / "launch_core.py"),
                                      "--receipt", str(receipt_path), "--", str(stub_core)],
                                     capture_output=True, text=True,
                                     env={**os.environ, "PYTHONPATH": str(MODULE_PATH.parent / "mavis")})
                self.assertEqual(run.returncode, 0, run.stderr)
                launched = json.loads(receipt_path.read_text())
                self.assertEqual(launched["state"], "exited")
                self.assertEqual(launched["core_exit_code"], 0)
                self.assertGreater(launched["core_pid"], 0)

    def test_rolled_back_experiment_cannot_launch_stale_profile(self):
        from mavis.experiments import ExperimentStore

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._accepted_profile(home, 1, "model-a", "A")
            self._accepted_profile(home, 2, "model-b", "B", 1)
            codex_home = home / "codex"
            persona = home / "persona.toml"
            instructions = home / "instructions.md"
            persona.write_text('name = "Mavis"\n')
            instructions.write_text("Base\n")
            argv = ["prepare_runtime.py", "--home", str(codex_home), "--mavis-home", str(home),
                    "--base-url", "http://127.0.0.1:8001/v1", "--persona-template", str(persona),
                    "--instructions-template", str(instructions)]
            with patch.object(sys, "argv", argv), patch.object(prepare_runtime, "read_json", return_value=[
                {"id": "model-b", "model_type": "llm", "loaded": True}]):
                self.assertEqual(prepare_runtime.main(), 0)
            prepared_receipt = next((home / "launches").glob("*.json"))
            ExperimentStore(home).rollback("exp-v2", reason="regression")
            marker = home / "core-ran"
            core = home / "core"
            core.write_text(f"#!/bin/sh\ntouch {marker}\n")
            core.chmod(0o755)
            blocked = subprocess.run([sys.executable, str(MODULE_PATH.parent / "launch_core.py"),
                                      "--receipt", str(prepared_receipt), "--", str(core)],
                                     capture_output=True, text=True,
                                     env={**os.environ, "PYTHONPATH": str(MODULE_PATH.parent / "mavis")})
            self.assertEqual(blocked.returncode, 2)
            self.assertFalse(marker.exists())
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                prepare_runtime.accepted_main_profile(home)
            profile_path = home / "profiles" / "main" / "v2.json"
            stale = json.loads(profile_path.read_text())
            stale["accepted_experiment"]["sha256"] = hashlib.sha256(
                (home / "experiments" / "records" / "exp-v2.json").read_bytes()).hexdigest()
            profile_path.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "active promoted experiment"):
                prepare_runtime.accepted_main_profile(home)
            self._accepted_profile(home, 1, "model-a", "A", 2)
            self.assertEqual(prepare_runtime.accepted_main_profile(home)["model_identity"]["model_id"], "model-a")

    def test_profile_cli_overrides_are_rejected_before_core_spawn(self):
        path = MODULE_PATH.parent / "launch_core.py"
        for arguments in (("-c", 'model_provider="other"'),
                          ("--config=model_instructions_file=\"/tmp/x\"",),
                          ("--config", 'model_catalog_json="/tmp/other"'),
                          ("--profile", "other"), ("-pother",),
                          ("--model", "other"), ("-mother",),
                          ("--oss",), ("--local-provider", "ollama"),
                          ("--enable", "example")):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, str(path), "--check-args", "--", *arguments],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)

    def test_shipped_persona_is_mavis(self):
        persona = (MODULE_PATH.parent / "persona.toml").read_text(encoding="utf-8")
        self.assertIn('name = "Mavis"', persona)

    def test_installer_exposes_mavis_command(self):
        installer = (MODULE_PATH.parent / "install.sh").read_text(encoding="utf-8")
        self.assertIn('"$install_bin/mavis"', installer)
        self.assertIn("Desktop/Mavis.command", installer)
        self.assertIn('"$repo_root/local-codex/launch_core.py" "$install_share/launch_core.py"', installer)

    def test_launcher_uses_isolated_mavis_runtime(self):
        launcher = (MODULE_PATH.parent / "bin" / "local-codex").read_text(
            encoding="utf-8"
        )
        self.assertIn("http://127.0.0.1:8001/v1", launcher)
        self.assertIn('"$share_dir/launch_core.py" --managed', launcher)
        self.assertIn('"-m", "mavis", "runtime", "ensure"',
                      (MODULE_PATH.parent / "launch_core.py").read_text(encoding="utf-8"))
        self.assertIn("http://127.0.0.1:8000/v1", launcher)
        self.assertIn(".local-codex/mavis-service", launcher)
        self.assertNotIn("${HOME}/.mavis", launcher)

    def test_rejects_non_loopback_server(self):
        with self.assertRaisesRegex(ValueError, "localhost"):
            prepare_runtime.local_base_url("https://api.openai.com/v1")

    def test_selects_loaded_default_and_preserves_context(self):
        records = [
            {
                "id": "other",
                "model_type": "llm",
                "engine_type": "batched",
                "loaded": False,
                "is_default": False,
                "model_context_length": 32768,
            },
            {
                "id": "qwen-local",
                "display_name": "Qwen Local",
                "model_type": "vlm",
                "engine_type": "vlm",
                "loaded": True,
                "is_default": True,
                "model_context_length": 262144,
                "reasoning_effort_options": ["low", "high"],
                "reasoning_effort_default": "low",
            },
        ]
        catalog, selected = prepare_runtime.prepare_catalog(
            records, None, "native prompt"
        )
        self.assertEqual(selected, "qwen-local")
        model = catalog["models"][0]
        self.assertEqual(model["slug"], "qwen-local")
        self.assertEqual(model["context_window"], 262144)
        self.assertEqual(model["shell_type"], "unified_exec")
        self.assertEqual(model["apply_patch_tool_type"], "function")
        self.assertEqual(model["input_modalities"], ["text", "image"])
        self.assertEqual(model["base_instructions"], "native prompt")

    def test_persona_is_separate_and_name_is_editable(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            template = home / "template.toml"
            template.write_text(
                'name = "Nova"\ninstructions = "Stay concise."\n', encoding="utf-8"
            )
            prepare_runtime.ensure_persona(home, template)
            rendered = (home / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("working name is Nova", rendered)
            self.assertIn("separate from Iris", rendered)
            self.assertIn("Stay concise.", rendered)

    def test_catalog_serializes_as_json(self):
        catalog, _ = prepare_runtime.prepare_catalog(
            [
                {
                    "id": "local",
                    "model_type": "llm",
                    "engine_type": "batched",
                    "loaded": True,
                    "is_default": True,
                    "model_context_length": 65536,
                }
            ],
            None,
            "native prompt",
        )
        json.dumps(catalog)

    def test_profile_disables_remote_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            prepare_runtime.write_profile(home, "http://127.0.0.1:8000/v1", "local")
            profile = (home / "config.toml").read_text(encoding="utf-8")
            self.assertIn('metrics_exporter = "none"', profile)
            self.assertIn("plugins = false", profile)
            self.assertIn("remote_plugin = false", profile)
            self.assertIn("check_for_update_on_startup = false", profile)
            self.assertIn("requires_openai_auth = false", profile)
            self.assertIn('command = "python3 -m mavis pre-compact"', profile)
            parsed = tomllib.loads(profile)
            self.assertEqual(
                parsed["hooks"]["PreCompact"][0]["hooks"][0]["command"],
                "python3 -m mavis pre-compact",
            )
            key = f"{(home / 'config.toml').resolve()}:pre_compact:0:0"
            self.assertEqual(
                parsed["hooks"]["state"][key]["trusted_hash"],
                prepare_runtime.mavis_archive_hook_hash(),
            )
            self.assertEqual(
                parsed["hooks"]["SessionStart"][0]["hooks"][0]["command"],
                "python3 -m mavis compaction-handoff",
            )
            handoff_key = f"{(home / 'config.toml').resolve()}:session_start:0:0"
            self.assertEqual(
                parsed["hooks"]["state"][handoff_key]["trusted_hash"],
                prepare_runtime.mavis_command_hook_hash(
                    "session_start", prepare_runtime.MAVIS_HANDOFF_COMMAND,
                    prepare_runtime.MAVIS_HANDOFF_STATUS,
                ),
            )

    def test_profile_preserves_local_user_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.toml").write_text(
                '[mcp_servers.fixture]\ncommand = "fixture-server"\n',
                encoding="utf-8",
            )
            prepare_runtime.write_profile(home, "http://127.0.0.1:8000/v1", "local")
            prepare_runtime.write_profile(home, "http://127.0.0.1:9000/v1", "second")
            profile = (home / "config.toml").read_text(encoding="utf-8")
            self.assertEqual(profile.count("BEGIN LOCAL CODEX MANAGED CONFIG"), 1)
            self.assertIn('base_url = "http://127.0.0.1:9000/v1"', profile)
            self.assertIn('[mcp_servers.fixture]\ncommand = "fixture-server"', profile)

    def test_profile_registers_only_explicit_trusted_gateway(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = root / "gateway"
            script = gateway / "bin" / "mcp-server.sh"
            script.parent.mkdir(parents=True)
            script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            script.chmod(0o755)
            home = root / "home"
            prepare_runtime.write_profile(
                home, "http://127.0.0.1:8001/v1", "local", gateway
            )
            prepare_runtime.write_profile(
                home, "http://127.0.0.1:8001/v1", "local", gateway
            )
            profile = (home / "config.toml").read_text(encoding="utf-8")
            self.assertEqual(profile.count("[mcp_servers.model-gateway]"), 1)
            self.assertIn(f'command = "{script.resolve()}"', profile)
            self.assertNotIn("[mcp_servers.openrouter]", profile)

    def test_profile_registers_local_memory_bridge_with_disable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            service = root / "service"
            service.mkdir()
            project = root / "project"
            project.mkdir()
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            models = root / "models"
            models.mkdir()
            prepare_runtime.write_profile(
                home, "http://127.0.0.1:8001/v1", "local",
                mavis_home=service, project_root=project, model_dir=models,
            )
            parsed = tomllib.loads((home / "config.toml").read_text())
            memory = parsed["mcp_servers"]["mavis-memory"]
            self.assertEqual(memory["args"], ["-m", "mavis.mcp_server"])
            self.assertEqual(memory["command"], sys.executable)
            self.assertEqual(memory["env"]["MAVIS_HOME"], str(service.resolve()))
            self.assertEqual(memory["env"]["MAVIS_PROJECT_ROOT"], str(project.resolve()))
            self.assertEqual(memory["env"]["MAVIS_LIBRARIAN_MODEL_PATH"],
                             str((models / "Mavis-Qwen3.5-4B-HF-Eval").resolve()))
            self.assertTrue(Path(memory["env"]["PYTHONPATH"]).is_dir())
            self.assertNotIn("sandbox_workspace_write", parsed)
            with patch.dict(os.environ, {"MAVIS_MEMORY_MCP_ENABLED": "0"}):
                prepare_runtime.write_profile(
                    home, "http://127.0.0.1:8001/v1", "local",
                    mavis_home=service, project_root=project, model_dir=models,
                )
            disabled = tomllib.loads((home / "config.toml").read_text())
            self.assertNotIn("mavis-memory", disabled.get("mcp_servers", {}))
            (home / "config.toml").write_text(
                '[mcp_servers.mavis-memory]\ncommand = "unmanaged"\n')
            with patch.dict(os.environ, {"MAVIS_MEMORY_MCP_ENABLED": "0"}):
                with self.assertRaisesRegex(ValueError, "existing Mavis memory"):
                    prepare_runtime.write_profile(
                        home, "http://127.0.0.1:8001/v1", "local",
                        mavis_home=service, project_root=project,
                    )
            (home / "config.toml").write_text(
                '[mcp_servers."mavis-memory"]\ncommand = "unmanaged"\n')
            with patch.dict(os.environ, {"MAVIS_MEMORY_MCP_ENABLED": "0"}):
                with self.assertRaisesRegex(ValueError, "existing Mavis memory"):
                    prepare_runtime.write_profile(
                        home, "http://127.0.0.1:8001/v1", "local",
                        mavis_home=service, project_root=project,
                    )

    def test_profile_passes_gateway_environment_file_path_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "gateway" / "bin" / "mcp-server.sh"
            script.parent.mkdir(parents=True)
            script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            script.chmod(0o755)
            env_file = root / "secrets.env"
            env_file.write_text("MINIMAX_API_KEY=fixture-secret\n", encoding="utf-8")
            home = root / "home"
            prepare_runtime.write_profile(
                home,
                "http://127.0.0.1:8001/v1",
                "local",
                script.parent.parent,
                env_file,
            )
            profile = (home / "config.toml").read_text(encoding="utf-8")
            parsed = tomllib.loads(profile)
            self.assertEqual(
                parsed["mcp_servers"]["model-gateway"]["env"]["MODEL_GATEWAY_ENV_FILE"],
                str(env_file.resolve()),
            )
            self.assertNotIn("fixture-secret", profile)


if __name__ == "__main__":
    unittest.main()
