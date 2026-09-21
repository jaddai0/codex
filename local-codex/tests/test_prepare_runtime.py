import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "prepare_runtime.py"
SPEC = importlib.util.spec_from_file_location("prepare_runtime", MODULE_PATH)
prepare_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_runtime)


class PrepareRuntimeTests(unittest.TestCase):
    def test_shipped_persona_is_mavis(self):
        persona = (MODULE_PATH.parent / "persona.toml").read_text(encoding="utf-8")
        self.assertIn('name = "Mavis"', persona)

    def test_installer_exposes_mavis_command(self):
        installer = (MODULE_PATH.parent / "install.sh").read_text(encoding="utf-8")
        self.assertIn('"$install_bin/mavis"', installer)

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


if __name__ == "__main__":
    unittest.main()
