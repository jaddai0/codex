from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from mavis.cli import main
from mavis.embedding_index import EmbeddingIdentity
from mavis.embedding_provider import LocalEmbeddingProvider
from mavis.maintenance_runtime import host_admission
from mavis.runtime import RuntimeConfig, mavis_generation_lease


REVISION = "a" * 40
MODEL = "Qwen3-Embedding-0.6B"


def model_row(**changes):
    row = {"id": MODEL, "loaded": True, "model_type": "embedding",
           "engine_type": "embedding", "model_path": f"/cache/snapshots/{REVISION}"}
    row.update(changes)
    return row


class EmbeddingCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=self.root, check=True)
        (self.root / "source.py").write_text("needle exact fact\nsemantic passage\n")
        subprocess.run(["git", "add", "source.py"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.root, check=True)

    def command(self, *args):
        output = StringIO()
        with mock.patch("mavis.cli.home_from_env", return_value=self.root / "home"):
            with redirect_stdout(output):
                status = main(["project-index", "--project", str(self.root), *args])
        return status, json.loads(output.getvalue())

    def options(self):
        return ("--embedding-model", MODEL, "--embedding-revision", REVISION,
                "--embedding-dimensions", "2")

    def fake_provider(self, **overrides):
        calls = []

        def request(endpoint, path, **kwargs):
            calls.append((endpoint, path, kwargs))
            return {"model": MODEL, "data": [{"index": 0, "embedding": [0.6, 0.8]}]}

        provider = LocalEmbeddingProvider(
            EmbeddingIdentity(MODEL, REVISION, "mavis-code-passage-8192-v1", 2),
            self.root / "home", inventory_reader=lambda _: [model_row()], request=request,
            ownership_check=lambda _: True, compute_admission=lambda *a, **k: (True, "clear"),
            **overrides,
        )
        return provider, calls

    def test_public_route_refresh_search_and_exact_default(self):
        provider, calls = self.fake_provider()
        with mock.patch("mavis.cli._embedding_provider", return_value=provider):
            status, result = self.command("refresh-embeddings", *self.options())
            self.assertEqual(status, 0)
            self.assertEqual(result["identity"], provider.identity.as_dict())
            self.assertEqual(result["changed"], 1)
            _, index_status = self.command("status")
            self.assertEqual(index_status["embedding"]["identity"], provider.identity.as_dict())
            self.assertTrue(index_status["embedding"]["snapshot_current"])
            status, result = self.command("search", "concept", "--embeddings", *self.options())
            self.assertEqual(status, 0)
            self.assertIsNone(result["embedding_error"])
            self.assertTrue(any(hit["source"] == "embedding" for hit in result["hits"]))
        self.assertTrue(calls)
        self.assertTrue(all(call[1] == "/v1/embeddings" for call in calls))
        self.assertTrue(all(call[2]["headers"] == {"X-OMLX-Require-Loaded": "true"}
                            for call in calls))
        self.assertTrue(all(isinstance(call[2]["payload"]["input"], str) for call in calls))
        status, result = self.command("search", "needle exact fact")
        self.assertEqual(status, 0)
        self.assertEqual(result[0]["source"], "live-fallback")
        (self.root / "source.py").write_text("needle exact fact\nchanged passage\n")
        _, index_status = self.command("status")
        self.assertFalse(index_status["embedding"]["snapshot_current"])

    def test_public_search_reports_provider_failure_and_keeps_exact_hit(self):
        provider, calls = self.fake_provider()
        with mock.patch("mavis.cli._embedding_provider", return_value=provider):
            self.command("refresh-embeddings", *self.options())
        calls.clear()
        provider.compute_admission = lambda *a, **k: (False, "IRIS busy")
        with mock.patch("mavis.cli._embedding_provider", return_value=provider):
            status, result = self.command("search", "needle exact fact", "--embeddings", *self.options())
        self.assertEqual(status, 0)
        self.assertEqual(result["hits"][0]["source"], "live-fallback")
        self.assertIn("IRIS busy", result["embedding_error"]["message"])
        self.assertEqual(calls, [])

    def test_missing_identity_is_reported_without_provider_request(self):
        status, result = self.command("search", "needle exact fact", "--embeddings")
        self.assertEqual(status, 0)
        self.assertEqual(result["hits"][0]["source"], "live-fallback")
        self.assertIn("--embedding-model", result["embedding_error"]["message"])
        status, result = self.command("refresh-embeddings")
        self.assertEqual(status, 2)
        self.assertEqual(result["status"], "failed")

    def test_public_refresh_fails_closed_without_loaded_model(self):
        provider, calls = self.fake_provider()
        provider.inventory_reader = lambda _: [model_row(loaded=False)]
        with mock.patch("mavis.cli._embedding_provider", return_value=provider):
            status, result = self.command("refresh-embeddings", *self.options())
        self.assertEqual(status, 2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("not loaded", result["embedding_error"]["message"])
        self.assertEqual(calls, [])

    def test_provider_checks_identity_admission_and_response(self):
        provider, calls = self.fake_provider()
        provider.inventory_reader = lambda _: [model_row(model_path="/cache/snapshots/" + "b" * 40)]
        with self.assertRaisesRegex(RuntimeError, "revision"):
            provider.embed_query("query")
        self.assertEqual(calls, [])
        provider.inventory_reader = lambda _: [model_row(), model_row(id="other", model_type="llm")]
        with self.assertRaisesRegex(RuntimeError, "generation model"):
            provider.embed_query("query")
        provider.inventory_reader = lambda _: [model_row()]
        provider.ownership_check = lambda _: False
        with self.assertRaisesRegex(RuntimeError, "does not own"):
            provider.embed_query("query")
        with self.assertRaisesRegex(ValueError, "port 8001"):
            LocalEmbeddingProvider(provider.identity, provider.home,
                                   endpoint="http://127.0.0.1:8000/v1")
        provider.ownership_check = lambda _: True
        provider.request = lambda *a, **k: {"model": "wrong", "data": []}
        with self.assertRaisesRegex(RuntimeError, "invalid embedding response"):
            provider.embed_query("query")

    def test_request_holds_shared_lease_and_rechecks_admission(self):
        provider, calls = self.fake_provider()
        observed = []

        def admit(*args, **kwargs):
            observed.append(kwargs["lease_held"])
            return True, "clear"

        provider.compute_admission = admit
        self.assertEqual(provider.embed_query("query"), [0.6, 0.8])
        self.assertEqual(observed, [True])
        with mavis_generation_lease(RuntimeConfig(home=provider.home), purpose="other-work"):
            with self.assertRaisesRegex(RuntimeError, "host lease"):
                provider.embed_query("query")
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(RuntimeError, "lease is not held"):
            host_admission(provider.home, lease_held=True)


if __name__ == "__main__":
    unittest.main()
