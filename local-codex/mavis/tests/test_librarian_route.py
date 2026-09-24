import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.cli import main
from mavis.helpers import HelperSession
from mavis.librarian_route import LibrarianAskError, ask
from mavis.transcripts import TranscriptArchive


BASE_URL = "http://127.0.0.1:8001/v1"
MODEL = "Qwen3.5-0.8B"
SERVICE_VERSION = "test-service/1.0"


def _citation(packet):
    return {key: packet[0][key] for key in ("path", "line", "sha256")}


def _stub_binding_reader(home: Path, base_url: str, model_id: str, model_path: Path,
                         inventory_reader=None, status_reader=None,
                         settings_reader=None) -> dict:
    """Stub helper_eval._model_binding used by route-internal binding tests.

    Reads the requested model_path directly so tests can prove the route
    actually invokes the binding with the expected candidate path.
    """
    if model_path is None:
        raise ValueError("helper service has no unique model ID/path mapping")
    expected = model_path.resolve(strict=False)
    if not expected.exists():
        raise ValueError("helper model path must be a directory")
    return {
        "model_path": str(expected),
        "service_version": SERVICE_VERSION,
        "engine_type": "mlx",
        "model_context_length": 8192,
        "global_settings_sha256": "deadbeef" * 8,
    }


class LibrarianRouteNormalAnswerTests(unittest.TestCase):
    def test_no_model_returns_verified_packet_only(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history")
            archive.append_segment([
                {"content": "Decision: keep the blue engine after the second review."},
            ])
            result = ask(home, home, "history", "Which engine?", ["blue engine"],
                         no_model=True)
            self.assertEqual(result["mode"], "deterministic")
            self.assertFalse(result["model_called"])
            self.assertEqual(result["evidence_lines"], 1)
            self.assertEqual(len(result["citations"]), 1)
            self.assertIn("blue engine", result["answer"])
            self.assertIn("helper was not called", result["uncertainty"])
            self.assertIsNone(result["followup_query"])
            self.assertEqual(result["prior_citations_kept"], 0)

    def test_model_call_validates_strict_and_stores_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history")
            archive.append_segment([{"content": "Decision: keep the blue engine."}])
            packet = archive.search("blue engine")
            citation = _citation(packet)

            def ask_fn(*args, **kwargs):
                return {"answer": "Keep the blue engine.",
                        "uncertainty": "The reason is not recorded.",
                        "citations": [citation]}

            result = ask(home, home, "history", "Which engine?", ["blue engine"],
                         model_id=MODEL, model_path=home,
                         ask_fn=ask_fn,
                         binding_reader=_stub_binding_reader)
            self.assertEqual(result["mode"], "model")
            self.assertTrue(result["model_called"])
            self.assertEqual(result["model_id"], MODEL)
            self.assertEqual(result["evidence_lines"], 1)
            self.assertIn("context_expires_at_epoch", result)
            self.assertEqual(result["request_settings"]["temperature"], 0)
            self.assertIn("citations", result["response_schema"]["properties"])
            self.assertEqual(result["model_binding"]["service_version"], SERVICE_VERSION)
            session = HelperSession(home, "librarian", context_id="history")
            context = session.followup_context()
            self.assertIsNotNone(context)
            self.assertEqual(context["query"], "Which engine?")
            self.assertEqual(context["citations"], [citation])

class LibrarianRouteFollowupTests(unittest.TestCase):
    def test_followup_passes_prior_query_and_drops_stale_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history")
            archive.append_segment([{"content": "Decision: keep the blue engine"}])
            archive.append_segment([{"content": "Decision: delete stale paths"}])
            prior_hit = archive.search("blue engine")[0]
            session = HelperSession(home, "librarian", context_id="history")
            session.store_query_context("Which engine?",
                                        [_citation([prior_hit])])
            current_hit = archive.search("stale paths")[0]
            citation = _citation([current_hit])

            def ask_fn(*args, **kwargs):
                messages = args[2]
                user = json.loads(messages[1]["content"])
                self.assertEqual(user["followup"], {"query": "Which engine?"})
                return {"answer": "Delete stale paths.",
                        "uncertainty": "No other context.",
                        "citations": [citation]}

            result = ask(home, home, "history", "What about stale paths?",
                         ["stale paths"], model_id=MODEL, model_path=home,
                         followup=True, ask_fn=ask_fn,
                         binding_reader=_stub_binding_reader)
            self.assertEqual(result["followup_query"], "Which engine?")

    def test_followup_isolated_per_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            for cid in ("alpha", "beta"):
                archive = TranscriptArchive(home, cid)
                archive.append_segment([{"content": f"Decision in {cid}"}])
            first_session = HelperSession(home, "librarian", context_id="alpha")
            first_session.store_query_context("alpha question",
                                              [{"path": "x", "line": 1,
                                                "sha256": "deadbeef"}])
            beta_archive = TranscriptArchive(home, "beta")
            beta_hit = beta_archive.search("beta")[0]
            citation = _citation([beta_hit])

            def ask_fn(*args, **kwargs):
                return {"answer": "beta decision",
                        "uncertainty": "single line only",
                        "citations": [citation]}

            ask(home, home, "beta", "?", ["beta"], model_id=MODEL,
                model_path=home, followup=True, ask_fn=ask_fn,
                binding_reader=_stub_binding_reader)
            self.assertEqual(first_session.followup_context()["query"],
                             "alpha question")

    def test_followup_expires_after_sixty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history")
            archive.append_segment([{"content": "Decision: keep the blue engine"}])
            packet = archive.search("blue engine")
            citation = _citation(packet)

            def ask_fn(*args, **kwargs):
                return {"answer": "blue", "uncertainty": "u",
                        "citations": [citation]}

            ask(home, home, "history", "Which engine?", ["blue engine"],
                model_id=MODEL, model_path=home, ask_fn=ask_fn,
                binding_reader=_stub_binding_reader, now=100.0)
            session = HelperSession(home, "librarian", context_id="history")
            self.assertIsNotNone(session.followup_context(now=159.99))
            self.assertIsNone(session.followup_context(now=160.0))


class LibrarianRoutePagingTests(unittest.TestCase):
    def test_paging_with_offset_returns_subsequent_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history")
            archive.append_segment([
                {"content": "decision one"},
                {"content": "decision two"},
                {"content": "decision three"},
                {"content": "decision four"},
            ])
            result = ask(home, home, "history", "list decisions", ["decision"],
                         no_model=True, limit=2, offset=2)
            self.assertEqual(result["evidence_lines"], 2)
            self.assertEqual(len(result["citations"]), 2)
            self.assertIn("decision three", result["answer"])
            self.assertIn("decision four", result["answer"])

class LibrarianRouteRejectionTests(unittest.TestCase):
    def test_rejects_followup_citation_laundering(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history")
            archive.append_segment([
                {"content": "Decision: keep the blue engine"},
                {"content": "Decision: delete stale paths"},
            ])
            prior_hit = archive.search("blue engine")[0]
            session = HelperSession(home, "librarian", context_id="history")
            session.store_query_context("Which engine?",
                                        [_citation([prior_hit])])
            borrowing_prior = _citation([prior_hit])

            def ask_fn(*args, **kwargs):
                return {"answer": "blue engine", "uncertainty": "u",
                        "citations": [borrowing_prior]}

            with self.assertRaisesRegex(LibrarianAskError, "outside the verified evidence"):
                ask(home, home, "history", "stale paths?",
                    ["stale paths"], model_id=MODEL, model_path=home,
                    followup=True, ask_fn=ask_fn,
                    binding_reader=_stub_binding_reader)

class LibrarianRouteFailureTests(unittest.TestCase):
    def test_no_evidence_never_calls_model(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            TranscriptArchive(home, "history").append_segment([{"content": "red engine"}])
            with patch("mavis.librarian_route._completion") as completion:
                with self.assertRaisesRegex(LibrarianAskError, "no verified transcript"):
                    ask(home, home, "history", "?", ["blue engine"],
                        model_id=MODEL, model_path=home,
                        binding_reader=_stub_binding_reader)
                completion.assert_not_called()


class LibrarianRouteCliTests(unittest.TestCase):
    def test_cli_no_model_routes_and_prints_payload(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            project = home / "repo"
            project.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=project, check=True)
            archive = TranscriptArchive(project / ".mavis", "conv-1")
            archive.append_segment([{"content": "Decision: blue engine"}])
            output = io.StringIO()
            with patch.dict(os.environ, {"MAVIS_HOME": str(home), "MAVIS_PROJECT_ROOT": ""}), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(main([
                    "librarian", "ask", "conv-1", "which engine?",
                    "--search-term", "blue engine", "--project", str(project),
                    "--no-model",
                ]), 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["mode"], "deterministic")
            self.assertEqual(payload["evidence_lines"], 1)
            self.assertEqual((project / ".mavis" / ".gitignore").read_text(), "*\n")
            for extra_args in ([], ["--project", str(project), "--model-id", MODEL]):
                output = io.StringIO()
                with patch.dict(os.environ, {"MAVIS_HOME": str(home),
                                                 "MAVIS_PROJECT_ROOT": ""}), \
                     contextlib.redirect_stdout(output):
                    self.assertEqual(main([
                        "librarian", "ask", "conv-1", "which engine?",
                        "--search-term", "blue engine", "--no-model", *extra_args,
                    ]), 1)
                self.assertEqual(json.loads(output.getvalue())["status"], "fail")
            output = io.StringIO()
            with patch.dict(os.environ, {"MAVIS_HOME": str(home),
                                             "MAVIS_PROJECT_ROOT": str(project)}), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(main(["librarian", "ask", "conv-1", "which engine?",
                                       "--search-term", "blue engine", "--no-model",
                                       "--project", str(home)]), 1)
            self.assertIn("not a Git checkout", json.loads(output.getvalue())["reason"])

class LibrarianRouteCliBindingRegressionTests(unittest.TestCase):
    """Exercise the real binding at the CLI boundary before completion."""

    def test_cli_binding_errors_are_structured(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            project = home / "repo"
            project.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=project, check=True)
            TranscriptArchive(project / ".mavis", "conv-1").append_segment(
                [{"content": "Decision: keep the blue engine"}])
            argv = ["librarian", "ask", "conv-1", "Which engine?",
                    "--search-term", "blue engine", "--project", str(project),
                    "--model-id", "fake-model"]
            for model_path, service_error in (
                (home / "missing-model", None),
                (home, RuntimeError("service inventory unavailable")),
            ):
                with self.subTest(model_path=model_path):
                    output = io.StringIO()
                    with patch.dict(os.environ, {"MAVIS_HOME": str(home), "MAVIS_PROJECT_ROOT": ""}), \
                         patch("mavis.librarian_route._completion") as completion, \
                         patch("mavis.runtime.inventory", side_effect=service_error), \
                         contextlib.redirect_stdout(output):
                        rc = main([*argv, "--model-path", str(model_path)])
                    self.assertEqual(rc, 1)
                    result = json.loads(output.getvalue())
                    self.assertEqual(result["status"], "fail")
                    self.assertIn("helper binding refused", result["reason"])
                    completion.assert_not_called()

    def test_cli_service_and_archive_homes_are_distinct(self):
        """Project-local archive must work with a separate service home."""
        import shutil
        import subprocess
        root = Path(tempfile.mkdtemp())
        try:
            service_home = root / "service"
            project = root / "project"
            archive_home = project / ".mavis"
            service_home.mkdir()
            project.mkdir()
            archive_home.mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=project, check=True)
            archive = TranscriptArchive(archive_home, "conv-1")
            archive.append_segment([{"content": "Decision: keep the blue engine"}])
            packet = archive.search("blue engine")
            citation = {key: packet[0][key] for key in ("path", "line", "sha256")}
            observed = {}

            def fake_completion(*args, **kwargs):
                return {"answer": "Keep the blue engine.",
                        "uncertainty": "low.",
                        "citations": [citation]}

            def spy_binding(home, base_url, model_id, model_path,
                            inventory_reader=None, status_reader=None,
                            settings_reader=None):
                observed["service_home"] = home
                observed["model_path"] = model_path
                return _stub_binding_reader(home, base_url, model_id, model_path,
                                            inventory_reader=inventory_reader,
                                            status_reader=status_reader,
                                            settings_reader=settings_reader)

            output = io.StringIO()
            err_buf = io.StringIO()
            with patch.dict(os.environ, {"MAVIS_HOME": str(service_home),
                                         "MAVIS_PROJECT_ROOT": str(project)}), \
                 patch("mavis.librarian_route._completion", side_effect=fake_completion), \
                 patch("mavis.librarian_route._model_binding", side_effect=spy_binding), \
                 contextlib.redirect_stdout(output), \
                 contextlib.redirect_stderr(err_buf):
                rc = main([
                    "librarian", "ask", "conv-1", "Which engine?",
                    "--search-term", "blue engine",
                    "--model-id", MODEL,
                    "--model-path", str(root),
                ])
            if rc != 0:
                self.fail(f"cli rc={rc} stdout={output.getvalue()!r} stderr={err_buf.getvalue()!r}")
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(observed["service_home"], service_home,
                             "binding must use the service home, not the archive home")
            self.assertEqual(payload["model_binding"]["service_version"], SERVICE_VERSION)
            session = HelperSession(archive_home, "librarian", context_id="conv-1")
            self.assertIsNotNone(session.followup_context(),
                                 "follow-up session must live in the archive home")
            output = io.StringIO()
            with patch.dict(os.environ, {"MAVIS_HOME": str(service_home),
                                             "MAVIS_PROJECT_ROOT": str(project)}), \
                 patch("mavis.librarian_route._completion", side_effect=fake_completion), \
                 patch("mavis.librarian_route._model_binding", side_effect=spy_binding), \
                 patch("mavis.librarian_route.HelperSession.store_query_context",
                       side_effect=RuntimeError("expiry worker could not start")), \
                 contextlib.redirect_stdout(output):
                rc = main(["librarian", "ask", "conv-1", "Which engine?",
                           "--search-term", "blue engine", "--model-id", MODEL,
                           "--model-path", str(root)])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "fail")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
