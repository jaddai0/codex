"""Negative fixtures for the source-only model-gateway ownership guard."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mavis.architecture import check_project, check_source
from mavis.evaluations import _local_response_url


PROJECT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def check_fixture(self, name: str, source: str) -> set[str]:
        with TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(source, encoding="utf-8")
            return {finding.rule for finding in check_source(path)}

    def test_current_project_passes(self):
        self.assertEqual(check_project(PROJECT), [])

    def test_comments_and_strings_are_not_imports_or_calls(self):
        self.assertEqual(self.check_fixture("worker.py", '''
# import anthropic; requests.post("https://api.anthropic.com")
message = "provider_registry = {'openai': 1, 'anthropic': 2}"
'''), set())

    def test_cloud_sdk_imports_and_dynamic_imports_fail(self):
        self.assertIn("forbidden-import", self.check_fixture("worker.py", "from openai import OpenAI as Client\n"))
        self.assertIn("forbidden-import", self.check_fixture("worker.py", "import google.genai as genai\n"))
        self.assertIn("dynamic-import", self.check_fixture("worker.py", "client = __import__('anthropic')\n"))

    def test_direct_http_transport_fails_outside_local_adapters(self):
        self.assertIn("network-import", self.check_fixture("worker.py", "import requests as r\nr.post('https://example.com')\n"))
        self.assertIn("network-import", self.check_fixture("worker.py", "from urllib import request as web\nweb.urlopen('https://example.com')\n"))
        self.assertIn("network-call", self.check_fixture("worker.py", "from urllib.request import urlopen\nurlopen('https://example.com')\n"))
        self.assertIn("network-process", self.check_fixture(
            "worker.py", "import subprocess\nsubprocess.run(['curl', 'https://api.example.com'])\n"))

    def test_local_adapter_rejects_unbounded_request_or_extra_call(self):
        source = '''
from urllib import request as web
def _post_json(url):
    safe = web.Request(_local_response_url(url))
    web.urlopen(safe)
    unsafe = web.Request("https://api.example.com")
    web.urlopen(unsafe)
'''
        rules = self.check_fixture("evaluations.py", source)
        self.assertIn("unbounded-url", rules)
        self.assertIn("network-call", rules)

    def test_nested_file_cannot_inherit_adapter_allowlist(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mavis" / "nested" / "runtime.py"
            path.parent.mkdir(parents=True)
            path.write_text("from urllib.request import urlopen\n")
            self.assertIn("network-import", {finding.rule for finding in
                            check_source(path, display="mavis/nested/runtime.py")})

    def test_provider_catalogs_fail_but_single_assignment_does_not(self):
        self.assertIn("provider-registry", self.check_fixture(
            "worker.py", "routes = {'openai': 'a', 'anthropic': 'b'}\n"))
        self.assertIn("provider-registry", self.check_fixture(
            "worker.py", "PROVIDER_REGISTRY = {'one': 'provider'}\n"))
        self.assertIn("provider-registry", self.check_fixture(
            "worker.py", "PROVIDERS = {'my-cloud': 'client'}\n"))
        self.assertNotIn("provider-registry", self.check_fixture(
            "worker.py", "assignment = {'provider': 'minimax', 'model': 'M3'}\n"))

    def test_manifest_checks_required_and_optional_dependencies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mavis").mkdir()
            (root / "mavis" / "__init__.py").write_text("")
            (root / "pyproject.toml").write_text('''
[build-system]
requires = ["cohere>=5"]
[project]
dependencies = ["OpenAI>=1", "requests>=2"]
[project.optional-dependencies]
cloud = ["anthropic[bedrock]>=0.3"]
''')
            forbidden = [finding.detail for finding in check_project(root)
                         if finding.rule == "forbidden-dependency"]
            self.assertEqual(forbidden, ["anthropic", "cohere", "openai", "requests"])

    def test_invalid_source_and_manifest_fail_closed(self):
        self.assertIn("syntax", self.check_fixture("worker.py", "def broken(:\n"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mavis").mkdir()
            (root / "pyproject.toml").write_text("[project\n")
            self.assertIn("manifest", {finding.rule for finding in check_project(root)})

    def test_e0_http_adapter_rejects_cloud_and_credentials(self):
        for url in ("https://api.openai.com/v1/responses",
                    "http://example.com/v1/responses",
                    "http://user:pass@127.0.0.1:8001/v1/responses"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _local_response_url(url)
        self.assertEqual(_local_response_url("http://127.0.0.1:8001/v1/responses"),
                         "http://127.0.0.1:8001/v1/responses")


if __name__ == "__main__":
    unittest.main()
