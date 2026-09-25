"""The direct GLM reviewer is found beside the configured gateway launcher."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.gateway import GatewayUnavailable, glm_review_environment, glm_review_launcher


class GlmReviewLauncherTests(unittest.TestCase):
    def _codex_home(self, root: Path, *, launcher: bool) -> Path:
        bin_dir = root / "gateway" / "bin"
        bin_dir.mkdir(parents=True)
        server = bin_dir / "mcp-server.sh"
        server.write_text("#!/bin/sh\n")
        server.chmod(0o755)
        if launcher:
            glm = bin_dir / "glm-codex.sh"
            glm.write_text("#!/bin/sh\n")
            glm.chmod(0o755)
        home = root / "codex-home"
        home.mkdir()
        # A path to a placeholder env file, never a live credential.
        (home / "config.toml").write_text(
            '[mcp_servers.model-gateway]\n'
            f'command = "{server}"\n'
            f'env = {{ MODEL_GATEWAY_ENV_FILE = "{root / "synthetic.env"}" }}\n'
        )
        return home

    def test_launcher_sits_beside_the_configured_gateway(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = self._codex_home(root, launcher=True)
            with patch.dict(os.environ, {"CODEX_HOME": str(home)}):
                self.assertEqual(glm_review_launcher(), root / "gateway" / "bin" / "glm-codex.sh")
                environment = glm_review_environment()
                self.assertEqual(environment["MODEL_GATEWAY_ENV_FILE"], str(root / "synthetic.env"))
                self.assertNotIn("CODEX_HOME", environment)

    def test_unset_codex_home_uses_the_mavis_codex_home(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._codex_home(root, launcher=True)
            (root / ".local-codex").symlink_to(root / "codex-home")
            environment = {key: value for key, value in os.environ.items() if key != "CODEX_HOME"}
            with patch.dict(os.environ, environment, clear=True), \
                    patch("mavis.gateway.Path.home", return_value=root):
                self.assertEqual(glm_review_launcher(), root / "gateway" / "bin" / "glm-codex.sh")

    def test_missing_launcher_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            home = self._codex_home(Path(directory), launcher=False)
            with patch.dict(os.environ, {"CODEX_HOME": str(home)}):
                with self.assertRaisesRegex(GatewayUnavailable, "GLM reviewer launcher"):
                    glm_review_launcher()


if __name__ == "__main__":
    unittest.main()
