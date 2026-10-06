"""Metadata flags must never enter managed model preparation."""

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mavis"))

import launch_core


class PreparationReached(Exception):
    pass


class LaunchMetadataTests(unittest.TestCase):
    def test_exact_metadata_commands_forward_without_preparation(self):
        arguments = (
            ["--help"],
            ["-h"],
            ["--version"],
            ["-V"],
            ["exec", "--help"],
            ["exec", "-h"],
        )
        for metadata in arguments:
            with self.subTest(metadata=metadata):
                argv = ["/owned/trusted-core", *metadata]
                with (
                    patch.object(
                        launch_core,
                        "bind_project_root",
                        side_effect=AssertionError("metadata entered preparation"),
                    ),
                    patch.object(
                        launch_core,
                        "generation_lease",
                        side_effect=AssertionError("metadata entered model lease"),
                    ),
                    patch.object(
                        launch_core, "_launch_unlocked", return_value=17
                    ) as forward,
                ):
                    self.assertEqual(
                        launch_core.managed_launch(types.SimpleNamespace(), argv), 17
                    )
                    forward.assert_called_once_with(None, argv)

    def test_prompt_options_and_end_of_options_retain_preparation(self):
        arguments = (
            [],
            ["--help is a literal task"],
            ["--model", "--help"],
            ["-c", 'prompt="--help"'],
            ['--config=prompt="--help"'],
            ["--", "--help"],
            ["exec", "--", "--help"],
            ["exec", "--", "-h"],
            ["exec", "--", "--version"],
            ["--help", "literal task"],
            ["literal task", "--help"],
            ["exec", "--help", "literal task"],
            ["exec", "literal task", "-h"],
            ["exec", "--model", "--help"],
            ["exec", "--version"],
            ["--remote", "--help"],
            ["--help", "--remote"],
            ["remote-control", "--help"],
            ["--use-agent-identity-auth", "--version"],
            ["-c", 'model_provider="openai"', "--help"],
            ["--help=value"],
            ["--version=value"],
            ["-hV"],
            ["exec", "task"],
            ["--model"],
            ["help"],
            ["e", "--help"],
        )
        for ordinary in arguments:
            with self.subTest(ordinary=ordinary):
                argv = ["/owned/trusted-core", *ordinary]
                with (
                    patch.object(
                        launch_core,
                        "bind_project_root",
                        side_effect=PreparationReached,
                    ) as bind,
                    patch.object(launch_core, "_launch_unlocked") as forward,
                ):
                    with self.assertRaises(PreparationReached):
                        launch_core.managed_launch(types.SimpleNamespace(), argv)
                    bind.assert_called_once_with(argv)
                    forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
