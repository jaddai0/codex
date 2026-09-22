from pathlib import Path
import tempfile
import unittest

from mavis.profiles import ProfileStore


def profile(profile_id):
    return {
        "profile_id": profile_id,
        "model_identity": {
            "architecture": "qwen",
            "weights_fingerprint": "weights",
            "tokenizer_fingerprint": "tokenizer",
            "chat_template_fingerprint": "template",
            "quantization": "4bit",
        },
        "runtime": {"name": "omlx", "version": "1"},
        "prompts": {},
        "tool_settings": {},
        "context_policy": {},
        "experiments": [],
    }


class ProfileStoreTests(unittest.TestCase):
    def test_switching_back_restores_exact_accepted_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            first = store.create_candidate("main", profile("a"))
            store.activate("main", 1, "exp-a", "verify-a")
            store.create_candidate("main", profile("b"), inherited_from="a")
            store.activate("main", 2, "exp-b", "verify-b")
            restored = store.restore("main", 1)
            self.assertEqual(restored["profile_id"], "a")
            self.assertEqual(first.read_text(), (Path(directory) / "profiles/main/v1.json").read_text())
            self.assertEqual(store.active_version("main"), 1)

    def test_family_inheritance_never_inherits_pass_or_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            path = store.create_candidate("main", profile("child"), inherited_from="parent")
            payload = __import__("json").loads(path.read_text())
            self.assertEqual(
                payload["candidate_inheritance"],
                {"profile_id": "parent", "passed_status_inherited": False, "adapters_inherited": False},
            )


if __name__ == "__main__":
    unittest.main()
