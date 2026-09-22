import json
from pathlib import Path
import tempfile
import unittest

from mavis.profiles import ProfileStore


def profile(profile_id, experiments=None):
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
        "experiments": experiments or [],
    }


class ProfileStoreTests(unittest.TestCase):
    def promotion_evidence(self, root: Path, experiment_id: str):
        experiment = root / "experiments" / f"{experiment_id}.json"
        verifier = root / "verifications" / f"{experiment_id}.json"
        experiment.parent.mkdir(parents=True, exist_ok=True)
        verifier.parent.mkdir(parents=True, exist_ok=True)
        experiment.write_text(json.dumps({
            "schema_version": "mavis.experiment/v1",
            "experiment_id": experiment_id,
            "promotion_decision": "promote",
        }))
        verifier.write_text(json.dumps({
            "schema_version": "mavis.verifier/v1",
            "verdict": "accepted",
        }))
        return experiment, verifier

    def test_switching_back_restores_exact_accepted_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            exp_a, verify_a = self.promotion_evidence(Path(directory), "exp-a")
            exp_b, verify_b = self.promotion_evidence(Path(directory), "exp-b")
            first = store.create_candidate("main", profile("a", ["exp-a"]))
            store.activate("main", 1, exp_a, verify_a)
            store.create_candidate("main", profile("b", ["exp-b"]), inherited_from="a")
            store.activate("main", 2, exp_b, verify_b)
            restored = store.restore("main", 1)
            self.assertEqual(restored["profile_id"], "a")
            self.assertEqual(first.read_text(), (Path(directory) / "profiles/main/v1.json").read_text())
            self.assertEqual(store.active_version("main"), 1)

    def test_activation_rejects_unretained_arbitrary_evidence_names(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            store.create_candidate("main", profile("a", ["exp-a"]))
            with self.assertRaisesRegex(ValueError, "experiment store"):
                store.activate("main", 1, Path("exp-a"), Path("verify-a"))

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
