import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PhaseZeroContractTests(unittest.TestCase):
    def test_six_versioned_contracts_are_strict_json_schemas(self):
        expected = {
            "objective.schema.json": "mavis.objective/v1",
            "worker-assignment.schema.json": "mavis.worker-assignment/v1",
            "evidence-receipt.schema.json": "mavis.evidence-receipt/v1",
            "model-profile.schema.json": "mavis.model-profile/v1",
            "memory-record.schema.json": "mavis.memory-record/v1",
            "experiment.schema.json": "mavis.experiment/v1",
        }
        observed = {}
        for name, version in expected.items():
            payload = json.loads((ROOT / "contracts" / name).read_text())
            self.assertFalse(payload["additionalProperties"])
            self.assertEqual(payload["properties"]["schema_version"]["const"], version)
            observed[name] = version
        self.assertEqual(observed, expected)

    def test_templates_match_versioned_contracts(self):
        worker = json.loads((ROOT / "handoff" / "worker-assignment.template.json").read_text())
        verifier = json.loads((ROOT / "handoff" / "verifier.template.json").read_text())
        self.assertEqual(worker["schema_version"], "mavis.worker-assignment/v1")
        self.assertEqual(verifier["schema_version"], "mavis.verifier/v1")

    def test_baseline_separates_iris_and_records_exact_model(self):
        baseline = json.loads((ROOT / "baseline" / "2026-09-22.json").read_text())
        self.assertEqual(baseline["local_runtime"]["iris_endpoint"], "http://127.0.0.1:8000/v1")
        self.assertEqual(baseline["local_runtime"]["loaded_main_model"], "Qwen3.8-Flash-Next-Abliterated-MLX-4bit")
        self.assertEqual(len(baseline["model_identity"]["architecture_config_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
