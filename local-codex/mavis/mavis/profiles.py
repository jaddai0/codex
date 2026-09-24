"""Versioned model profiles with exact identity and rollback."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .experiments import ExperimentStore, _profile_transition_lease
from .profile_transition import pending as profile_transition_pending
from .storage import profile_boundary_lock, read_json, require_safe_id, sha256_file, write_json


class ProfileStore:
    def __init__(self, home: Path, gateway_status_reader: Callable[[str], dict[str, Any]] | None = None):
        self.root = Path(home) / "profiles"
        self.experiments = ExperimentStore(Path(home), gateway_status_reader=gateway_status_reader)

    def _role_root(self, role: str) -> Path:
        return self.root / require_safe_id(role, "profile role")

    def create_candidate(self, role: str, payload: dict[str, Any], inherited_from: str | None = None) -> Path:
        with profile_boundary_lock(self.root.parent):
            if profile_transition_pending(self.root.parent):
                raise ValueError("an interrupted profile transition needs recovery")
            role_root = self._role_root(role)
            versions = [int(path.stem[1:]) for path in role_root.glob("v*.json") if path.stem[1:].isdigit()]
            version = max(versions, default=0) + 1
            profile = deepcopy(payload)
            profile.update(
                {
                    "schema_version": "mavis.model-profile/v1",
                    "role": role,
                    "version": version,
                    "previous_version": self.active_version(role),
                    "status": "candidate",
                }
            )
            if inherited_from:
                profile["candidate_inheritance"] = {"profile_id": inherited_from, "passed_status_inherited": False, "adapters_inherited": False}
            path = role_root / f"v{version}.json"
            write_json(path, profile)
            return path

    def active_version(self, role: str) -> int | None:
        pointer = self._role_root(role) / "active.json"
        return int(read_json(pointer)["version"]) if pointer.exists() else None

    def activate_bootstrap_baseline(self) -> Path:
        """Accept the reviewed seed as a launchable first main profile."""
        from .e1_bootstrap import validate_bootstrap

        home = self.root.parent.resolve()
        with _profile_transition_lease(home), profile_boundary_lock(home):
            if profile_transition_pending(home):
                raise ValueError("an interrupted profile transition needs recovery")
            self.experiments._assert_objective_boundary()
            if self.active_version("main") is not None:
                raise ValueError("main already has an accepted profile")
            bootstrap = validate_bootstrap(home, status_reader=self.experiments.gateway_status_reader)
            active = self.experiments.active("main")
            if active.get("experiment_id") is not None:
                raise ValueError("bootstrap baseline needs the unpromoted seed")
            baseline = self.experiments._read_snapshot(active["configuration"])
            if baseline != {"prompts": bootstrap["prompts"], "tool_settings": {}, "retrieval": {}}:
                raise ValueError("bootstrap prompt differs from the active seed")
            root = self._role_root("main")
            path = root / "v1.json"
            profile = {
                "schema_version": "mavis.model-profile/v1",
                "profile_id": "main-e0-bootstrap",
                "role": "main", "version": 1, "previous_version": None,
                "status": "active", "model_identity": bootstrap["model_identity"],
                "runtime": {"name": "omlx", "version": "installed"},
                "prompts": bootstrap["prompts"], "tool_settings": {},
                "context_policy": {"retrieval": {}}, "experiments": [],
                "accepted_bootstrap": {"path": bootstrap["_source_path"],
                                       "sha256": bootstrap["_source_sha256"]},
                "verifier_receipt": bootstrap["independent_review"],
            }
            if path.exists():
                if read_json(path) != profile:
                    raise ValueError("existing first profile differs from bootstrap")
            else:
                write_json(path, profile)
            write_json(root / "active.json", {"version": 1, "path": str(path.resolve())})
            return path

    def activate(
        self,
        role: str,
        version: int,
        accepted_experiment: Path,
        verifier_receipt: Path,
    ) -> None:
        home = self.root.parent.resolve()
        with _profile_transition_lease(home), profile_boundary_lock(home):
            if profile_transition_pending(home):
                raise ValueError("an interrupted profile transition needs recovery")
            self.experiments._assert_objective_boundary()
            if self.active_version(role) == version:
                active = read_json(self._role_root(role) / f"v{version}.json")
                if (active.get("status") == "active"
                        and active.get("accepted_experiment", {}).get("path") == str(Path(accepted_experiment).resolve())
                        and active.get("accepted_experiment", {}).get("sha256") == sha256_file(accepted_experiment)
                        and active.get("verifier_receipt", {}).get("path") == str(Path(verifier_receipt).resolve())
                        and active.get("verifier_receipt", {}).get("sha256") == sha256_file(verifier_receipt)):
                    return
            self._activate_unlocked(role, version, accepted_experiment, verifier_receipt)

    def _activate_unlocked(
        self,
        role: str,
        version: int,
        accepted_experiment: Path,
        verifier_receipt: Path,
    ) -> None:
        role_root = self._role_root(role)
        target = role_root / f"v{int(version)}.json"
        profile = read_json(target)
        experiment_path = Path(accepted_experiment).resolve()
        verifier_path = Path(verifier_receipt).resolve()
        home = self.root.parent.resolve()
        if home / "experiments" not in experiment_path.parents:
            raise ValueError("accepted experiment must be retained under the Mavis experiment store")
        if home / "verifications" not in verifier_path.parents:
            raise ValueError("verifier receipt must be retained under the Mavis verification store")
        experiment = read_json(experiment_path)
        if experiment.get("schema_version") != "mavis.experiment-lifecycle/v1":
            raise ValueError("profile activation requires a promoted lifecycle record")
        experiment_id = require_safe_id(str(experiment.get("experiment_id") or ""), "experiment id")
        if experiment_path != self.experiments._record_path(experiment_id).resolve():
            raise ValueError("profile activation requires the canonical lifecycle record")
        experiment = self.experiments.assert_promoted(experiment_id)
        if experiment["scope"] != role:
            raise ValueError("promoted experiment belongs to another profile role")
        if verifier_path != Path(experiment["review"]["path"]).resolve() or sha256_file(verifier_path) != experiment["review"]["sha256"]:
            raise ValueError("profile verifier receipt does not match promoted experiment")
        if str(experiment.get("experiment_id")) not in profile.get("experiments", []):
            raise ValueError("promoted experiment is not bound to this profile")
        candidate = self.experiments._read_snapshot(experiment["candidate"])
        profile_config = {"prompts": profile.get("prompts"), "tool_settings": profile.get("tool_settings"),
                          "retrieval": (profile.get("context_policy") or {}).get("retrieval", {})}
        if candidate != profile_config:
            raise ValueError("profile configuration differs from promoted candidate")
        if role == "main":
            identity = profile.get("model_identity") or {}
            if not isinstance(identity.get("model_id"), str) or not identity["model_id"]:
                raise ValueError("main profile requires exact model_identity.model_id")
            if (profile.get("runtime") or {}).get("name") != "omlx":
                raise ValueError("main profile requires oMLX runtime")
            prompts = profile.get("prompts")
            if not isinstance(prompts, dict) or set(prompts) - {"system"} or not isinstance(prompts.get("system", ""), str):
                raise ValueError("main profile has unsupported prompt settings")
            if profile.get("tool_settings") != {} or profile.get("context_policy") != {"retrieval": {}}:
                raise ValueError("main profile has settings the Codex launcher cannot apply")
        previous = self.active_version(role)
        profile["status"] = "active"
        profile["previous_version"] = previous
        profile["accepted_experiment"] = {
            "path": str(experiment_path),
            "sha256": sha256_file(experiment_path),
        }
        profile["verifier_receipt"] = {
            "path": str(verifier_path),
            "sha256": sha256_file(verifier_path),
        }
        write_json(target, profile)
        write_json(role_root / "active.json", {"version": version, "path": str(target.resolve())})

    def restore(self, role: str, version: int) -> dict[str, Any]:
        if profile_transition_pending(self.root.parent):
            raise ValueError("an interrupted profile transition needs recovery")
        target = self._role_root(role) / f"v{int(version)}.json"
        profile = read_json(target)
        if self.active_version(role) == version and profile.get("status") == "active":
            return profile
        if profile.get("status") not in {"active", "previous"}:
            raise ValueError("only accepted active or previous profiles can be restored")
        experiment = profile.get("accepted_experiment") or {}
        verification = profile.get("verifier_receipt") or {}
        for retained, label in ((experiment, "experiment"), (verification, "verification")):
            if not isinstance(retained, dict) or not retained.get("path") or not retained.get("sha256"):
                raise ValueError(f"accepted profile is missing retained {label} evidence")
            if sha256_file(Path(retained["path"])) != retained["sha256"]:
                raise ValueError(f"retained {label} evidence hash changed")
        self.activate(role, version, Path(experiment["path"]), Path(verification["path"]))
        return read_json(target)
