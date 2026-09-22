"""Versioned model profiles with exact identity and rollback."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .storage import read_json, require_safe_id, write_json


class ProfileStore:
    def __init__(self, home: Path):
        self.root = Path(home) / "profiles"

    def _role_root(self, role: str) -> Path:
        return self.root / require_safe_id(role, "profile role")

    def create_candidate(self, role: str, payload: dict[str, Any], inherited_from: str | None = None) -> Path:
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

    def activate(self, role: str, version: int, accepted_experiment: str, verifier_receipt: str) -> None:
        role_root = self._role_root(role)
        target = role_root / f"v{int(version)}.json"
        profile = read_json(target)
        previous = self.active_version(role)
        if previous is not None:
            previous_path = role_root / f"v{previous}.json"
            previous_profile = read_json(previous_path)
            previous_profile["status"] = "previous"
            write_json(previous_path, previous_profile)
        profile["status"] = "active"
        profile["accepted_experiment"] = accepted_experiment
        profile["verifier_receipt"] = verifier_receipt
        write_json(target, profile)
        write_json(role_root / "active.json", {"version": version, "path": str(target.resolve())})

    def restore(self, role: str, version: int) -> dict[str, Any]:
        target = self._role_root(role) / f"v{int(version)}.json"
        profile = read_json(target)
        if profile.get("status") not in {"active", "previous"}:
            raise ValueError("only accepted active or previous profiles can be restored")
        self.activate(role, version, str(profile.get("accepted_experiment") or "restored"), str(profile.get("verifier_receipt") or "retained"))
        return read_json(target)
