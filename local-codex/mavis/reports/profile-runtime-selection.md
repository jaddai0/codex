# Accepted main profile launch selection

## Scope and result

This source slice connects an accepted `main` profile to the installed Mavis launcher. The launcher resolves the accepted profile before calling `runtime ensure`, so oMLX receives its exact `model_identity.model_id`. `prepare_runtime.py` reads the same versioned profile, rejects a conflicting explicit model, and writes Codex's selected model and combined base plus accepted system prompt into the managed catalog. The installed launcher now includes `launch_core.py`, which spawns Codex and writes a host-observed receipt containing the process ID, profile version, previous version, exact model identity metadata, selected oMLX model ID, and hashes of the profile, effective config, catalog, and system prompt. It records the exit code when Codex exits.

The accepted main profile supports one applied candidate field: `prompts.system`. `tool_settings` must be empty and `context_policy` must be exactly `{"retrieval": {}}`. Activation and launch reject unsupported settings, missing exact model ID, wrong runtime, missing retained evidence, and changed evidence hashes. The profile store records the actual prior active version when activating or restoring an accepted profile. Launch resolution now also checks the canonical active experiment pointer, promoted record, candidate snapshot, and retained review receipt. A rolled-back experiment cannot remain launchable through a stale profile pointer. The launch wrapper refuses command line model, alternate provider, arbitrary config, feature, and named Codex profile overrides. It rechecks active profile and experiment state and effective file hashes immediately before spawning Codex; the launcher performs the argument check before runtime startup.

## Verification

- `PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -v`: 95 passed.
- `python3 -m unittest discover -s local-codex/tests -p test_prepare_runtime.py -v`: 15 passed. Fixtures exercised A, B, and rollback to A, checked generated TOML and catalog, spawned a stub core through the wrapper, and ran the actual launcher against a fake oMLX inventory and stub runtime to verify model handoff and receipt.
- `zsh -n local-codex/bin/local-codex`, `python3 -m compileall -q local-codex/prepare_runtime.py local-codex/launch_core.py local-codex/mavis/mavis`, and `git diff --check`: passed.

## Evidence boundary and remaining work

No model was loaded and no live or installed Mavis configuration was changed. The test proves source wiring, a stub process spawn, rollback rejection, and that command line overrides are rejected before runtime startup. It does not prove the installed Codex binary ran with a real accepted profile, the oMLX server actually served the selected weight files, or that profile fingerprint metadata matches those on-disk files. Those need a later safe installed-runtime canary. E0 and E1 performance or quality comparisons were not run here and cannot be called passing from these tests. Runtime version remains retained metadata; the launcher gates on oMLX by name but does not independently attest the live oMLX version. Profile promotion status was checked at activation; launch checks retained file hashes but does not refresh external gateway or verifier status.
