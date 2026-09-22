# Mavis Phase 3 source-only experiment lifecycle

**Date:** 2026-09-22
**Branch:** `codex/mavis-phase3-experiments` in an isolated worktree from `omlx-only` commit `ffbe0e8307aa1da73255a8901ca37a66a7360235`.
**Scope:** reversible prompt, tool-setting, and retrieval configuration changes. No model load, live service action, or paid request.

## What was built

- Added `mavis.experiments.ExperimentStore` and a versioned lifecycle record schema. The Mavis home now holds content-hashed, 0600 configuration snapshots under 0700 store directories and atomic JSON state records. The active baseline can be seeded once. Each candidate changes only one declared section (`prompts`, `tool_settings`, or `retrieval`); the frozen baseline, workload, held-out case IDs, and minimum target gain are retained.
- Comparison requires baseline and candidate results tied to their exact configuration and workload hashes, the same held-out cases, retained output file hashes, a passing candidate mandatory gate, finite scores, and the stated minimum improvement. A baseline may fail the regression being repaired; the candidate may not fail mandatory cases. Changed evidence blocks later staging and promotion.
- Review imports a separate receipt bound to the baseline, candidate, and comparison hashes. It then asks the configured gateway for the review worker's current status and applies the same host-receipt validation used by `ObjectiveStore`. The gateway must report an accepted completed worker, a distinct Terra verifier job, exact acceptance hashes, and a Mavis binding for this experiment whose requirement contains the exact comparison digest. A reviewer name or file alone cannot promote. Staging and promotion re-read the gateway status and reject a changed receipt or revoked acceptance.
- Staging leaves the active configuration untouched. Promotion requires an explicit between-objectives boundary, the same active baseline, and the staged candidate. It retains the previous active pointer. Rollback restores that prior pointer with a recorded reason. If the active pointer was written but the promotion record was interrupted, active reads fail closed and a repeated promotion finalizes the record.
- `ProfileStore.activate` now rejects its former arbitrary `mavis.experiment/v1` plus `mavis.verifier/v1` file pair. It requires the canonical promoted lifecycle record, a candidate configuration exactly matching the profile, its exact review receipt hash, and fresh gateway/Terra acceptance. Restoring an earlier profile requires rolling back the newer experiment first. This closes a file-only promotion bypass.
- `ExperimentStore.enqueue`, `checkpoint`, and `resume_checkpoint` use the existing maintenance queue. A foreground task prevents claim; a running job checkpoints to `paused` when foreground work returns. The checkpoint binds to the experiment record hash and state, so changed work cannot silently resume from an old step.

## Verification

| Requirement | Evidence |
|---|---|
| Frozen baseline and one-section candidate | Tests reject a second baseline, altered snapshot, and a candidate changing another section. |
| Matched comparison | Tests reject insufficient gain, reordered held-out cases, failed candidate mandatory case, and changed output evidence. |
| Independent review | Tests reject a file outside verification storage, same-job verifier, modified receipt, and gateway status whose verifier changes. |
| Stage, promote, rollback | Tests show staging preserves active, promotion requires the between-objectives flag, and rollback restores the exact prior active record across a reopened store. Profile activation rejects forged legacy evidence and restores A after B rolls back. |
| Resumable maintenance | Tests show foreground yielding, paused job reclaim, exact state restoration, and refusal to resume an obsolete checkpoint. |
| Interrupted promotion | A fixture simulates the active-pointer write before record completion; active reads refuse it and a repeated promotion completes it. |

Commands in the isolated worktree:

```text
PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -q
# 60 tests passed
python3 -m compileall -q local-codex/mavis/mavis
# passed
git diff --check
# passed
```

## Exact acceptance gaps

The new lifecycle has only source and fixture evidence. No real Mavis failure has yet produced a candidate and regression case. No E1 held-out workload has run both frozen arms, no native gateway review assignment has been dispatched for the comparison digest, and no installed profile loader reads the active experiment pointer. No real staged promotion or rollback has been observed. The source tests inject gateway status and write synthetic output files; they establish the local gate logic, not live Phase 3 acceptance. The receipt is revalidated through the configured gateway, but files and test fixtures within the same user account are not a cryptographic trust boundary against an attacker with unrestricted local write access.

The CLI still has no explicit profile selection or lifecycle commands. This Phase 1 integration gap is outside the trust-boundary repair in this branch; the new source store does not make profile switching available to a user or the installed runtime.

The next integration step is a maintenance runner that executes E1 shards, stores host output receipts, schedules a native gateway review assignment with the exact `experiment-comparison` requirement, and applies the active configuration between objectives. Then run one real failure through that chain and deliberately roll back the promoted candidate after a seeded critical regression. Phase 3 remains **partial** until those observations exist.

## Repository delivery

The source slice is committed locally as `89c422fc6` on `codex/mavis-phase3-experiments`. Push is pending: `gh repo view jaddai0/codex --json visibility` reported `PUBLIC`, and the checkout has no private Codex remote. The requested destination was a private fork branch, so this lane did not upload the commit to the public fork. A private destination or explicit destination change is needed.
