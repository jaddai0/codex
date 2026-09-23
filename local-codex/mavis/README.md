# Mavis implementation workspace

This directory turns `IMPLEMENTATION_PLAN.md` into an executable, evidence-led
system. `LEDGER.json` maps each delivery phase to its acceptance surface;
`contracts/` contains the six versioned machine contracts; `handoff/` contains
worker and verifier templates; and `baseline/` freezes the pre-change state.

Runtime data is stored outside Git under `~/.local-codex/mavis-service` and project `.mavis/`
directories. IRIS remains on its existing oMLX service. The Mavis service uses
an isolated base path and endpoint and refuses to replace a process it does not
own.

The first live dual-load canary rejected concurrent use of the 111.55 GB Qwen
profile: the Mavis load timed out and IRIS stopped answering until the Mavis
process was removed. Runtime admission now blocks any second local generation
model by default. A future retry requires a proven sequential handoff or a
materially smaller candidate; free memory alone is not treated as safety proof.

Run the quick gate with:

```sh
PYTHONPATH=local-codex/mavis python3 -m mavis eval e0
```

The command exits nonzero and records `reject` until all ten mandatory cases
have real receipts. Missing local-model or native-harness work is reported as
blocked rather than replaced with synthetic success.

## Inspect a command receipt

`local-codex output inspect RECEIPT.json` (or
`PYTHONPATH=local-codex/mavis python3 -m mavis output inspect RECEIPT.json` in
the source checkout)
checks an existing Mavis host receipt and its complete raw logs, then prints the
host verdict, exit status, timeout, failure and count lines, and raw output
reference. Known test formats use deterministic parsing. For irregular output,
`--model-id ID --model-path PATH` opts into exact-line observations from a
small model already loaded on Mavis's local port 8001. The model cannot change
the host verdict. If the full log exceeds the bounded model request, the
command returns an inconclusive model result and a nonzero exit while keeping
the verified host evidence visible. This source route does not claim that an
installed model transport has been exercised.

## Architecture boundary check

Run `PYTHONPATH=local-codex/mavis python3 -m mavis.architecture` from the fork
root when Mavis code or dependencies change. The command exits nonzero for a
provider SDK dependency/import, a provider catalog, or an HTTP transport outside
the three reviewed local oMLX adapters. It parses Python syntax and the project
manifest, so comments and documentation do not trigger findings. Its allowlist
is in `mavis/architecture.py`; changes to a local adapter should review both
that list and the adapter's loopback URL validation. This check is a source
guard; runtime endpoint and installed-service acceptance remain separate.

## Source-only experiment lifecycle

`mavis.experiments.ExperimentStore` keeps prompt, tool-setting, and retrieval
candidate configurations under the Mavis home. It freezes the active baseline,
accepts matched held-out results with retained output hashes, requires a
separate candidate and review gateway worker accepted by Terra for the exact
candidate report and comparison, stages a
candidate, and promotes it only between objectives. Rollback restores the
previous accepted configuration. An experiment can be queued through
`MaintenanceQueue`; its checkpoint records the experiment state and record hash
so paused work can resume without repeating an accepted step.
Profile activation requires that same promoted candidate and a fresh accepted
gateway review; old file-only promotion evidence is rejected.

This module is not wired into the installed profile loader or a live E1
evaluation runner yet. The source tests use injected gateway statuses and
synthetic evaluation files. Phase 3 still needs a real failure, held-out
comparison, accepted verifier job, staged runtime promotion, and rollback
canary before it can be called complete.
