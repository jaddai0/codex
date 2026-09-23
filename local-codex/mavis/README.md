# Mavis implementation workspace

This directory turns `IMPLEMENTATION_PLAN.md` into an executable, evidence-led
system. `LEDGER.json` maps each delivery phase to its acceptance surface;
`contracts/` contains the six versioned machine contracts; `handoff/` contains
worker and verifier templates; and `baseline/` freezes the pre-change state.

Runtime data is stored outside Git under `~/.local-codex/mavis-service` and project `.mavis/`
directories. IRIS remains on its existing oMLX service. The Mavis service uses
an isolated base path and endpoint and refuses to replace a process it does not
own.

## Optional project embeddings

`project-index search` stays exact by default. To use vectors, first load an
embedding model through the separately admitted Mavis runtime, then explicitly
refresh its index. The route checks that Mavis owns port 8001, the selected
embedding model is already loaded, its revision matches the oMLX model path,
and shared compute admission allows each request. It never starts a server or
loads a model. The revision can be the 40-character Hugging Face snapshot name.

```sh
PYTHONPATH=local-codex/mavis python3 -m mavis project-index --project /path/to/project \
  refresh-embeddings --embedding-model Qwen3-Embedding-0.6B \
  --embedding-revision SNAPSHOT_REVISION --embedding-dimensions 1024
PYTHONPATH=local-codex/mavis python3 -m mavis project-index --project /path/to/project \
  search 'describe the needed code' --embeddings \
  --embedding-model Qwen3-Embedding-0.6B \
  --embedding-revision SNAPSHOT_REVISION --embedding-dimensions 1024
```

`refresh-embeddings --rebuild` is required to change model, revision, index
version, or dimensions. `status` reports the bound identity and whether saved
passages still match live files. Search keeps current-file exact results and
returns `embedding_error` when the optional provider fails. A failed vector
refresh exits 2 and leaves prior vectors intact. The Qwen 0.6B route sends one
text per request because padded batches produced non-finite values in the
Phase 2 source comparison; installed service behavior still needs acceptance.

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
### Project evidence

When the Mavis launcher runs from a nested Git checkout, it records new project objectives,
host command receipts, and transcript handoffs under that checkout's ignored
`.mavis/` directory. `MAVIS_PROJECT_ROOT` can name the checkout explicitly for
service commands run elsewhere. The directory has a private `.gitignore` and
must pass `git check-ignore` before Mavis writes evidence. Project memory and
the search index already use the same directory. Shared verified coding
knowledge remains under `$MAVIS_HOME/knowledge`; service runtime, profiles,
and evaluation data remain under `$MAVIS_HOME`. The launcher's default
`~/Dev-Projects` directory resolves to the user's home Git repository and is
deliberately not treated as a project.

Older objectives and sessions under `$MAVIS_HOME` stay readable at their
original paths. Mavis selects an existing project record first and otherwise
reads the older service record. It does not rewrite old receipt paths or hashes.
To preserve a complete project copy of one older objective, run
`mavis project-evidence migrate-legacy-objective PROJECT OBJECTIVE_ID`. The
command copies its objective, host evidence, bound session records, and
transcript archive into `.mavis/legacy-import/objectives/OBJECTIVE_ID/`, then
writes a hash manifest. It leaves the source evidence intact so old acceptance
links continue to verify. A failed copy does not publish a final import. To
roll back new routing, remove `MAVIS_PROJECT_ROOT` from a direct service
invocation; old global records remain in place. Never delete the original
service evidence before all linked acceptance and retention records have been
checked separately.
