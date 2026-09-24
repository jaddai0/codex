# Phase 2 completed-work helper canary — 2026-09-24

## Decision

The installed 4B librarian succeeded inside one coding task and the native Mavis
turn completed. This is a capability signal, **not** a measured net benefit or
permission to promote the helper. Phase 2 remains open. The comparable baseline
produced correct code and passed host tests but did not finish its own turn;
the corrected helper arm used a network sandbox override that the baseline did
not record. One sequential pair does not establish a latency or reliability
distribution.

All fixtures, programs, raw logs, host checks, independent reviews, and the
constraint record live under
`/Users/dustinpainter/Dev-Projects/Mavis/evidence/phase2-benefit/`. No extra
top-level Mavis checkout was created.

## Frozen task and acceptance

`fixture.json` records identical initial Git revision
`8cd85bbb674818f00549dca1cd6e0ba5b358659f` and protected dirty-note
SHA-256 for both arms. Each arm had the same early accepted checkout decision
in a three-segment project archive. The agent had to implement `checkout.total`,
run the visible test, preserve the note, and finish. The host-only
`verify_checkout.py` checks the tax/discount/shipping rule, end-only
`ROUND_HALF_UP`, invalid inputs, scope, revision, and note hash in five tests.
The initial stub failed acceptance; `baseline-before.log` is retained.

Both arms used the same installed candidate from source revision `7ad62299b`
with shell launcher SHA-256
`7d0ec70042ba81938b11a7a4295808c73e09c459705cfb8a56c097a17d7ed733`.
The paired result JSON binds its before/after installed fingerprints. IRIS's
Qwen model stayed loaded according to the trial wrapper, Mavis unloaded after
each arm, and `gpu-lease status` returned free. ComfyUI's standing server was
left in place. The wrapper's continuity fields are historical attestations;
the independent comparison did not sample IRIS throughout every generation.

## What ran

| Arm | Native Mavis result | Host result | Task time | Helper load |
|---|---|---|---:|---:|
| Baseline, direct archive search | Interrupted after repeated false claims that visible tool output was hidden | 5/5 pass | 253.763 s before interruption | — |
| 4B librarian, corrected sandbox invocation | Completed and reported the actual visible test result | 5/5 pass | 89.193 s | 2.508 s |

The baseline's raw JSONL contains actual successful command output, the source
edit, and the visible passing test. The main model then re-read the same files
and repeatedly claimed that output was unavailable. It did not emit
`turn.completed`; the wrapper stopped its own child and released the GPU lease.
This is a real completion failure despite correct code. `baseline-result.json`,
`baseline-mavis.jsonl`, and `baseline-host.log` preserve the distinct facts.

The first helper arm was stopped and retained under `sandbox-denied/`: the
installed `mavis librarian ask` command failed with `helper binding refused:
<urlopen error [Errno 1] Operation not permitted>` inside the default
`workspace-write` sandbox. That was a sandbox invocation failure, not an
answer from the 4B model. The task-owned fixture was rebuilt at the same Git
revision and note hash. The corrected arm added
`sandbox_workspace_write.network_access=true` only to its installed core
invocation. Its actual shell tool returned `model_called: true`, the pinned
Qwen3.5-4B snapshot path, the complete accepted rule, and a cited archive
line whose file SHA-256 matches. The agent then edited `checkout.py`, ran the
visible test successfully, and emitted `turn.completed`. The independent host
checker passed all five requirements. Evidence: `candidate-mavis.jsonl`,
`candidate-result.json`, and `candidate-host.log`.

The sandbox defect is recorded under the broken-tool gate: valid command
failure, direct distinction from model failure, corrected invocation, and a
successful actual model call with a verified citation. Normal Mavis sessions
still need a supported way to reach the local helper without granting broad
network access; the corrected arm proves only the explicit override.

## Independent comparison and limits

`compare.py` reruns host acceptance; checks the actual test exit embedded in
each JSONL transcript; verifies the current installed core and shell hashes,
starting revisions, note hashes, changed-file scope, raw-log hashes, 4B model
path, and cited archive file bytes; and writes `comparison.json`. The
candidate has one accepted completed task; the baseline has zero because its
native turn did not finish. The comparison explicitly rejects promotion.

Native Terra's first review in `terra-review.txt` accepted the loopback-enabled
4B canary and rejected a net-benefit claim. It identified weaknesses in the
first comparison checker. A second native review in
`terra-comparison-v2-review.txt` found the revised checker's
`capability_signal_only` decision supported, then identified two remaining
binding checks; those were added to `compare.py` and rerun. Both reviews are
read-only, with raw event logs beside them. The final comparison remains a
single-pair canary, not a promotion receipt.

The recorded task times exclude main-model loading and cleanup. The helper
load time is separate. The baseline was interrupted, sandbox conditions were
different, and cloud billing was not independently reconciled; no time saving
or zero-cloud-spend claim follows. `constraint_record.json` marks the observed
constraint as agent continuation, the model-quality parity gate as failed for
promotion, and the promotion verdict inconclusive. Its identify, act,
promote-record, and trial-closeout validators pass. These validators check the
experiment record; they do not mark Phase 2 complete.

## Next acceptance

Provide a supported local-helper route in Mavis's default coding sandbox,
canary it through the installed launcher, and run several matched completed
tasks with the same sandbox settings, full task-boundary timing, host checks,
independent review, and cloud usage receipts before enabling the 4B helper by
default. The false tool-output loop is retained as a Phase 3 failure case.
