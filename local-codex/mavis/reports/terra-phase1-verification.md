# Terra independent verification: Mavis Phase 0 / Phase 1

**Date:** 2026-09-22  
**Mode:** read-only audit  
**Verifier:** Terra / Codex, independent of the implementation workers  
**Scope:** `/Users/dustinpainter/Dev-Projects/local-codex` at `cb75ad1e8` and `/Users/dustinpainter/Dev-Projects/ai-skills-dev-mavis-gateway` on `codex/mavis-gateway-phase1`.

## Verdict

**REJECT — neither Phase 0 nor Phase 1 is accepted.**

The source-level units are healthy, and the Mavis oMLX process is separate from
IRIS. That is useful progress, not a completed milestone. Phase 0 still lacks
the required real MiniMax and Terra harness canaries. Phase 1 has no E0 runner,
no independently verified real maintenance or multi-file task, no installed
Desktop launcher at the standard required path, and an evidence gate that can
accept fabricated minimal records. The live Mavis load remains **IN PROGRESS**,
not passed or failed.

## What was independently checked

- The local-codex checkout was clean on `omlx-only` at `cb75ad1e8`.
- The gateway worktree had exactly its declared Phase 1 files modified or
  untracked; `git diff --check` passed. No unrelated worktree edits were found.
- Mavis units passed: `19` tests. The source launcher also passed `zsh -n` and
  `python3 -m mavis --help`.
- Gateway targeted tests passed: `87`; its complete plugin test directory
  passed: `104`.
- Focused test files contain no skipped or expected-failure tests. The gateway
  uses test doubles for FastMCP, provider generation, and tmux, so these are
  deterministic code checks rather than installed gateway or subscription proof.
- I made no code, configuration, process, service, GPU, IRIS, commit, or push
  changes. This report is the sole verifier artifact created.

## Runtime observation: in progress, left undisturbed

At inspection time, separate loopback listeners were present:

| Service | PID | Port | Observation |
| --- | ---: | ---: | --- |
| IRIS oMLX | 63526 | 8000 | Existing listener; untouched. |
| Mavis oMLX | 30090 | 8001 | Runtime state records the Mavis base path and this PID; the process has Mavis-local log files open. |

The Mavis log records startup at `14:33` and begins loading
`Qwen3.8-Flash-Next-Abliterated-MLX-4bit` at `14:38`. It does not yet contain a
completion, model-ready, prompt, or evaluation receipt. I did not send a health
or generation request because the stated live canary may still be loading.

This supports the narrow claim that the server launch is isolated from IRIS. It
does not prove model admission, model readiness, launcher acceptance, cache and
history separation under use, or E0 success.

## Blocking findings

### 1. The recorded E0 acceptance command does not exist

`LEDGER.json` and `mavis/AGENTS.md` name `python3 -m mavis eval e0` as the
runtime gate. The actual CLI offers only `runtime`, `objective`, `run`, and
`archive-search`; invoking the recorded command exits with argparse's
`invalid choice: 'eval'` error. There is no evaluation directory or receipt in
the checkout.

This blocks every E0 requirement: launch/model/tool round-trip; patch/test
task; dirty-work preservation; seeded repair; false-success rejection; large
output preservation; compaction/restart recovery; escalation; external-harness
completion; and unavailable-service recovery.

### 2. Objective acceptance can be fabricated

The implementation keeps JSON schemas, but does not validate stored assignment,
receipt, or verifier data against them. In particular:

- `ObjectiveStore.add_receipt` only checks `schema_version`.
- `ObjectiveStore.add_verification` only checks that the claimed provider name
  differs from an assignment's claimed provider name.
- `_assert_acceptance` only needs a receipt whose `verdict` is `pass` and a
  final verifier dictionary whose `verdict` is `accepted`.

The existing passing test deliberately demonstrates this weak path: it writes a
receipt containing only `schema_version`, `verdict`, and `changed_revision`,
then accepts an equally minimal verifier dictionary. It omits the contract's
required command, timestamps, raw-output reference, artifact hashes, verifier
identity, protected fixtures, and checks. A worker can therefore fabricate a
passing receipt and acceptance without host-recorded evidence.

This violates the plan's central rule that workers cannot certify themselves
and that evidence must contain command, exit status, timestamps, raw output,
revision, artifacts, and an independent verdict. Treat this as a release-
blocking evidence-integrity defect, not a missing test detail.

### 3. The required Phase 0 harness canaries are not all present

The frozen baseline labels GLM/ZCode `accepted` with only the string
`MAVIS_GLM_CANARY_OK`; no durable receipt or raw output was found. It labels
MiniMax/OpenCode `blocked` for missing authentication and Terra/Codex `queued`.
The current verifier run is not a native Terra harness canary.

Phase 0's explicit success condition is that **each** required external harness
performs a small real isolated-fixture task. It is therefore rejected even if
the reported GLM string is accurate.

### 4. The installed Desktop-launcher requirement is unproven and currently absent at its standard path

The source `Mavis.command` exists, and the source installer would copy it to
`~/Desktop/Mavis.command`. At inspection, `/Users/dustinpainter/Desktop/Mavis.command`
did not exist. `/Users/dustinpainter/.local/bin/mavis` does exist, but I did not
run it because it would interact with the in-progress model load.

The source installer also has not supplied installed-app evidence for the exact
candidate. The plan requires the Desktop launcher to start/reconnect to Mavis's
own services and verify identities without affecting IRIS. That acceptance
remains open.

### 5. Native delegation is a tested job primitive, not a completed routing integration

The gateway adds a tmux job runner and exposes lane names `minimax`, `zcode`,
and `terra`. It safely fails tool-using work when those literal lane names are
sent to `delegate_task`, but the normal GLM provider is named `zhipu`, not
`zcode`; normal routing returns `zhipu`. There is no mapping from a routed GLM
assignment to a constructed ZCode command, no assignment-to-job record, and no
objective receipt created from a completed job.

The implementation report acknowledges that callers must construct harness
commands themselves. That is a useful lower-level component, but it does not
meet the plan's delivery of native MiniMax, GLM, and Terra delegation with
structured assignments, dedicated worktrees, and durable completion evidence.

There is an additional job-identity weakness: distinct supplied IDs can
normalize to the same `safe_session_name` and job directory (for example,
punctuation variants that normalize to the same slug). The runner has no
collision check against the original job ID before reusing that directory.

### 6. Jev is not an operable Phase 1 decision integration

The spend ledger is append-only and its tested unknown-cost/corrupt-ledger gate
fails closed. The Jev adapter is advisory and correctly defaults off. However,
there is no registered `jev` provider: the gateway's own valid provider list
does not include it. The Phase 1 test that calls `delegate_task(provider_id="jev")`
passes only because it replaces `provider_generate` with a fake success.

The implementation report also states that the actual Jev provider adapter is
not registered. A cap gate ahead of a later real-provider failure is not Jev
decision integration. This is a missing requirement, not a successful live
path.

### 7. The remaining Phase 1 orchestration is not connected to the real harness

The service has individual code for profiles, objective state, command output,
and transcript files, but it is not wired into Codex lifecycle or compaction.
The CLI cannot add assignments or verifications, cannot create profile versions
or select an active profile, and has no evaluation command. A profile can be
activated with arbitrary strings for experiment and verifier receipt; their
existence or independent acceptance is not checked. Transcript hashes are
written but not rechecked during search.

Consequently, source methods exist for parts of raw output capture, archive
search, and interruption classification, but there is no proof that Mavis
captures real pre-truncation tool output, preserves a real Codex transcript
before compaction, restores an interrupted objective, enforces two distinct
repair attempts, or blocks an unjustified profile promotion.

## Requirement ledger

| Requirement | Independent status | Evidence / reason |
| --- | --- | --- |
| Phase 0 contracts, ownership map, templates, baseline | PARTIAL | Artifacts are committed and source tests pass, but contract enforcement is absent and harness evidence is incomplete. |
| Phase 0 real MiniMax, GLM, Terra isolated-fixture canaries | REJECT | MiniMax blocked, Terra queued, GLM has only an uncoupled status string. |
| Isolated server | PARTIAL / runtime in progress | Separate 8001 process and Mavis-local paths observed; no completed model/admission or service canary. |
| Desktop launcher | REJECT | Standard Desktop path absent; installed candidate was not safely runnable during the live load. |
| Versioned model profiles and selection | REJECT | Store exists but arbitrary promotion is allowed and it is not connected to launcher selection. |
| Objective, evidence, bounded repair, independent verification | REJECT | State storage exists, but required evidence and independent verifier records can be forged; repair logic does not compare new evidence. |
| Full tool-output capture, transcripts, compaction handoff, archive search | PARTIAL | Standalone local helpers pass units; no Codex hook or actual compaction receipt. |
| Native MiniMax, GLM, Terra delegation | PARTIAL | Tmux runner and fail-closed primitives pass units; route-to-harness integration, assignment receipts, and real canaries are absent. |
| Jev decision integration and $10 cap | PARTIAL | Ledger and advisory adapter pass units; no registered Jev provider or live decision path. |
| Deterministic parsing and interruption recovery | PARTIAL | Small helper tests pass; no E0/real interrupted-objective evidence. |
| E0 plus independently verified maintenance and multi-file tasks | REJECT | E0 command is absent and no real-task receipts or independent acceptance artifacts exist. |
| Preserve unrelated dirty work | NOT DEMONSTRATED | Gateway change set is narrowly scoped, but the required real-task proof is absent. |

## Required path to a future acceptance attempt

1. Implement the real E0 runner and receipts before rerunning any acceptance
   claim.
2. Make schema validation mandatory at every write and acceptance boundary;
   verify receipts are host-produced, exist, hash-match, cover required checks,
   and bind an independent verifier to the exact revision and assignment.
3. Wire profiles, objective assignments, raw tool capture, compaction handoffs,
   job recovery, and harness jobs into the actual Mavis/Codex path.
4. Add safe, collision-resistant job identifiers and map routed MiniMax/GLM/
   Terra work to native harness command builders and worktree ownership.
5. Register a real Jev adapter or explicitly defer that requirement; do not
   use a fake provider-generation success as evidence of dispatch.
6. Resolve OpenCode authentication, run all three real isolated-fixture
   canaries, install the exact Desktop launcher candidate, and wait for the
   currently loading model canary to finish naturally.
7. Only then run E0, one bounded maintenance task, and one multi-file task;
   retain host receipts and submit them to a fresh independent verifier.

## Commands run

```text
git status --short --branch; git log --oneline
git diff --check; git diff --name-only; git ls-files --others --exclude-standard
PYTHONPATH=. python3 -m unittest discover -s tests -v       # 19 passed
zsh -n Mavis.command
PYTHONPATH=. python3 -m mavis --help
PYTHONPATH=. python3 -m mavis eval e0                       # invalid command
python3 -m pytest ...test_harness_runner.py ...test_spend_ledger.py ...test_job_runner.py ...test_subscription_ledger.py ...test_gateway_tools.py -q  # 87 passed
python3 -m pytest marketplace/plugins/model-gateway/tests -q  # 104 passed
lsof / ps / log inspection for ports 8000 and 8001 (read-only)
```

No GPU model was launched by this verification. No process or service was
stopped, and no request was sent to the live Mavis or IRIS servers.
