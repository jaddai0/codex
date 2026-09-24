# Installed local librarian bridge — 2026-09-24

## Decision

The installed Mavis candidate at source revision
`eba4c2b814d7090766987c650f5081d7956ade9c` can call its project-bound
librarian from the ordinary `workspace-write` coding sandbox. A real agent
turn made a deterministic archive lookup and a pinned 4B model call, each with
host-verified citations, without a sandbox network override. This clears the
default-sandbox capability gap in the earlier completed-work canary. It does
not establish a net benefit to completed coding work or promote the 4B helper
as a default agent.

## Installed change and boundaries

`mavis.mcp_server` serves one stdio tool, `mavis_librarian_ask`. The installed
launcher supplies the absolute service home, Git project root, and resolved
4B model path as trusted MCP environment values. The tool exposes no project
path argument. It uses the existing librarian's archive verification and
loaded-model binding; it does not start a model, alter IRIS, or grant the coding
shell general loopback access. The generated registration approves this
specific local tool without interactive prompts. That approval mode was needed
because the first installed call was blocked by Codex's `never` approval
policy. Its failed log is retained in
`Mavis/evidence/phase2-mcp-installed/approval-blocked/`.

The installer now rejects an uncommitted checkout before copying or building,
checks the complete installed script set against its recorded commit, and
records its Python interpreter and MCP package version. The current host uses
Python MCP 1.16.0. The bridge still depends on that host-installed package;
the installer checks it but does not provision an isolated Python environment.

An earlier read-only Terra source review found a quoted-TOML bypass of the
disable guard and a source-revision provenance gap. Both were repaired before
installation. The guard parses preserved TOML and rejects an unmanaged
`mavis-memory` server even when its name is quoted. The focused profile and
MCP tests passed 16/16 and 2/2 after repair. The final Mavis source suite
passed 314 tests with two skips after installation; its complete output is
`Mavis/evidence/phase2-mcp-installed/final-source-suite.log`.

## Real installed canary

The fixture and raw records are under
`/Users/dustinpainter/Dev-Projects/Mavis/evidence/phase2-mcp-installed/`.
The project is a separate Git checkout with an intentionally dirty note and a
three-segment archived checkout decision. A non-default model directory
contains links to the installed main and pinned 4B candidates. The canary
launched `~/Desktop/Mavis.command exec --json -C <fixture> -s workspace-write`
without a network override while the shared GPU lease belonged to
`codex-mavis`. IRIS and ComfyUI remained standing tenants.

The agent completed two `mcp_tool_call` events on `mavis-memory`: `no_model`
returned the exact archived lines, and the model-backed call returned the
accepted tax, discount, shipping, rounding, and invalid-input rules with
`model_called: true`. The second result bound the helper to the pinned Qwen
4B snapshot. Both calls cited the original segment file and its SHA-256. The
turn ended normally without changing tracked project files or the pre-existing
user note. The helper wrote its expected short-lived context under ignored
`.mavis/`. The final raw log is `mavis.jsonl` (SHA-256
`632468d24b72c110192475cbee337fcc83f845e8498b54be180e1d4269b35376`).

`run-installed-canary.py` saved the exact generated registration as
`installed-config.toml` before any later Mavis run could replace it. The
host's session rollout recorded `workspace-write` with
`network_access: false`, `never` approvals, and restricted network permission;
`sandbox-receipt.json` binds that context to the completed thread and rollout
hash. `verify-installed-canary.py` independently re-read the saved config,
listed the installed MCP schema, validated each citation against its file,
compared model path and candidate fingerprints, and checked the terminal
events and cleanup. The final wrapper exited zero, and the independent verifier
passed and wrote `verification.json`. An earlier successful two-call run had a
wrapper exit 1 only because its assertion expected a clean Git diff even though
the fixture began with a dirty user note; it is retained under
`pre-config-snapshot/`. Another rerun saved a config snapshot but had a
wrapper variable-name error after both calls; it is retained under
`config-snapshot-harness-error/`. Neither earlier run is used as final
acceptance. In the final run the note's bytes were unchanged, and the only
tracked diff remained that pre-existing file.

The helper unloaded, Mavis's server parked, IRIS's model stayed loaded, and
`gpu-lease` returned free. One later E0 raw-output rerun was manually stopped
after an event-stream start/completion pair was misread as two commands. Its
cleanup failed while its own child was still exiting; the host confirmed no
Mavis listener or child remained and manually released `codex-mavis`. The next
strict raw-output run completed and passed.

## Installed E0 after package change

The prior E0 receipts were invalidated by this installation. Fresh installed
repair, raw-output, terminal-compaction, and server-recovery observers ran on
the new candidate. The small repair passed a separate native Terra review.
The first full E0 aggregation rejected old observations, and the second
rejected a new raw-output observation because the agent used a chained
preflight command. A constrained replay ran `produce_log.py` exactly once,
searched its complete raw output with one `rg` command, reported the failed
exit and exact buried marker, and passed the host gate. The final E0 run
`326665986826461ba7c2a74b8705789e` passed all 10 mandatory cases.
`~/.local-codex/mavis-service/evaluations/e0/final-installed-shared.json`
binds matching before/after installed fingerprints, IRIS loaded, Mavis
unloaded, and the GPU lease released.

## Remaining acceptance

The earlier paired coding trial had one incomplete baseline and used different
sandbox settings. Several new matched, completed trials with identical
sandboxes, full task-boundary timing, host checks, independent review, and
cloud-use receipts are needed before claiming that the helper improves work.
Automatic loading and orderly eviction of the helper are also unproven. The
broader Phase 2/3/4 plan remains open. The first separate installed Terra
review marked the bridge capability and E0 as evidenced but identified the
missing saved config and sandbox receipts, and an overbroad no-edits claim.
The final run and report address those findings;
`terra-installed-followup.txt` begins `ACCEPT` for the bridge capability and
E0 on this candidate. Its native Codex event stream, saved final message, and
review receipt are retained beside the canary evidence. That acceptance
explicitly excludes matched net benefit and whole-plan completion.
