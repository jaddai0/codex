# Phase 2 librarian integration — 2026-09-24

Status: source milestone implemented; installed and completed-task benefit gates remain open.
Owner: MiniMax M3 via OpenCode; Sol repaired review findings after two denied out-of-scope `/tmp` tool calls stopped its follow-up runs.

## Behavior

`mavis librarian ask` requires `--project` or `MAVIS_PROJECT_ROOT`, creates and checks a Git-ignored project `.mavis`, then searches that project's conversation archive using explicit terms and paging. It does not silently read older shared-home archives. `--no-model` returns verified lines without an interpretation. Model mode binds the exact loaded ID/path to Mavis's local port 8001 before generation. The host checks every citation against current archive bytes. Follow-up context stays in the project and expires after 60 seconds. No model is loaded implicitly.

## Review finding and repair

The worker's first CLI called the answer route without binding the model path. With a valid archived line and stubbed completion, `--model-id fake-model --model-path /nonexistent/model` returned exit 0, `status: pass`, and no binding record. Independent reproduction is in the task transcript and the MiniMax event log at `~/.Codex/reports/mavis-phase2-librarian-20260924.jsonl`.

The repaired route binds the service home and model identity before completion. A second independent run proved the answer could no longer pass, but found an uncaught `FileNotFoundError` traceback. Sol's review repair wraps expected binding and service failures into a JSON `status: fail` response with exit 1. Empty evidence blocks model generation. Malformed citations are checked by the host validator before any follow-up reuse. The focused CLI test exercises the real binding call with a missing path and an unavailable inventory; neither reaches completion.

Terra's first review rejected shared-home fallback and validation outside the JSON handler. Its second found an ignored invalid `--project` and an expiry-worker traceback. The third found that citation line `1.0` could pass as integer line `1`; the host now requires an actual positive integer. The rejected reviews remain under `Mavis/evidence/phase2-librarian-terra-review*20260924.md`.

## Checks

| Check | Result |
| --- | --- |
| MiniMax source suite before Sol review repair | 336 tests passed, 2 skipped |
| Final focused helper and route suites | 24 tests passed |
| Final full source suite | 311 tests passed, 2 skipped; `Mavis/evidence/phase2-librarian-final-source-tests-v4.log` |
| Final compileall | Exit 0; `Mavis/evidence/phase2-librarian-compileall-v4.json` |
| Installed CLI and live helper on port 8001 | Pending |
| Net benefit to completed coding work | Pending |
| Independent Terra review | Three reviews rejected prior versions; final review pending |

The new route, tests, and CLI change are under `local-codex/mavis/`. The prior installed candidate and its E0/E2 receipts do not apply to this changed source. Static source shows this route has no model-server lifecycle operation; live IRIS isolation remains an installed acceptance gate.

## Evidence and limits

The OpenCode worker event logs are `~/.Codex/reports/mavis-phase2-librarian-20260924.jsonl` and `~/.Codex/reports/mavis-phase2-librarian-repair-cont-20260924.jsonl`. They include the failed first tests, repair, and worker-reported full-suite output. The later `/tmp` permission denials are recorded in `~/.Codex/reports/mavis-phase2-librarian-repair-20260924.jsonl` and `~/.Codex/reports/mavis-phase2-librarian-final-repair-20260924.jsonl`; those runs made no accepted code changes. The test suite uses a stub for the local model; it does not establish live helper quality or speed.
