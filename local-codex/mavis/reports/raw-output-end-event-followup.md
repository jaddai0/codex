# Mavis raw-output end-event follow-up

Date: 2026-09-22. Branch: `codex/mavis-raw-output`. This is a source-only follow-up to the post-exit drain repair. The installed runtime and services were not changed.

## Finding

The first repair made `exec_command` and `write_stdin` wait for the raw-output reader after exit, but the separate background stream still stopped after the ordinary 100 ms grace. Its command-end event could report a normal exit with a partial transcript before the raw capture finished or timed out. This was found by an independent read-only review at `/Users/dustinpainter/.Codex/reports/local-codex-5f19f0e24-raw-drain-review.md`.

## Repair

For Mavis raw capture, the background stream now remains active until the reader closes. It uses the same 30-second post-exit limit as the tool response. If the reader remains open at that limit, the stream marks the process failed and terminates it. The end watcher checks the reader outcome after the stream finishes, so it cannot report a normal command end while raw capture is incomplete. Ordinary Codex retains the 100 ms grace.

## Verification

- `just test -p codex-core mavis_`: 8 passed. A controlled late tail arrives 150 ms after exit and appears in the completed event transcript and complete raw file. A held output pipe reaches the 30-second limit, produces a failed event, and leaves the raw reference incomplete. The existing 1.6 MB exec/stdin burst test also passes.
- `ZDOTDIR=$(mktemp -d) just test -p codex-core unified_exec`: 179 passed, including ordinary Codex event and timeout paths.
- `just fmt`, `git diff --check`, and the follow-up closeout ledger check passed.

## Limit

This verifies source behavior in the isolated checkout. The installed binary still needs to be built and rerun against the buried-log E0 case before live acceptance can be claimed.
