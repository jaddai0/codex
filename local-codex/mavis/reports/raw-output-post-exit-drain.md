# Mavis post-exit raw-output drain repair

Date: 2026-09-22. Branch: `codex/mavis-raw-output`. Worktree: `/Users/dustinpainter/Dev-Projects/local-codex-mavis-raw-output`. This report covers a source repair only; it has not been installed.

## Live defect and cause

The installed E0 case `buried-live-098ed22159d944b89438b100f7844802` ran `produce_log.py`, which prints 800,000 `A` characters, a random `MAVIS_E0_FAILURE` marker, then 800,000 `Z` characters and exits 1. Its core reference reported 647,168 bytes as still streaming while also reporting exit code 1. The raw file later stopped at 980,224 bytes, short of the roughly 1.6 MB written by the script. This evidence was read only; the installed service and model were untouched.

The local output reader writes and syncs each received chunk. The collector in `process_manager.rs` could stop 50 ms after exit even while that reader was still receiving bytes. Both initial `exec_command` and `write_stdin` then treated the process as exited, released it, and built a reference. Releasing the last process handle aborts the reader in `UnifiedExecProcess::Drop`. The reported reference therefore described a file that could never receive the missing tail.

## Repair

- Mavis raw capture now waits for its output reader to close after the command exits. It allows up to 30 seconds for pipes held by a descendant, then fails the command and terminates/releases the process. It never reports that stalled capture as complete.
- Both initial `exec_command` and `write_stdin` finish that wait before releasing an exited process or returning its reference. Bytes arriving after the ordinary 50 ms collection cap are folded into the response's bounded head/tail summary and byte count.
- Ordinary Codex still uses the 50 ms post-exit collection cap and its existing response behavior. A running Mavis process still yields normally with an incomplete reference and a process ID; the drain wait applies only once exit is observed.

## Verification

- `just test -p codex-core mavis_exited_burst_finishes_raw_capture_for_exec_and_stdin`: passed. It drives the production manager through initial exec and an interactive `write_stdin` exit, each with an 800,000 A + marker + 800,000 Z burst. The initial raw file exactly equals the expected bytes. The interactive raw file contains the full burst after PTY line-ending normalization. Both references report `complete=true` and byte counts equal to their files.
- `ZDOTDIR=<empty temporary directory> just test -p codex-core unified_exec`: 177 passed on the final test run. The empty temporary Zsh startup directory avoids an unrelated user `.zshenv` dependency in a PATH-restricted shell-snapshot test. The first broad run passed after one retry of the new burst test because it assumed a busy interactive command would exit within a single poll; the test now polls a still-running session and the final run passed without retry.
- `just fmt`, `git diff --check`, and the closeout ledger check passed. No full workspace suite, live model call, install, or service mutation was performed.

## Limits

The installed binary named in the E0 evidence predates this repair. A new core build and installed rerun of the same buried-log case are still required before claiming live acceptance. The repair makes incomplete capture fail closed after a 30-second post-exit limit; it does not make a descendant that holds the output pipe open forever into a successful command. A separate exec-server source spool continues to preserve remote bytes, with the earlier replay-gap failure gate still in place.
