# Mavis raw unified exec output capture

Date: 2026-09-22
Branch: `codex/mavis-raw-output`
Base: `80b752465c3882ce821258ebf5513804a0d8608e`
Worktree: `/Users/dustinpainter/Dev-Projects/local-codex-mavis-raw-output`

## Scope and result

The `local-codex` launcher now sets `MAVIS_RAW_OUTPUT_REQUIRED=1`. Core opens a unique private file under `$MAVIS_HOME/tool-output/` for each unified exec process only when that marker is exactly `1`. The directory is mode `0700`; each file is created at mode `0600` on Unix. Core appends the original byte chunks before the 1 MiB `HeadTailBuffer` drops middle bytes. Each tool response syncs the file and includes its absolute path, byte count, and whether the process output stream has closed. Function-call transcript output carries this in the header; code-mode tool output carries `raw_output_ref` as structured JSON. A running process keeps the same file path across `write_stdin` calls.

For native local PTY output, the Mavis path replaces the intermediate lossy broadcast combiner with a bounded channel that waits for its reader. The normal Codex path retains its existing combiner and response formatting. For exec-server output, core records each received original chunk before the cap. If the event channel lags or sequence recovery reveals missing output, the process fails and the reference cannot be marked complete. A file append or sync failure also fails the Mavis tool call.

## Evidence

- `cargo test -p codex-core --lib raw_output --no-default-features`: 3 passed. The two added tests place `FAILURE: buried diagnostic` between 700 KB of head and tail material. The 1 MiB buffer drops that phrase, while the private file retains it; the local output task test exercises the actual Mavis reader path.
- `cargo test -p codex-core --lib exec_command_tool_output_ --no-default-features`: 4 passed. The new test checks the exact transcript reference and structured code-mode reference; existing formatting tests show responses without a reference are unchanged.
- `just fmt` and a final `cargo fmt --all --quiet` completed. `git diff --check` passed. `zsh -n local-codex/bin/local-codex` passed.
- No live Mavis or IRIS model call, service modification, install, or full Rust workspace suite was run.

## File map

- `local-codex/bin/local-codex`: Mavis-only marker.
- `codex-rs/core/src/unified_exec/raw_output_spool.rs`: private spool, durable sync, reference, cap/permission test.
- `codex-rs/core/src/unified_exec/process.rs`: output reader capture before cap, Mavis backpressure, gap/failure handling, reader integration test.
- `codex-rs/core/src/unified_exec/process_manager.rs`: reference on `exec_command` and `write_stdin` results.
- `codex-rs/core/src/tools/context.rs`: transcript header and code-mode JSON reference.
- Other changed test constructors: initialize the optional reference to `None`.

## Limits and follow-up

- This branch changes the core binary. It requires a new build and install before a live Mavis session can use it. The source tests do not prove installed runtime behavior.
- Exec-server retains only a bounded replay window. The new failure checks prevent a known gap from being labeled complete, but a large burst can fail capture there. A future exec-server transport spool or backpressured stream is needed for a full guarantee on remote or exec-server-backed sessions.
- This stores raw output files and supplies stable transcript references. Mavis evidence ingestion can resolve the paths later; this slice does not index or prune them. These files may contain sensitive command output, so retention policy remains a separate requirement.
- Sandbox-denial conversion and spawn-time errors may return an error before a normal `ExecCommandToolOutput` reference is emitted. The raw file remains, but transcript linkage for those error paths is not covered by this slice.

## Closeout status

`work-closeout check /tmp/mavis-raw-output-ledger.json --stage closeout` returned `FAIL`: this slice is **partial** against universal exec-server capture because that transport retains only 1 MiB of replay. The local Mavis PTY path and transcript reference are verified. Existing `unified_exec_persists_across_requests` and `unified_exec_uses_remote_exec_server_when_configured` tests each passed after the change; these prove compatibility, not lossless remote burst capture.
