# Mavis raw unified exec output capture

Date: 2026-09-22
Branch: `codex/mavis-raw-output`
Base: `80b752465c3882ce821258ebf5513804a0d8608e`
Worktree: `/Users/dustinpainter/Dev-Projects/local-codex-mavis-raw-output`

## Scope and result

The `local-codex` launcher now sets `MAVIS_RAW_OUTPUT_REQUIRED=1`. Core opens a unique private file under `$MAVIS_HOME/tool-output/` for each unified exec process only when that marker is exactly `1`. The directory is mode `0700`; each file is created at mode `0600` on Unix. Core appends the original byte chunks before the 1 MiB `HeadTailBuffer` drops middle bytes. Each tool response syncs the file and includes its absolute path, byte count, and whether the process output stream has closed. Function-call transcript output carries this in the header; code-mode tool output carries `raw_output_ref` as structured JSON. A running process keeps the same file path across `write_stdin` calls.

For native local PTY output, the Mavis path replaces the intermediate lossy broadcast combiner with a bounded channel that waits for its reader. The normal Codex path retains its existing combiner and response formatting. For exec-server output, core sends an explicit Mavis requirement on the process request. The executor opens a private raw file and process-ID manifest before it spawns the command, then writes and syncs each chunk before its bounded replay may evict it. The start response confirms this storage; Mavis core terminates an executor that does not confirm it, including an older peer. Core still records received chunks, and fails rather than reporting a complete core file if its event stream loses a chunk. Server-side output remains in the source file in that case. A file append or sync failure fails the Mavis process.

## Evidence

- `cargo test -p codex-core --lib raw_output --no-default-features`: 3 passed. The two added tests place `FAILURE: buried diagnostic` between 700 KB of head and tail material. The 1 MiB buffer drops that phrase, while the private file retains it; the local output task test exercises the actual Mavis reader path.
- `cargo test -p codex-core --lib exec_command_tool_output_ --no-default-features`: 4 passed. The new test checks the exact transcript reference and structured code-mode reference; existing formatting tests show responses without a reference are unchanged.
- `cargo test -p codex-exec-server --lib mavis_ --no-default-features`: 2 passed. A real executor start confirmed private source storage before launching the command. A separate 1.9 MiB burst evicted a buried diagnostic from replay, while the synced private source file retained it and its manifest linked the file to the process ID.
- `cargo test -p codex-core --lib mavis_remote_path_requires_executor_spool_acknowledgement --no-default-features`: 1 passed. Mavis rejects an unconfirmed executor; ordinary Codex accepts it; a confirmed executor passes.
- `cargo test -p codex-exec-server-protocol --lib --no-default-features`: 25 passed, including old response compatibility. Existing bounded replay test passed.
- `just fmt`, `git diff --check`, and `zsh -n local-codex/bin/local-codex` passed.
- No live Mavis or IRIS model call, service modification, install, or full Rust workspace suite was run.

## File map

- `local-codex/bin/local-codex`: Mavis-only marker.
- `codex-rs/core/src/unified_exec/raw_output_spool.rs`: private spool, durable sync, reference, cap/permission test.
- `codex-rs/core/src/unified_exec/process.rs`: output reader capture before cap, Mavis backpressure, gap/failure handling, reader integration test.
- `codex-rs/core/src/unified_exec/process_manager.rs`: reference on `exec_command` and `write_stdin` results.
- `codex-rs/core/src/tools/context.rs`: transcript header and code-mode JSON reference.
- `codex-rs/exec-server-protocol/src/protocol.rs`: request requirement and start acknowledgement, both backward compatible when absent.
- `codex-rs/exec-server/src/mavis_output_spool.rs` and `local_process.rs`: private source spool, manifest, write-before-replay, fail-on-write-error, and >1 MiB burst test.
- `codex-rs/exec-server/src/client.rs`, `remote_process.rs`, `process.rs`: carry executor acknowledgement to core.
- Other changed test constructors: initialize the optional reference or acknowledgement to its ordinary value.

## Limits and follow-up

- This branch changes the core binary. It requires a new build and install before a live Mavis session can use it. The source tests do not prove installed runtime behavior.
- Exec-server still retains only a bounded replay window for transport. A Mavis process is accepted only when executor-side durable source storage is confirmed. If a transport burst creates a gap, core fails the command rather than presenting a complete core file; the executor source file and process-ID manifest preserve the full bytes. On an external executor, this file lives on that host and may require its file service to retrieve it. If that host lacks `MAVIS_HOME`, process start fails before running the command.
- This stores raw output files and supplies stable transcript references. Mavis evidence ingestion can resolve the paths later; this slice does not index or prune them. These files may contain sensitive command output, so retention policy remains a separate requirement.
- Sandbox-denial conversion and spawn-time errors may return an error before a normal `ExecCommandToolOutput` reference is emitted. The raw file remains, but transcript linkage for those error paths is not covered by this slice.

## Closeout status

The first closeout audit returned `FAIL` because exec-server source bytes were not yet durable. This revision adds source storage before replay, a burst test, and an explicit executor acknowledgement. Source-level capture is verified. Installed runtime acceptance and any external executor file retrieval remain separate checks for the parent release lane. Existing `unified_exec_persists_across_requests` and `unified_exec_uses_remote_exec_server_when_configured` tests passed; they prove routing compatibility, not installed capture.

The source-slice ledger passed: `work-closeout check /tmp/mavis-raw-output-ledger.json --stage closeout`. `cargo check -p codex-exec-server --tests --no-default-features` also passed, compiling the updated integration test constructors. This closeout covers code and focused tests; the installed binary remains a separate acceptance step.
