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

## Independent review repair, 2026-09-22

Terra rejected the preceding revision. This follow-up addresses its six findings:

1. **Terminal-only replay gap (P0):** Core now checks the executor's `next_seq` against every replayed output sequence and only allows sequence positions for terminal events it has not already seen. If output was evicted and replay contains only `Exited` and `Closed`, core records a failure and cannot mark its partial local file complete. A focused regression test covers that exact case and a prior `Exited` event.
2. **Spawn before controller spool (P1):** The process manager opens the Mavis controller spool before either a local child or remote executor starts. The process constructors receive that already-open spool. Failure to open the controller file therefore stops command launch. This also removes the remote-command orphan path caused by a post-acknowledgement controller open failure. Missing executor acknowledgement still requests remote termination.
3. **Sync before display (P1):** Core now calls `sync_data` for each local append before it updates the display buffer or sends the output chunk, matching executor-side behavior.
4. **Windows privacy (P1):** Mavis raw capture now fails closed on Windows. Unix mode `0700`/`0600` is not a Windows ACL and this branch does not implement an equivalent. Ordinary Codex remains unaffected. The installed Mavis lane is macOS.
5. **Stable paths (P2):** Both spools reject a relative `MAVIS_HOME` and canonicalize the created directory before producing a reference or manifest. The launcher rejects a relative override before runtime admission or model loading.

Verification after repair:

- `just test -p codex-core terminal_only_replay_cannot_hide_evicted_output`: 1 passed.
- `just test -p codex-core mavis_spool_failure_prevents_command_launch`: 1 passed. With an invalid Mavis home, the unified exec command returns an error and leaves its marker file absent.
- `just test -p codex-exec-server mavis`: 3 passed, including the source-side burst test.
- `ZDOTDIR=<empty temporary directory> just test -p codex-core unified_exec`: 175 passed.
- `just fmt`, `git diff --check`, and `zsh -n local-codex/bin/local-codex`: passed. The formatter also touched unrelated Python files; those formatting-only changes were discarded before commit.
- An initial run of the 175 core tests had one unrelated shell-snapshot failure because the user's `.zshenv` invokes `grep` and `cut` under a test that deliberately restricts `PATH`. Repeating that test and the full focused set with an empty temporary `ZDOTDIR` passed.
- The complete `codex-exec-server` crate test run had unrelated parallel registration-retry failures; one representative failure passed alone. The Mavis-targeted executor tests passed. No full workspace suite, live service call, model load, or install was run.

This is source-level repair evidence. The parent release lane still needs a new core binary build and installed macOS acceptance. A lost controller replay must be treated as a failed command even though the executor-side source file is durable. Windows Mavis execution remains explicitly unsupported until private ACL storage is implemented.
