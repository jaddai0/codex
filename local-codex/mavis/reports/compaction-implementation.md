# Mavis Phase 1 compaction slice — implementation evidence

Date: 2026-09-22. Branch/worktree: `codex/mavis-core-hooks` at `/Users/dustinpainter/Dev-Projects/local-codex-mavis-core-hooks`.

## Delivered

- `mavis.transcripts.TranscriptArchive.import_rollout` archives new complete JSONL records only, records source byte ranges and prefix hashes, refuses changed prefixes and incomplete trailing records, and fsyncs segment content before publishing the manifest. Source and archive remain local.
- `mavis pre-compact` reads Codex hook JSON, requires a matching session ID in the rollout header, confines the resolved rollout path to `CODEX_HOME`, imports the transcript, and writes a manifest-linked handoff. It records the current turn/trigger and explicitly marks objective state unavailable rather than inventing goals or decisions.
- `prepare_runtime.py` registers `python3 -m mavis pre-compact` as a managed `PreCompact` command. The Mavis launcher exports `MAVIS_HOME` and `PYTHONPATH` to the Codex core/hook process.
- `codex-core` uses the existing `Session::flush_rollout` barrier before `PreCompact` when `MAVIS_HOME` is set. A missing transcript, failed flush, absent matching hook, or failed hook result stops Mavis compaction. Ordinary Codex sessions retain existing behavior.

## Checks run

| Command | Observed result |
|---|---|
| `PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -q` | 42 tests passed, including incremental import, changed-prefix rejection, partial-record rejection, and hook CLI canary. |
| `python3 -m unittest discover -s local-codex/tests -p test_prepare_runtime.py -q` | 10 tests passed. Generated profile also parsed with `tomllib`; managed hook present. |
| `python3 -m compileall -q local-codex/mavis/mavis` | Passed. |
| `just fmt` in `codex-rs` | Passed. Its unrelated Python formatting edits were reverted. |
| `just test -p codex-core --lib hook_runtime` | 6 tests passed; 2,574 skipped by filter. |
| `just test -p codex-core --test all compact_hooks_respect_matchers_and_post_runs_after_compaction` | 1 integration test passed; 2,015 skipped by filter. Existing unrelated unused-import warning in `openai_file_mcp.rs`. |
| `git diff --check` | Passed. |

The first attempt at the focused integration test used the wrong target name (`suite`); Cargo identified the actual target as `all`. The corrected command passed. No full workspace suite, installed Mavis session, live compaction, or active service was run.

## Limits and next checks

- The handoff fields for goals, accepted decisions, completed requirements, current changes, and failures still need to be populated by the objective coordinator. Until that integration exists, the file says those fields are unknown and points to the durable transcript. This is a truthful recovery artifact, not a complete semantic handoff.
- The current core gate uses the Mavis launcher environment marker. An installed-canary must prove the launcher, Codex config loader, hook command, transcript flush, archive, and interrupted/restarted search end to end. A missing hook should stop compaction under this marker.
- This slice does not capture arbitrary Codex shell output before its 1 MiB head/tail cap. Mavis-owned `run_command` already records its own full stdout/stderr; broad raw-output spooling remains a separate slice.
- This worktree was not committed or pushed, per assignment. It did not modify the main checkout.

## Independent review repair (2026-09-22)

Terra rejected the first slice because a successful unrelated hook could satisfy the gate, and because `MAVIS_HOME` alone could change ordinary Codex. The repair now:

- Uses the launcher-only `MAVIS_PRECOMPACT_REQUIRED=1` marker. Incidental `MAVIS_HOME` has no effect.
- Requires a completed synchronous command hook from Mavis's `config.toml`, carrying the distinct Mavis archive status label. An unrelated completed hook does not satisfy the gate.
- Adds the exact normalized hook trust hash to the generated profile. If the managed command or its settings change without regeneration, Codex does not trust/run it and Mavis compaction stops. A Rust test checks the Python-generated hash against Codex's own hash function.
- Returns valid `{"continue": true}` PreCompact hook output after the archive is written. The earlier receipt-shaped stdout was not a valid hook protocol response.
- Locks the archive during import so two processes cannot publish duplicate segments from one rollout offset; a two-process test covers this.

Post-repair focused results: 43 Mavis Python tests passed, 10 profile tests passed, 9 `codex-core` hook-runtime tests passed, and the existing compact-hook integration test passed. The installed end-to-end canary is still pending. The handoff's objective-derived fields remain an open integration item.
