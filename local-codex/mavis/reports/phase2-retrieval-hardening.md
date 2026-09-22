# Phase 2 retrieval hardening

Date: 2026-09-22. Worktree: `/Users/dustinpainter/Dev-Projects/local-codex-mavis-phase2-retrieval`. Branch: `codex/mavis-phase2-retrieval`. Base: `ffbe0e8307aa1da73255a8901ca37a66a7360235`.

## Scope and result

This slice hardens deterministic project retrieval without loading an embedding model or touching a live Mavis or IRIS service. It covers stale, deleted, and renamed source; failed index jobs; exact text, symbol, and dependency lookup; paging; and project-first verified global coding records.

- Retrieval reads current candidate files and verifies each live hash before using cached content. A changed Git HEAD, absent or failed job, or hash mismatch triggers live fallback. Deleted paths are omitted. Changed uncommitted files are marked `changed-file`.
- Refresh records the Git revision, changed/deleted paths, hash-identical renames, and job failure. `project-index status` shows whether the snapshot is current and the last job result.
- `git ls-files -z` avoids line-based filename parsing. Symlinks are excluded from refresh and live retrieval, and `.mavis/` plus a project-nested shared home are omitted.
- Exact search ranks project hits first. Verified `global-coding` records under Mavis home are only returned while all declared source hashes match. Paging supports limits through 200 and offsets. CLI exposes `search`, `symbol`, and `dependency` paging.

## Verification

- `git diff --check`: passed.
- `python3 -m compileall -q local-codex/mavis/mavis`: passed.
- `PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -v`: 54 passed, 0 failed.
- New tests cover committed stale content, rename/delete before refresh, failed job with live fallback, exact symbol/dependency lookup, wider retrieval and paging, project-first global records, invalidated source hashes, internal symlink exclusion, and CLI argument parsing.

## Limits and remaining Phase 2 work

- This is a deterministic exact retrieval slice. Embedding comparison, indexing decisions/failures/tool recipes/history, and model-backed helper evaluation remain open. No model was loaded here.
- Hash-based rename reporting recognizes an unchanged file moved to a new path. A rename with simultaneous content edits is treated as delete plus add; retrieval still serves the current path.
- Global knowledge records are read-only here. This slice does not author, curate, or reverify the meaning of claims; it checks their declared source hashes before serving them.
- Retrieval scans current candidate files to prevent stale answers. Large repositories may need bounded performance work after a representative benchmark.
