# Phase 2 helper source interfaces

Date: 2026-09-22. Worktree: `/Users/dustinpainter/Dev-Projects/local-codex-mavis-phase2-helpers`. Branch: `codex/mavis-phase2-helpers`. Base: `8afb3c4ccd41074040bbc2ba8293915456285bb4`.

## Scope

The existing Phase 2 retrieval slice already covers current source hashes, stale and deleted files, paging, and verified global records. This slice adds source-verifiable boundaries for the two helper roles without running a model or touching installed services.

`LibrarianEvidence` searches the existing transcript archive. An answer must include text, explicit uncertainty, and citations to lines in the verified evidence packet. The validator checks the archive manifest, segment hash, line number, and exact source text again when the answer is accepted. It rejects foreign files and altered segments. It does not assert that a model's interpretation of the cited line is correct; that needs independent review.

`OutputReader` reads the host's command receipt and both complete raw logs. It verifies their stored hashes and byte count, carries the exit status, timeout, failure lines, count lines, and raw reference forward, and derives a verdict that cannot be promoted by model prose. For a non-passing result, a proposed freeform summary is rejected, leaving the host's failure evidence visible. The full raw logs remain available for inspection.

## Verification

- `PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -v`: 79 passed, zero failed after the zero-failure-count repair.
- `python3 -m compileall -q local-codex/mavis/mavis`: passed.
- `git diff --check`: passed.
- `just fmt` ran as instructed. It reformatted many unrelated tracked Python files; those formatting-only changes were reverted so this branch contains only the helper slice.
- `~/.claude/bin/work-closeout check local-codex/mavis/reports/phase2-helpers-ledger.json --stage closeout`: passed for this bounded slice.

The new tests cover cited early decisions, missing uncertainty, forged and altered sources, a failure buried among 2,000 ordinary output lines, attempted success promotion, nonzero exit status, raw output tampering, and retained test counts.

An independent review caught a false failure in the first commit: a passing summary such as `10 passed, 0 failed` was treated as a failure because the reader matched the word `failed`. The repair strips only zero-failure and zero-error counts before checking failure markers, while a positive failure count or explicit failure marker on the same or later line still fails. Two regression tests cover the passing count and mixed-format failure. The full suite passed after this repair.

## Remaining Phase 2 acceptance

These interfaces are source verified but not yet wired to a helper model or installed Mavis runtime. The Phase 2 success gate still requires repeated compactions with early-decision recovery, a live large-log failure trial, helper identity/cache isolation under actual model use, latency and memory checks, and evidence that helpers improve completed work. Embedding model comparison and wider retrieval history/indexing also remain open. No model was loaded or trained in this slice.
