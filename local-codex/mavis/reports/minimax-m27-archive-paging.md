# Mavis Archive Search Paging — Minimax M2.7 Consultant Report

**Date:** 2026-09-22
**Task:** Add nonnegative offset and limit 1..200 pagination to `TranscriptArchive.search`, thread through `LibrarianEvidence.search` and `archive-search` CLI
**Outcome:** MiniMax implementation and coordinator review repair complete — 104/104 tests pass

---

## Prior Local Mavis Attempts

Two local Mavis sessions (`Mavis` bounded owner, `MiniMax M2.7` consultant) attempted this work before this session and stopped without committing edits. No prior diff was available to review. This session began fresh from the exploration baseline.

---

## What Changed

### `mavis/transcripts.py` — `TranscriptArchive.search`

Signature changed from `(query, limit=20)` to `(query, limit=20, offset=0)`.

- Added exact-integer, nonnegative offset guard → `ValueError("offset must be nonnegative")`
- Added `limit` bounds guard `1..200` → `ValueError("limit must be 1..200")` (previously only `LibrarianEvidence` validated this)
- Skips matches before the offset and holds at most one page of results. It stops scanning content when the page is full.
- Archive integrity checking (`sha256_file` vs manifest) still checks every segment, including those after a full page
- Return fields unchanged: `path`, `line`, `text`, `sha256`

### `mavis/helper_interfaces.py` — `LibrarianEvidence.search`

Signature changed from `(query, limit=20)` to `(query, limit=20, offset=0)`.

- Added exact-integer, nonnegative offset guard → `ValueError("offset must be nonnegative")`
- Delegates `offset` through to `self.archive.search(query, limit=limit, offset=offset)`
- `validate_answer` path is unaffected

### `mavis/cli.py` — `archive-search`

- Added `--offset` argument (type=int, default=0) to the argument parser
- Handler passes `args.offset` as third positional to `archive.search()`

---

## Tests Added

| File | Test | Coverage |
|------|------|----------|
| `test_transcripts.py` | `test_search_pagination_page1_and_page2` | Page 1 (offset=0, limit=2), page 2 (offset=2) return correct slices |
| `test_transcripts.py` | `test_search_offset_beyond_results_returns_empty` | offset beyond results → `[]` |
| `test_transcripts.py` | `test_search_invalid_negative_offset_raises` | offset=-1 raises `ValueError` |
| `test_transcripts.py` | `test_search_invalid_limit_out_of_range_raises` | limit=0 and limit=201 raise `ValueError` |
| `test_transcripts.py` | `test_search_stops_scanning_after_page_but_verifies_later_segments` | Bounded page across segments; late tampering still rejected |
| `test_helper_interfaces.py` | `test_librarian_search_validates_offset` | offset validation, page1/page2, out-of-range, negative |
| `test_cli.py` | `test_archive_search_cli_returns_requested_page` | Actual CLI handler returns page two across segments with citation hashes |
| `test_cli.py` | `test_archive_search_cli_accepts_offset_argument` | `--offset 10` parses correctly |
| `test_cli.py` | `test_archive_search_cli_accepts_limit_and_offset_together` | `--limit 5 --offset 20` |
| `test_cli.py` | `test_archive_search_cli_defaults_offset_to_zero` | Default offset is 0 |

---

## Test Results

**Focused unit tests:** 26 tests across `test_transcripts`, `test_helper_interfaces`, `test_cli` — all pass.

**Full Mavis suite:** 104 tests, 0 failures, in ~22s after the review repair.

```
Ran 104 tests in 21.553s
OK
```

---

## Limitations

1. **Installed CLI not yet rerun** — the source CLI handler and a direct `python3 -m mavis archive-search` canary pass, but the final installed package is older than this patch.

2. **No behavioral test for `LibrarianEvidence.validate_answer` with paged evidence** — the validation path is unchanged but untested with paged result sets.

3. **Offset validation in `TranscriptArchive.search` is redundant with `LibrarianEvidence`** — both layers now validate `offset < 0`. This duplication is intentional: the archive layer is self-validating and the helper layer enforces the contract for external callers.

4. **Archive search still verifies all segment hashes** — this is intentional integrity work. The independent review's 25,000-match fixture measured approximately 1.35 MiB peak with `limit=1`, down from 7.42 MiB before the repair. Deep offsets still require scanning preceding matches.

5. **eval e0 suite not run** — per AGENTS.md, `python3 -m mavis eval e0` requires the installed runtime environment and must retain host-recorded receipts. This was not executed in the bounded checkout context.

---

## Files Edited

The change is confined to `mavis/transcripts.py`, `mavis/helper_interfaces.py`, `mavis/cli.py`, and their three matching test modules. The later coordinator repair added the bounded scan and complete CLI test in those same files.

MiniMax's first tmux job encountered a provider authentication response, while a fresh isolated tmux read canary succeeded. Its next job wrote the code and passed tests; writing the report outside the owned checkout was denied. The same OpenCode session was resumed to write this report inside the checkout and return a final answer. Independent review then found the collect-all paging defect; the coordinator repaired it and repeated the tests. No model services, connected search tools, unrelated files, or protected evaluation tests were changed.
