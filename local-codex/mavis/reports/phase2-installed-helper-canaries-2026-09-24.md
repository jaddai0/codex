# Phase 2 installed helper canaries — 2026-09-24

Status: **installed role canaries passed with pinned 4B; Phase 2 remains partial.**

The installed source revision was `b763b5c19c8c8a0c31d1d61a153a9b1537714826`.
Source and installed package hashes matched at
`8abf9ed790da2655561f6de6df1e7b3581e8ade2c7dbb22a0f232993b057ade9`.
The previous installation was backed up at
`~/.local/share/local-codex-backups/mavis-before-b763b5c19` before installation.
The fresh E0 run `70e44738e2c44eef83b75ede98b8a6b3` passed 10/10 cases;
the wrapper recorded the same candidate fingerprint before and after, IRIS
loaded, Mavis unloaded, and the GPU lease released. Its receipt is
`~/.local-codex/mavis-service/evaluations/e0/final-installed-shared.json`.

## Librarian

The production `mavis librarian ask` CLI read a Git-ignored project `.mavis`
archive. The first question asked why SQLite was chosen first; the follow-up
asked when that choice might change. Both model runs bound the loaded model ID
and exact local path to Mavis port 8001 and returned citations checked against
the archive bytes. The 60-second follow-up retained the prior query, while
citations were checked against the new question's evidence packet.

| Candidate | Technical route | Factual answer |
| --- | --- | --- |
| Qwen3.5-0.8B MLX 4-bit | Both calls and citation checks passed | Rejected: invented an in-memory rationale and missed the multi-user remote trigger. |
| Pinned Qwen3.5-4B BF16, Hub revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Both calls and citation checks passed | Accepted: SQLite was first because Mavis is local-first and single-user; PostgreSQL is reconsidered if multi-user remote needs emerge. |

The receipts are `Mavis/evidence/phase2-librarian-installed-canary-Qwen3.5-0.8B-MLX-4bit.json`
and `Mavis/evidence/phase2-librarian-installed-canary-Mavis-Qwen3.5-4B-HF-Eval.json`.
Native Terra independently accepted the 4B result and rejected the 0.8B
semantic result in `Mavis/evidence/phase2-librarian-installed-terra-review.txt`.
The completed native review thread is
`01a0d238-9370-7ea0-82b7-f7198f09c6d5`.

## Output reader

The installed host recorded `python3 emit_irregular.py` exiting 7. Its complete
stdout was 9,848 bytes, with the marker `HALT_SENTINEL_4F2B` buried on line 72;
there was no recognizable test-count or failure line. The host verdict was
`fail` before either helper was called. `mavis output inspect` then received
the same immutable receipt and full raw output in both candidate runs.

The 0.8B model used all 768 response tokens, returned incomplete JSON, and was
reported `unavailable`; the host verdict remained `fail`. The 4B model returned
the exact line 72 as an observation, and the host verdict remained `fail`.
Receipts are `Mavis/evidence/phase2-output-installed-canary-Qwen3.5-0.8B-MLX-4bit.json`
and `Mavis/evidence/phase2-output-installed-canary-Mavis-Qwen3.5-4B-HF-Eval.json`.
Native Terra independently accepted the 4B result and the fail-closed 0.8B
classification in `Mavis/evidence/phase2-output-installed-terra-review.txt`.
The completed native review thread is
`01a0d23e-0b82-70e0-ab4b-d07fa763474d`.

## Isolation and limits

Each live receipt retained the same installed candidate and IRIS process/model
before and after, parked the owned Mavis service, and released the locked GPU
lease. ComfyUI's standing server was left alone. A separate E0 buried-output
run logged one Metal prefill hang and a successful reconnect; its final answer
matched the host's raw log, and the later compaction and E0 canaries had no
repeat. That event remains in the E0 observer log, not hidden as a clean run.

The pinned 4B candidate is the provisional helper for both roles because the
smaller candidate failed real installed questions. The 4B alias
`~/models/vlms/Mavis-Qwen3.5-4B-HF-Eval` points to the verified Hub snapshot
and remains available for a paired completed-work comparison. These role
canaries do not establish a net benefit to accepted coding work or a production
helper profile. Phase 2's completed-task comparison and full closeout remain
open; source/package tests and role canaries are narrower evidence.
