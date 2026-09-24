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

The installed Python package's `python3 -m mavis librarian ask` route read a Git-ignored project `.mavis`
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

The shell launcher was not yet dispatching `mavis librarian ask` when these
model calls ran. A later launcher review found that integration gap; the
subsequent launcher checkpoint and installed canary are recorded separately below.

## Installed shell launcher repair

The installed shell entrypoint lacked `librarian` in its Python-service command
case. The launcher now routes that command alongside `output`, and a regression
test invokes a real project archive through the shell without loading a model.
The first source suite after the dispatch change passed 311 tests with two
skips; the expanded focused CLI suite passed 18 tests. The final full source
suite, including the new regression test, passed 312 tests with two skips in
`Mavis/evidence/phase2-librarian-launcher-final-source-tests-7ad62299b.log`.
Terra accepted the
narrow launcher diff in native review thread
`01a0d24d-61f5-7523-bab1-bdf3d764be85`.

Commit `7ad62299b99a48ea26fefe06b095403fa020c27e` was installed after
backing up the previous candidate at
`~/.local/share/local-codex-backups/mavis-before-librarian-launcher-20260924`.
The installed Mavis package hash remained
`8abf9ed790da2655561f6de6df1e7b3581e8ade2c7dbb22a0f232993b057ade9`;
the shell launcher hash changed to
`7d0ec70042ba81938b11a7a4295808c73e09c459705cfb8a56c097a17d7ed733`.
The installed `mavis librarian ask --no-model` command returned a verified
project-local citation. The live 4B shell command then answered both questions
accurately. Its receipt at
`Mavis/evidence/phase2-librarian-installed-canary-Mavis-Qwen3.5-4B-HF-Eval-launcher-hash-7ad62299b.json`
records the shell hash matching the manifest both before and after, the same
installed candidate fingerprint, IRIS unchanged, Mavis parked, and GPU lease
released.

The changed installation received fresh candidate-matched E0 observers. Run
`9682efbf5d8f40ee9abe5f16ecf75ac9` passed all 10 cases with the same
before/after candidate fingerprint, IRIS loaded, Mavis unloaded, and GPU lease
released. E0's fingerprint covers the Desktop launcher; the shell hash is
bound separately in the live librarian receipt and install manifest. This
repairs the shell integration gap, while the Phase 2 completed-task benefit
comparison remains open.

Native Terra independently accepted this installed aggregate in
`Mavis/evidence/phase2-librarian-launcher-installed-terra-review.txt`, completed
thread `01a0d259-45fd-7a11-a015-5251669a3774`. It checked the shell bytes
against the manifest and canary, both archive hashes and answers, E0 10/10,
and the recorded IRIS/Mavis/GPU cleanup. This acceptance is limited to the
installed launcher and librarian role.

## Completed-work trial

The next installed coding pair is recorded in
`phase2-completed-work-canary-2026-09-24.md`. The 4B librarian arm completed
and passed five independent host checks. The baseline passed those host checks
but fell into a false tool-output loop and did not finish. A default-sandbox
helper invocation failed on loopback permission; the successful 4B arm used an
explicit network override. Native Terra accepted the narrow capability canary
and rejected promotion from this single, unmatched pair. The helper remains
experimental until default access and matched completed-work benefit pass.
