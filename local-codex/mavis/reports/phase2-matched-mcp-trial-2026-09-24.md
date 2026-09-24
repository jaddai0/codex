# Installed project-memory helper: matched coding trial

Date: 2026-09-24. Evidence: `/Users/dustinpainter/Dev-Projects/Mavis/evidence/phase2-matched-mcp/`.

## Decision

Keep the pinned Qwen3.5-4B librarian available on request through the installed project-bound MCP tool. Do not make it an automatic default or claim a general net benefit yet. The installed capability worked in the ordinary `workspace-write` sandbox with network access disabled, but the paired trial repeats one checkout task. Native Terra accepted the corrected evidence packet and rejected default promotion.

## Frozen trial and observed results

Six disposable checkouts began at Git revision `8cd85bbb674818f00549dca1cd6e0ba5b358659f`. Each held the same three-segment accepted checkout decision, the same protected dirty note, and the same hidden five-check host acceptance. The same installed Mavis candidate and local Qwen3.8 main model ran all six arms. Baseline agents searched the archive directly; helper agents called the pinned 4B librarian via `mavis_librarian_ask`. Pair 2 reversed arm order. Each arm used the normal workspace-write sandbox without a network override.

| Pair | Direct-search baseline | 4B helper | Measured model-arm cycle |
| --- | --- | --- | --- |
| 1 | Completed; host 5/5; 157.8 s | Completed; host 5/5; 133.5 s | Helper shorter by 24.3 s |
| 2 | Completed; host 5/5; 155.1 s | Completed; host 5/5; 150.3 s | Helper shorter by 4.9 s |
| 3 | Native exit 0, no code edit, host failure | Completed; host 5/5; 120.9 s | No accepted baseline completion to compare |

The arm timer includes model admission, agent work, and model cleanup; the independent host test runs immediately afterward outside that timer. The accepted-task-per-minute calculation in `constraint_record.json` is therefore a measure of *agent cycles whose later host check passed*, not exact time to accepted completion. It is a signal, not a reliable performance estimate. The third baseline's normal native exit was correctly rejected because it did no repair and failed the host gate. Every helper arm made a completed model-backed MCP call with cited project evidence. The installed candidate hashes, source revision, note hash, sandbox receipts, native logs, raw host logs, and cleanup observations are retained and checked by `compare.py`.

The first pair-1 helper run passed the host gate but its outer timing recorder crashed on a failed MCP result; it was not counted. Pair-3 helper had one attempt interrupted by an IRIS server reset and one blocked by the required 15-minute IRIS idle gate. Those attempts are retained in separate folders and were not counted. The final third helper arm ran after the idle gate, returned native exit 0, and passed the host gate. Mavis and the 4B helper were unloaded after each completed run; IRIS stayed loaded and the shared GPU lease was released.

## Independent review

The first native Terra review returned `PARTIAL` because it read `constraint_record.json` while the third-pair result was being incorporated. That record was corrected, its wording narrowed to the actual timer boundary, and both the record and comparison were hash-frozen for a follow-up review. The follow-up native Terra turn returned `ACCEPT` for packet consistency. Its before and after hashes match, and `review-closeout.json` binds the comparison, constraint record, initial partial review, and accepted follow-up. Both reviews reject automatic helper promotion: there are only two pairs where both arms completed, all pairs use one task, timings were not randomized, and external cloud billing was not independently reconciled. No active helper setting changed.

## Remaining Phase 2 acceptance

Run distinct held-out coding tasks under the same installed and host-checked conditions, measure exact accepted-completion time including host verification, and reconcile cloud use for those objectives. Keep any close timing result inconclusive until repeated. The local capability and default-sandbox route are accepted; a general completed-work benefit remains open.
