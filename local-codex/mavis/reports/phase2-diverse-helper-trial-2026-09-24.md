# Installed project-memory helper: distinct coding tasks

Date: 2026-09-24. Evidence: `/Users/dustinpainter/Dev-Projects/Mavis/evidence/phase2-diverse-mcp/`.

## Decision

Reject automatic default use of the pinned Qwen3.5-4B librarian. Keep the installed, project-bound `mavis_librarian_ask` tool available on request. One of two distinct helper tasks failed mandatory host acceptance, even though its archive answer and model call were valid. Native Terra accepted the frozen packet and independently rejected promotion.

## Trial

The settings parser and queue selector were separate frozen Git fixtures with separate accepted decisions stored in three-segment project archives. Within each task, direct-search and helper arms began at the same commit, with the same dirty user note and the same visible and hidden tests. The host kept five acceptance checks outside the agent checkout. The installed Mavis main model and candidate fingerprint were identical across all four arms. Every native session used local oMLX, ordinary `workspace-write`, and restricted network access. Pair order was settings baseline then helper, queue helper then baseline.

| Task | Direct archive search | Pinned 4B librarian |
| --- | --- | --- |
| Settings parser | Completed; host 5/5; 207.8 s | Cited 4B call; no code edit; no visible test run; host failure |
| Queue selector | Completed; host 5/5; 261.1 s | Completed; host 5/5; 215.2 s |

The accepted-completion timer includes local model admission, the agent turn, cleanup, and the independent host check. The queue helper completed about 45.9 seconds sooner. The settings helper's shorter 152.6-second arm is a failed completion and contributes no accepted-time benefit. It located the correct decision and even explained a viable implementation in its reasoning, but ended its native turn before editing `settings.py`. Native exit zero alone did not pass the host gate.

The retained results include before/after installed fingerprints, protected-note hashes, changed-file lists, native event logs, actual model-backed MCP calls and source citations, host logs, sandbox rollout references, and cleanup. `compare.py` replays the host checks and hashes. Every completed arm unloaded Mavis and the helper, retained IRIS's loaded model, and released the shared GPU lease. No default setting changed.

## Review and remaining work

Native Terra returned `ACCEPT` for evidence integrity and `REJECT` for default promotion. `review-closeout.json` binds the frozen fixture, comparison, review, and native receipt; fixture and comparison hashes did not change during review. This expands the earlier repeated-checkout trial with two different tasks and exposes a real helper-assisted completion failure. Phase 2's general net-benefit criterion remains open. A future candidate must address the early-turn failure and pass mandatory held-out tasks before comparison of accepted completion time and cloud use can justify promotion.
