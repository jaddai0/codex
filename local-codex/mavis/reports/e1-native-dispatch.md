# E1 native candidate dispatch source slice

## Ownership and current boundary

Mavis owns the frozen E1 manifest, candidate snapshot, prepared case checkouts, host acceptance checks, and comparison. The active Model Gateway worktree is `/Users/dustinpainter/Dev-Projects/ai-skills-dev-mavis-gateway` on `codex/mavis-gateway-phase1`. Its `harness_assignment_start` tool starts a native OpenCode MiniMax or ZCode GLM harness job in tmux. `harness_job_status` returns the durable process receipt and, after independent Terra acceptance, a `mavis_binding` that includes the assignment and report hashes. The gateway start tool gained `mavis_objective_id`, `mavis_requirements`, and `mavis_owner` in gateway commit `8378227d` while this Mavis source slice was being built.

## Implemented

- `e1 dispatch-candidate ID CASE --job-id JOB --lane minimax|zcode --model MODEL --task TASK` requires a frozen experiment and an exact clean prepared candidate checkout. It sends the snapshot hash, starting revision, checkout ownership, frozen check IDs, Mavis objective/requirements/owner, and task to the configured local gateway through `harness_assignment_start`. It requires a successful start receipt for the exact job, reads the gateway's durable `assignment.json`, checks the assigned fields, and saves a private dispatch record.
- `e1 compare-native ID CASE` uses the saved gateway report path. It requires a completed, independently accepted status with exact job, objective, starting/changed revision, checkout, ownership, required checks, assignment hash, candidate requirement, and report hash. It then passes the retained report into the existing paired comparison. The original `e1 compare --candidate-job-id --candidate-report` remains available for manual or recovery work.
- The gateway MCP client uses the same configured trusted command for status and start, and rejects unsuccessful tool responses. No direct provider API or flat completion is used.

The native path is deliberately narrow: one selected candidate case is dispatched and bound to the comparison's candidate job ID. E1 still requires host checks for every frozen baseline and candidate case. The gateway worker does not automatically apply a model profile or propagate a code repair among separate case clones. A full Phase 3 run must prove that those checkouts use the intended candidate configuration and that each accepted result reflects the assigned work.

## Evidence

- `PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -q`: 92 tests passed on the final source.
- `python3 -m compileall -q local-codex/mavis/mavis`: passed on the final source.
- `git diff --check`: passed on the final source.

Source tests use a gateway stub and exercise dispatch arguments, retained report ingestion, and rejection of a mismatched start receipt or report hash. They do not claim a live native worker, oMLX run, installed Mavis candidate, or Phase 3 acceptance. The active gateway MCP process may need a restart before its tools expose the newly added Mavis fields. A safe source canary is to list its tool schema, then verify `harness_assignment_start` includes all three fields; a real dispatch should be reserved for an authorized E1 fixture and independently verified.
