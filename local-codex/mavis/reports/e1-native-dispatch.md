# E1 native candidate dispatch source slice

## Ownership and current boundary

Mavis owns the frozen E1 manifest, candidate snapshot, prepared case checkouts, host acceptance checks, and comparison. The active Model Gateway worktree is `/Users/dustinpainter/Dev-Projects/ai-skills-dev-mavis-gateway` on `codex/mavis-gateway-phase1`. Its `harness_assignment_start` tool starts a native OpenCode MiniMax or ZCode GLM harness job in tmux. `harness_job_status` returns the durable process receipt and, after independent Terra acceptance, a `mavis_binding` that includes the assignment and report hashes. The gateway start tool gained `mavis_objective_id`, `mavis_requirements`, and `mavis_owner` in gateway commit `8378227d` while this Mavis source slice was being built.

## Implemented

- `e1 dispatch-candidate ID CASE --job-id JOB --lane minimax|zcode --model MODEL --task TASK` requires a frozen experiment and an exact clean prepared candidate checkout. It sends the snapshot hash, starting revision, checkout ownership, frozen check IDs, Mavis objective/requirements/owner, and task to the configured local gateway through `harness_assignment_start`. It requires a successful start receipt for the exact job, reads the gateway's durable `assignment.json`, checks the assigned fields, and saves a private dispatch record.
- `e1 compare-native ID CASE` now fails closed. A gateway job receipt proves the native worker's code task, but the installed Mavis selector does not yet issue a host-produced receipt proving that it applied the frozen candidate profile. The command reports this missing gate and leaves the experiment as a candidate. The original manual `e1 compare --candidate-job-id --candidate-report` command remains available; its result is a provisional source comparison, not installed-runtime proof.
- E1 host checks now require a committed, clean checkout before and after each check, including no ignored files. Comparison and later bundle validation recheck that the checkout is still clean and at the exact checked commit. This blocks attributing different dirty content at the same Git HEAD to an accepted worker. It also means a native worker must commit its repair and leave no generated files in the case checkout before E1 host checks run.
- The gateway MCP client uses the same configured trusted command for status and start, and rejects unsuccessful tool responses. No direct provider API or flat completion is used.

The native path is deliberately narrow: one selected candidate case can be dispatched as a repair task. It is **not** an E1 performance comparison yet. The gateway worker does not apply a Mavis model profile or propagate a code repair among separate case clones. A future installed-runtime selector must produce a host receipt that records the actual applied profile and binds it to each E1 case and check. Until that exists, native E1 comparison is blocked by design. Manual comparison can still be produced for source investigation, but its stamped configuration hash is not proof of applied runtime state.

## Evidence

- `PYTHONPATH=local-codex/mavis python3 -m unittest discover -s local-codex/mavis/tests -q`: 94 tests passed after the review repair.
- `python3 -m compileall -q local-codex/mavis/mavis`: passed on the final source.
- `git diff --check`: passed on the final source.

Source tests use a gateway stub and exercise dispatch arguments, rejection of a mismatched start receipt, the activation gate, dirty and ignored checkout rejection, and changed-content rejection after a host check. They do not claim a live native worker, oMLX run, installed Mavis candidate, or Phase 3 acceptance. The active gateway MCP process may need a restart before its tools expose the newly added Mavis fields. A safe source canary is to list its tool schema, then verify `harness_assignment_start` includes all three fields; a real dispatch should be reserved for an authorized E1 fixture and independently verified.
