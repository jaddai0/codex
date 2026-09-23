# Phase 3 E1 runner source slice

The E1 CLI freezes a versioned case manifest and prompt/tool/retrieval snapshot, prepares disposable clones at each exact starting commit, runs the same acceptance commands on both arms, retains complete host receipts and raw output, and computes a held-out pass fraction. The regression is separate from held-out work and must refer to an earlier failed Mavis host receipt whose raw output still hashes correctly. A missing case or changed receipt blocks comparison. The candidate must repair the regression, pass every held-out case, and beat the baseline by the frozen minimum gain.

The original failure receipt must name the regression case's source checkout, revision, command, and check ID. Its exit status, timeout flag, verdict, raw hashes, and raw byte count are checked against the retained output. A content digest covers the frozen manifest, original failure, both arms' case results, host receipts, and raw logs. Review, staging, promotion, and promoted-profile assertions recheck that digest and the underlying host evidence, so changing a raw log after comparison blocks later gates. Source tests include altered original working directories and tampering after review and staging.

The case manifest is `mavis.e1-cases/v1`:

```json
{
  "schema_version": "mavis.e1-cases/v1",
  "scoring": "held_out_pass_fraction/v1",
  "regression": "observed-failure",
  "failure_receipt": {"path": "/absolute/path/to/receipt.json", "sha256": "64 hexadecimal characters"},
  "held_out": ["held-a", "held-b"],
  "minimum_gain": 0.5,
  "cases": [
    {"id": "observed-failure", "source": "/absolute/path/to/git/repo", "revision": "full 40-character commit", "checks": [{"id": "accept", "argv": ["python3", "-m", "unittest", "tests.test_case"], "timeout_seconds": 120}]},
    {"id": "held-a", "source": "/absolute/path/to/git/repo", "revision": "full 40-character commit", "checks": [{"id": "accept", "argv": ["python3", "-m", "unittest", "tests.test_case"], "timeout_seconds": 120}]},
    {"id": "held-b", "source": "/absolute/path/to/git/repo", "revision": "full 40-character commit", "checks": [{"id": "accept", "argv": ["python3", "-m", "unittest", "tests.test_case"], "timeout_seconds": 120}]}
  ]
}
```

Run from the Mavis package with `PYTHONPATH=local-codex/mavis python3 -m mavis ...`:

1. `e1 seed-active SCOPE BASELINE.json` once for an already accepted baseline.
2. `e1 freeze ID --scope SCOPE --kind prompts --candidate CANDIDATE.json --cases CASES.json --hypothesis '...'`. This prints the exact candidate assignment requirement for the native gateway job.
3. `e1 prepare ID baseline` and `e1 prepare ID candidate`. Assign each checkout to its respective native worker, using the frozen configuration snapshot. The CLI does not launch a worker or claim that the profile was applied.
4. `e1 check ID baseline CASE` and `e1 check ID candidate CASE` for every regression and held-out case, after the worker has finished. `e1 coverage ID` lists unfinished cases. Each case can be checked once; reruns need a new experiment ID.
5. `e1 compare ID --candidate-job-id JOB --candidate-report REPORT.json`, where `REPORT.json` is the unchanged native candidate worker report. Then `e1 review-requirements ID` prints exact strings for a separate native Terra review assignment.
6. Copy the native review worker's unchanged JSON report to `$MAVIS_HOME/verifications/experiments/`, then `e1 review ID RECEIPT.json` and `e1 stage ID`. The existing store validates both gateway jobs and Terra acceptance on review and staging.

`e1 compare` records a provisional comparison. The candidate job ID and report are not accepted until `e1 review` verifies their native gateway bindings. This slice does not dispatch gateway assignments, apply the active profile in the installed runtime, assert an idle objective boundary, promote, or prove rollback. Those gates remain open for the real Phase 3 canary.

The case result records the intended configuration hash. It does not observe the worker's applied configuration or bind the worker to each checkout; the score is therefore a host-check comparison until the live assignment and installed profile path supply that evidence.
