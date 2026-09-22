# Mavis implementation workspace

This directory turns `IMPLEMENTATION_PLAN.md` into an executable, evidence-led
system. `LEDGER.json` maps each delivery phase to its acceptance surface;
`contracts/` contains the six versioned machine contracts; `handoff/` contains
worker and verifier templates; and `baseline/` freezes the pre-change state.

Runtime data is stored outside Git under `~/.local-codex/mavis-service` and project `.mavis/`
directories. IRIS remains on its existing oMLX service. The Mavis service uses
an isolated base path and endpoint and refuses to replace a process it does not
own.

The first live dual-load canary rejected concurrent use of the 111.55 GB Qwen
profile: the Mavis load timed out and IRIS stopped answering until the Mavis
process was removed. Runtime admission now blocks any second local generation
model by default. A future retry requires a proven sequential handoff or a
materially smaller candidate; free memory alone is not treated as safety proof.

Run the quick gate with:

```sh
PYTHONPATH=local-codex/mavis python3 -m mavis eval e0
```

The command exits nonzero and records `reject` until all ten mandatory cases
have real receipts. Missing local-model or native-harness work is reported as
blocked rather than replaced with synthetic success.
