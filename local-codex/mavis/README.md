# Mavis implementation workspace

This directory turns `IMPLEMENTATION_PLAN.md` into an executable, evidence-led
system. `LEDGER.json` maps each delivery phase to its acceptance surface;
`contracts/` contains the six versioned machine contracts; `handoff/` contains
worker and verifier templates; and `baseline/` freezes the pre-change state.

Runtime data is stored outside Git under `~/.mavis` and project `.mavis/`
directories. IRIS remains on its existing oMLX service. The Mavis service uses
an isolated base path and endpoint and refuses to replace a process it does not
own.
