# Mavis implementation contract

This directory owns Mavis-specific orchestration, evidence, profiles, memory,
evaluation, and launch integration. Shared provider discovery, subscription
accounting, and external harness adapters belong in the canonical
`ai-skills-dev/marketplace/plugins/model-gateway` source.

## Boundaries

- Never reuse or modify IRIS's oMLX base path, port, caches, logs, prompts, or
  process. IRIS owns port 8000. Mavis defaults to port 8001 and
  `~/.local-codex/mavis-service` (`~/.mavis` is a MiniMax alias here).
- Reuse model weights read-only. All mutable oMLX state must live below the
  Mavis home.
- Check aggregate memory and active local generations before loading a model.
  One local generation workload is the default.
- Cloud work goes through the trusted model gateway. Never put credentials,
  personal identifiers, or unredacted secrets in a worker assignment.
- Jev is advisory. Deterministic code owns permissions, spending, required
  checks, and acceptance. The calendar-month Jev cap is USD 10.
- A worker may not verify its own work. Changed code invalidates an earlier
  verification receipt.
- Keep model weights, databases, transcripts, raw logs, caches, and secrets out
  of Git. Project-local runtime data belongs in `.mavis/`.

## Required workflow

1. Create or resume an objective from a versioned objective document.
2. Record every worker assignment and its checkout/path ownership.
3. Capture complete command output before any display truncation.
4. Run deterministic acceptance checks and attach host-recorded receipts.
5. Obtain an independent verifier receipt for each coherent milestone.
6. Move an objective to `accepted` only when every mandatory check and verifier
   verdict passes.

Use `python3 -m unittest discover -s local-codex/mavis/tests -v` for the Mavis
package. Run `python3 -m compileall -q local-codex/mavis/mavis` as a syntax
gate. Runtime acceptance uses `python3 -m mavis eval e0` from the installed
environment and must retain its receipts.

The direct coordinator implementation exception for the initial bootstrap is
`MiniMax/OpenCode authentication canary failed on 2026-09-22`; it ends after a
valid native MiniMax canary and handoff are recorded.
