# Mavis workspace consolidation

On September 23, 2026, the Mavis source work was consolidated under `/Users/dustinpainter/Dev-Projects/Mavis/` after inventorying Git worktrees, tracked changes, ignored artifacts, and active runtime paths.

- Active Codex source: `/Users/dustinpainter/Dev-Projects/Mavis/checkouts/codex/`, branch `codex/mavis-unified-install`.
- Active model gateway source: `/Users/dustinpainter/Dev-Projects/Mavis/checkouts/gateway/`, branch `codex/mavis-gateway-phase1`.
- Removed 42 obsolete worktree folders. Git branch and commit objects were checked after removal and retained. The exact paths and revisions are in `Mavis/evidence/checkout-consolidation.json`.
- Copied the ignored embedding index and two untracked snapshot candidates into `Mavis/evidence/`; SHA-256 hashes were checked before their old worktrees were removed.
- The old raw-output checkout had 17 tracked Python diffs with the same parsed syntax tree as its committed revision. Its 83,213-byte patch was hashed and retained under `Mavis/evidence/` before those formatting edits and the 30 GB reproducible build cache were removed. Its Git branch remains.
- The clean oMLX drain-lease worktree was removed; the live IRIS oMLX process uses the installed editable source at `/Users/dustinpainter/Dev-Projects/omlx`, which was not moved.
- The original `Dev-Projects/local-codex` checkout on `omlx-only` was also removed. Its commit is an ancestor of the active branch and its remote branch remains. The only uncommitted change was report whitespace; its 796-byte patch and SHA-256 hash are retained in `Mavis/evidence/`. Removing that checkout also removed its reproducible 49 GB Rust build cache. A top-level directory check now finds only `Dev-Projects/Mavis` for Mavis or local-codex work.
- Updated the Mavis launcher's gateway default and the installed Mavis gateway configuration to the new gateway checkout. A fresh installed `harness_job_status` call returned the accepted E0 worker from that path.

The installed Mavis package and its runtime receipts remain in their live home paths. The previous installation was backed up before installing the candidate from `20c90f333`.
