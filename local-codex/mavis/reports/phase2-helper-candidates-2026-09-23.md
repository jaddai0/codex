# Phase 2 local helper comparison — 2026-09-23

Status: **source and local-service canary passed; installed workflow remains open.**

The frozen suites are under
`~/.local-codex/mavis-service/diagnostics/phase2-helpers/fixture-manifest-211a299eb394.json`.
They contain two cited-history questions, including a follow-up, and four raw-output
cases. The output suite includes a known pass, a buried failure, an irregular
failure, and an uncertain completion. Host code checks citation paths and hashes,
raw log lines, exit status, and the final verdict independently of model prose.

Both Qwen3.5-0.8B MLX 4-bit and pinned Qwen3.5-4B BF16 ran on Mavis oMLX port
8001 while IRIS's two original models stayed loaded on port 8000. The 4B Hub
revision was `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; `hf cache verify`
checked all 14 files. Its temporary model alias was removed after the run.
Every run records matching pre/post model inventories and an empty Mavis service
after unload.

| Request contract | 0.8B librarian | 0.8B output reader | 4B librarian | 4B output reader |
| --- | ---: | ---: | ---: | ---: |
| Initial free-form JSON request | 0/2 | 2/4 host-only cases | 0/2 | 2/4 host-only cases |
| Strict role JSON schema, thinking off | 2/2 | 4/4 | 2/2 | 4/4 |

The final request contract also sends only the prior question as follow-up
context. It does not offer old citations for reuse. A path normalization fix
keeps a same-conversation context valid when macOS resolves `/var` through
`/private/var`. The Mavis package suite passed 291 tests after these changes.

In one final run, the model-called cases took 25.7 seconds on 0.8B and 78.3
seconds on 4B. This is a local observation, not a stable speed estimate. Since
both passed these cases, 0.8B is the provisional choice for each helper role.
The role receipts are `qwen0-8b-followup-result.json` and
`qwen4b-followup-result.json` in the diagnostics directory. The MLX constraint
record there passes identify, act, and candidate checks with an inconclusive
production verdict.

These synthetic cases do not prove that either helper improves completed coding
work. The final installed Mavis package must still exercise the librarian after
repeated compactions, the output reader on a real long log, isolation, and a
completed-task comparison before Phase 2 can pass.

## Installed follow-up — 2026-09-24

The installed candidate changed the provisional selection. On a fresh archived
decision, 0.8B added an unsupported database rationale and missed the condition
for revisiting PostgreSQL. On an irregular 9,848-byte host log, 0.8B exhausted
its 768-token response limit and returned no usable observation; the host still
reported the failed exit. The pinned 4B model answered both history questions
accurately and cited the exact buried log line while retaining the failed host
verdict. Native Terra independently accepted those narrow 4B results and
classified both 0.8B failures. The 4B alias is retained for the next paired
completed-work comparison. The full receipts and limits are in
`phase2-installed-helper-canaries-2026-09-24.md`. Net benefit remains open.
