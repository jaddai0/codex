# Mavis: local coding execution with verified improvement

Planning handoff for Sol. Prepared September 22, 2026. This document records the plan and user decisions; it does not claim implementation or acceptance is complete. Recheck live model, provider, runtime, and subscription state before execution.

## 1. Outcome and starting point

Build Mavis into the primary developer for your projects. Astra and Fable can supply architectures and blueprints; Mavis carries them through implementation, diagnosis, testing, and completion. Cloud models provide bounded assistance and independent verification.

**Optimize for accepted coding work, completion time, and cloud usage together.** More local tokens are worthwhile when they improve results. Lower token counts, faster generation, or more agents do not count as improvements by themselves.

The first release must become useful before the advanced evaluation and training system is finished.

### What exists today

Read-only inspection found:

- An existing Mavis launcher, isolated configuration directory, editable persona, and local oMLX connection.
- Qwen3.8 Flash Next currently loaded in the running oMLX server.
- Codex extension interfaces, conversation storage, compaction hooks, and compressed archives that can be reused.
- Gateway provider discovery, quota records, report storage, and some CLI integration.
- Installed tmux, OpenCode, ZCode, Claude, and Codex executables. Their presence does **not** prove working authentication or delegation.

The main gaps are:

- Mavis currently shares IRIS’s server endpoint.
- The gateway’s ordinary delegation is mostly single-shot generation, not external agents operating in their harnesses.
- Large tool outputs can lose their middle before archival.
- Worker-reported success is not independently established.
- Model profiles do not yet preserve learned configurations and evaluation history.

### Decisions already settled

- Keep the current server as **IRIS’s server**. Create an isolated Mavis server.
- Begin with the currently loaded Qwen variant, identified by its actual files and configuration.
- Allow relevant code, transcripts, logs, and project context to reach cloud helpers. Filter credentials, identifying personal information, and similarly compromising material.
- Allow Jev API spending up to **$10 per calendar month**. Other paid APIs require approval; existing subscriptions are available.
- Subscription reserves are **soft preferences**, never fixed percentage cutoffs.
- Reversible prompt, tool, retrieval, and harness improvements may be promoted after independent checks. Trained adapters require your approval.
- Longer evaluations run in resumable idle windows, initially after 15 minutes without foreground agent work.
- Add the small librarian and output-reading models after the first useful release.
- Coding receives priority when tuning requires a tradeoff, without turning Mavis into an agent that refuses other requested work.

## 2. Architecture and operating contracts

### Keep ownership narrow

Use the existing fork and gateway rather than create a second general agent platform.

| Component | Responsibility |
|---|---|
| Mavis Codex extension | Connect tools, context, lifecycle events, and profile selection to the existing harness. |
| Mavis service | Track objectives, evidence, project memory, retrieval, and improvement experiments. Start as one Python package with separate modules. |
| Model gateway | Own provider discovery, subscription accounting, external harness jobs, routing, and usage receipts. |
| tmux runner | Keep external harness processes observable and recoverable. It does not decide whether work succeeded. |
| oMLX instances | Serve isolated identities and model contexts under a shared compute admission policy. |
| Independent verifier | Check requirements against actual changes, commands, artifacts, and runtime behavior. |

Use Codex’s existing extension interfaces. Add narrowly scoped core hooks only where necessary, particularly raw-output capture and durable compaction. Do not put subscription logic, training, or archive management inside `codex-core`.

Keep Mavis-specific source and handoff documents under `local-codex/mavis/`. Shared gateway improvements belong in its canonical plugin repository. Respect the fork’s existing restriction against adding general product documentation to its root `docs/` directory.

### Define six versioned contracts

Store machine-readable schemas alongside implementation tests:

1. **Objective:** approved blueprint, requirements, dependencies, scope, acceptance checks, and unresolved decisions.
2. **Worker assignment:** objective subset, starting revision, owned paths/worktree, allowed effects, model/effort, expected artifacts, and escalation conditions.
3. **Evidence receipt:** host-recorded command, exit status, timestamps, raw-output reference, changed revision, artifact hashes, and verification verdict.
4. **Model profile:** exact model identity, compatible runtime, prompts, tool settings, context policy, experiment history, and active/previous versions.
5. **Memory record:** scope, claim, source references, verification state, freshness, and superseding records.
6. **Experiment:** hypothesis, frozen baseline, candidate difference, workload, evaluation split, results, reviewer, and promotion decision.

Use explicit states: queued, running, awaiting verification, accepted, needs repair, escalated, blocked, and cancelled. A process exiting successfully does not move an objective directly to accepted.

### Separate Mavis and IRIS completely

- Preserve IRIS’s current process, settings, prompts, and caches.
- Start Mavis with its own oMLX base directory, endpoint, response state, memory cache, disk cache, and logs.
- Reuse immutable weight files; do not assume two processes share their loaded model memory.
- Keep Mavis’s existing isolated harness home. Explicitly register the trusted gateway there rather than importing the entire frontier configuration.
- Keep the main Mavis model connection local. Cloud access happens through gateway tools.
- Check aggregate memory and active workloads before loading another model. Do not let each server independently assume it can consume nearly all system memory.
- Default to one active local generation workload. Helper work runs while Mavis waits or during idle gaps. Concurrent local inference requires measured benefit.
- Use canary facts to test separation across prompts, histories, retrieval, server state, and caches.

Create `~/Desktop/Mavis.command`. It must start or reconnect to Mavis’s own services, verify their identities, and open a Mavis terminal session without replacing an unrelated process or changing IRIS.

### Make delegation enforceable

Sol coordinates interfaces and exceptions. It should not routinely implement plumbing, tests, or cleanup itself.

| Work | Default owner |
|---|---|
| Architecture, cross-component decisions, unresolved integration | Sol |
| Launchers, adapters, schemas, routine tests, hygiene | MiniMax through OpenCode |
| Routing, quota accounting, diagnostics, failure testing | GLM through ZCode |
| Ordinary coding after bootstrap | Local Mavis |
| Independent milestone verification | Terra through Codex |
| Escalated implementation or diagnosis | Sonnet, then Opus |
| Hardest unresolved architecture or reasoning | Astra/Fable |

Every implementation assignment must record an owner. Direct Sol implementation requires a short reason such as architecture-sensitive integration or failure of eligible cheaper workers. The coordinator checks this policy before dispatch.

Use native harnesses for tool-using work. Preserve ordinary gateway API generation for classification, bulk generation, and bounded opinions.

External workers run in dedicated tmux sessions with structured events and durable logs. Give concurrent writers separate worktrees. Allow only one writer per checkout.

**A worker cannot certify its own success.** The supervisor gathers evidence; deterministic checks run first; Terra then checks the coherent milestone. Sol receives a compact accepted receipt or a specific exception. Changed code or acceptance artifacts invalidate the old receipt.

Terra checks for missing requirements, disabled tests, fabricated success, unexpected edits, and fixes that merely hide the reported failure. Evaluation tests remain outside the worker’s writable area.

### Escalation and persistence

Mavis keeps ownership of the overall objective even when a consultant handles a difficult part.

- Try a materially different, evidence-based repair before escalating ordinary task difficulty.
- Two consecutive repair attempts with the same failure and no new evidence trigger consultation.
- Escalate sooner for contradictory evidence, expanding damage, missing capability, or an architectural decision outside the blueprint.
- Increase effort within a model when supported and useful; do not exhaust every effort level mechanically.
- Follow the requested MiniMax ladder: **M2.7 → M3 → Sonnet → Opus → Sol → Astra/Fable**.
- For GLM, use available same-family steps before crossing families.
- Skips require a recorded capability, availability, or urgency reason.
- Distinguish model difficulty, quota exhaustion, broken infrastructure, and missing authority. They require different responses.
- A broken required harness stops that lane until repaired and canaried; do not silently replace it with a flat API call.
- Exhausting an escalation route produces a recoverable blocked objective with evidence, not an endless loop.

### Adapt to subscriptions

Extend the existing gateway ledger to represent accounts, shared model pools, simultaneous usage windows, reset times, observations, and concurrent-job reservations.

Unknown usage stays unknown. Corrupt accounting cannot become “unused capacity.”

Routing considers task difficulty, measured success, time remaining before resets, recent personal usage, latency, and available subscriptions. Favor GLM/MiniMax for suitable cloud work. Preserve frontier headroom through a soft routing penalty, while allowing justified use for completion and verification.

Refresh availability at startup, after provider errors, and daily. Produce a monthly subscription-policy review. Changing subscriptions must update configuration and adapter availability without rewriting Mavis.

Do not silently fall through from a subscription to paid billing.

## 3. Delivery phases and success criteria

### Phase 0 — Establish the baseline and delegate the build

**Owner:** Sol; inventory and fixtures delegated to MiniMax/GLM.

Create:

- The implementation handoff, component ownership map, and requirement-to-evidence ledger.
- Project-level `AGENTS.md` and `CLAUDE.md` adapters covering boundaries, commands, delegation, verification, and cleanup.
- Worker-assignment and verifier templates.
- A frozen baseline covering installed binaries, source revisions, model identity, live services, and existing acceptance evidence.

Record existing failures separately from new regressions. Preserve unrelated work and paused workloads.

**Success:** every phase has an owner and executable acceptance criteria; each required external harness proves it can perform a small real task in an isolated fixture.

### Phase 1 — Make Mavis useful

**Owners:** MiniMax for launcher and evidence plumbing; GLM for gateway jobs and Jev; Terra verifies.

Deliver:

- Isolated Mavis server and Desktop launcher.
- Versioned model profiles and explicit model selection.
- Objective tracking, checkpoints, bounded repair attempts, and escalation.
- Complete raw tool-output capture before truncation.
- Persistent transcripts and compaction handoffs.
- Native MiniMax, GLM, and Terra delegation.
- Jev decision integration.
- Deterministic test-output parsing and direct archive search.
- Recovery after interruption.

Use Qwen Flash Next first. Keep generation settings close to the working baseline until comparisons justify changing them.

**Success:** the installed launcher passes the basic evaluation below, then Mavis completes one bounded real maintenance task and one multi-file task with independent verification. It must also preserve unrelated dirty work and recover an interrupted objective.

Do not delay this phase for embeddings, small helper models, broad benchmark runs, or LoRA training.

### Phase 2 — Add project memory, retrieval, and small helpers

**Owners:** Mavis/MiniMax implement; GLM tests retrieval; Terra verifies.

**Memory and retrieval**

- Keep full project evidence under a Git-ignored `.mavis/` directory; store shared coding knowledge in Mavis’s own home.
- Retrieve project knowledge first, then verified global coding knowledge.
- Combine exact text search, code-symbol/dependency navigation, and embeddings. Embeddings supplement exact lookup.
- Index source files, decisions, failures, tool recipes, and relevant history.
- Update changed content by hash. Handle renames, deletions, branch changes, and failed index jobs.
- Before embedding catches up, search changed files directly.
- Record embedding model and index versions. Never compare vectors from incompatible models.
- Deduplicate and rank results, but support paging and wider retrieval when evidence is insufficient.

Begin the lightweight embedding comparison with Qwen3-Embedding-0.6B and the already installed embedding model. Qwen publishes code-related retrieval support, but selection depends on Mavis’s tasks and compute cost, not its benchmark rank. [Model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

**Librarian**

- Save the complete conversation before compaction, with a manifest linking all prior segments.
- Write a new-context handoff containing goals, accepted decisions, completed requirements, current changes, recent work, unresolved failures, and evidence links.
- The librarian answers history questions with source citations and explicit uncertainty.
- Give it the largest context that passes its memory, latency, and retrieval tests; do not fill it merely because space exists.
- Keep query context for 60 seconds after completion for follow-ups, then clear it. Under compute pressure, clear sooner after the response.
- Isolate the librarian’s identity, history, and caches from Mavis, IRIS, and the output reader.

**Output reader**

- Parse known test formats first.
- Use the small model for irregular output, preserving failure messages, counts, exit status, and raw references.
- It cannot convert a failure, timeout, incomplete run, or uncertain result into success.
- Mavis can inspect the full output whenever needed.

Evaluate Qwen3.5-0.8B first, then 4B, for each helper independently. Choose the smallest passing candidate; do not force both roles onto one model. These are candidates, not established winners. [0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [4B](https://huggingface.co/Qwen/Qwen3.5-4B)

**Success:** recover an early decision after repeated compactions; catch a failure buried in a large log; handle stale and deleted code correctly; prove helper isolation and a net benefit to completed work.

### Phase 3 — Establish continuous improvement

**Owners:** Mavis generates candidates; MiniMax/GLM implement bounded repairs; Terra verifies promotion.

Collect objectives, model/profile versions, available model traces, tool calls, raw outputs, changes, tests, reviewer findings, retries, routing decisions, latency, and usage.

Do not assume unavailable private model reasoning can be captured.

| Interval | Work |
|---|---|
| Immediate | Record failures, preserve state, repair the current task, and create regression cases. |
| Daily | Group recurring failures, patch stale indexes, evaluate one focused prompt/tool/retrieval candidate. |
| Weekly | Run broader comparisons, review project-to-global memories, assess helper quality and escalation behavior. |
| Monthly | Review model profiles, subscriptions, architecture health, and candidates for targeted training. |

Make these resumable queued jobs. Background work yields at safe boundaries when foreground work returns; it must not kill an active task or interfere with unrelated training.

Promote technical lessons globally only after their evidence and applicability are checked. Personal rules still require your explicit approval.

Prompt optimization should use failure explanations and controlled comparisons. GEPA supplies a useful research starting point, not proof that automatic prompt evolution will improve Mavis. [GEPA paper](https://arxiv.org/abs/2507.19457)

**Success:** one real failure produces a candidate, regression test, held-out comparison, independent review, staged promotion, and demonstrated rollback.

### Phase 4 — Harden difficult and large-project work

**Owners:** Mavis with GLM/MiniMax assistance; Terra verifies.

Add evaluation and workflow coverage for:

- Large repositories and cross-package changes.
- Long objectives spanning multiple compactions and restarts.
- Unfamiliar code, ambiguous failures, flaky tests, and broken dependencies.
- Concurrent independent worktrees and integration conflicts.
- Dependency-aware task decomposition and targeted test selection.
- Tool discovery that reveals relevant tools without flooding every prompt.
- Review of downstream effects and final integrated behavior.

Mavis should form a repository map, identify ownership and dependencies, implement one coherent slice, and verify it before expanding. It must not load the entire project by default.

**Success:** complete a held-out multi-package task across compaction and restart, preserving the blueprint and unrelated changes, with independently verified end-to-end behavior.

### Phase 5 — Introduce a local Jev-style decision service

Hosted Jev remains a core harness component from Phase 1. Use it for bounded decisions: relevant evidence, tool/skill selection, failure classification, escalation recommendations, memory categorization, and suspicious completion claims.

Use the dedicated Decisions API with a pinned model version and validated request shapes. OpenRouter marks this interface alpha, so keep it behind an adapter with compatibility tests. [Official integration](https://github.com/OpenRouterTeam/ai-sdk-provider)

Jev advises; deterministic code retains authority over permissions, money, required checks, and completion.

For local candidates:

- Compare Bespoke Nimble’s MLX scorer and independently labeled training recipe.
- Evaluate the published OpenJev MLX build only for uses compatible with its noncommercial license.
- Include a small classifier baseline so a larger specialist must justify its compute cost.

Nimble currently documents merged, non-quantized weights for its MLX runner; OpenJev publishes a separate 4-bit MLX build. Neither is established as suitable on this machine by this research. [Nimble](https://github.com/bespokelabsai/nimble), [OpenJev MLX](https://huggingface.co/openjev/openjev-MLX-4bit)

Keep hosted Jev outputs separate from local-training targets. TypeSafe’s published agreement restricts distillation and replacement development; verify applicable OpenRouter terms before using those outputs for that purpose. Independently observed test outcomes and separately labeled examples form the default training corpus. [Published agreement](https://typesafe.ai/legal/mca)

**Success:** a local candidate passes held-out decision tests and improves the whole workflow after accounting for shared compute. Start with shadow decisions, then limited traffic. Keep hosted Jev available within budget.

### Phase 6 — Train targeted model adapters

Proceed only when repeated evidence shows a model weakness that prompt, retrieval, or tool changes have not adequately resolved.

- Build narrowly targeted datasets from verified work and corrections.
- Split by project, task lineage, and time to prevent near-duplicate leakage.
- Preserve hard failures and negative examples.
- Keep adapters tied to the exact base model, tokenizer, architecture, and training recipe.
- Test tool use, long-task behavior, escalation, and honest completion—not only training loss.
- Verify support for the actual Qwen/GLM/DeepSeek architecture before planning training capacity.
- Use local training when compatible and idle; paid compute requires approval.

MLX supports parameter-efficient training through LoRA/QLoRA, but that does not establish compatibility with every future model. [MLX training documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

**Success:** a targeted adapter improves unseen coding work without mandatory regressions, receives independent verification and your approval, and can be rolled back without losing the base profile.

## 4. Evaluation and promotion rules

### E0 — Quick working-state gate

Run on first setup and every model/profile/runtime change. Target a short run; timing is a goal, not permission to skip failures.

Cover:

1. Launch, model identification, and tool-call round trip.
2. Read, patch, test, and verify a small repository.
3. Preserve unrelated dirty files.
4. Diagnose and repair a seeded failure.
5. Reject a fabricated “tests passed” report.
6. Preserve a failure buried in long output.
7. Compact, restart, and recover required facts.
8. Escalate repeated no-progress attempts.
9. Complete and verify a real external-harness assignment.
10. Confirm IRIS/Mavis isolation and recover from an unavailable service.

**Pass:** all mandatory cases pass with receipts. This means “working enough to use and improve,” not “fully optimized.”

### E1 — Routine regression comparisons

A small, versioned set of real task types: bug repair, feature addition, refactoring, dependency failure, memory retrieval, and verification honesty.

Run affected cases after changes and broader shards during daily idle work. Compare identical starting revisions and acceptance checks.

### E2 — Difficult workloads

Weekly resumable shards covering large projects, long contexts, interruptions, integration conflicts, adversarial repository text, stale memory, and weak-test shortcuts.

Public benchmarks are supplemental. Project-specific held-out work is primary; benchmark scores alone cannot establish useful project competence. SWE-bench provides reproducible infrastructure, but has platform and evaluation limitations that must be recorded. [Harness documentation](https://www.swebench.com/SWE-bench/reference/harness/)

### E3 — Model-specific optimization

Compare prompt structure, effort, context size, compaction, tools, helper models, quantization, and serving settings separately.

For MLX changes, freeze the workload, measure the current bottleneck, change one relevant factor, compare complete task outcomes, then measure again. No promotion based solely on tokens per second or a small isolated benchmark.

### E4 — Training and local decision models

Evaluate adapters and Jev-style replacements on unseen projects and time-separated examples, including calibration, false success claims, and regressions in ordinary coding.

All advanced suites support pause/resume, explicit shard selection, and recorded unfinished coverage. They must not turn into one mandatory all-at-once benchmark.

### Promotion contract

Track:

- Independently accepted requirements.
- Time to accepted completion.
- Cloud usage and spend per accepted objective.
- Rework, unnecessary escalation, and missed escalation.
- False completion claims and damage to unrelated work.
- Retrieval and summarization omissions.

A candidate must pass mandatory gates and improve the targeted outcome on held-out work. Repeat close results; keep inconclusive candidates inactive. Do not claim a reliable 1% improvement from a small noisy sample.

Independent verification precedes promotion. Apply changes between objectives, preserve the prior version, and roll back on critical regression. Mavis cannot alter protected evaluation tests, promotion rules, or authority boundaries to make its own candidate pass.

## 5. Profiles, retention, and long-term hygiene

### Preserve progress when models change

Identify models using architecture, weight revision or fingerprint, tokenizer, chat template, quantization, and runtime compatibility—not display name alone.

Keep separate profiles for:

- Mavis’s main coding model.
- Jev/decision models.
- Librarian.
- Output reader.
- Embedding and ranking models.

Switching A → B → A must restore A’s accepted configuration, experiments, and history exactly.

Family inheritance creates a **candidate copy** with provenance. It may reuse promising prompts or tool settings, but never automatically inherits adapters, capability claims, or passed evaluation status.

### Retain evidence without growing context endlessly

- Preserve full transcripts and tool outputs locally.
- Compress closed project archives after 30 days without activity.
- Verify reconstruction and checksums before removing the uncompressed duplicate.
- Keep active objectives and referenced evidence readily accessible.
- Do not automatically delete unique source evidence.
- Monitor storage pressure; remove reproducible caches first and surface capacity problems before evidence writes fail.
- Keep model weights, databases, raw logs, and secrets out of Git.

Unlimited retention does not mean unlimited prompt injection. Return cited, relevant material with expansion available. This follows the useful distinction between durable storage and carefully selected working context. [Context engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

### Keep the system maintainable

Every new capability needs one owner, a narrow interface, tests, a disable path, and a migration/rollback path. Avoid creating another service merely because a new model role exists.

Maintain architecture checks for forbidden dependencies, duplicated provider registries, and direct cloud calls bypassing the gateway. Review upstream Codex changes in small increments.

Delegate cleanup, stale-document repair, fixture maintenance, and archive checks to MiniMax or Mavis. Terra verifies cleanup receipts. Remove only task-owned disposable material after recovery and retention checks pass.

Sol may adapt implementation details when evidence exposes a better route. It must record the reason and preserve the required behavior, authority, and success criteria.

**The delivery priority is fixed:** working Mavis → trustworthy evidence and delegation → memory/helpers → continuous improvement → difficult workloads → local decision models → targeted training.
