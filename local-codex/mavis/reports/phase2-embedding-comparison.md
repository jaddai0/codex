# Phase 2 embedding comparison — 2026-09-23

Status: **partial**. Both local model arms completed on a fixed Mavis code corpus, and the small model passed source-freshness canaries. Installed Mavis, full-project indexing, and the Phase 2 end-to-end work gate remain open.

## Frozen task and observations

- Source revision: `384ad8bc62bde23f2c459f7306fdb7214d38da01`.
- Corpus: 70 tracked `local-codex` text/code files, 305 passages of up to 48 lines / 8192 characters, copied without changing source bytes. Sixteen natural-language coding questions each name an expected source file. The corpus and cases are retained under `~/.local-codex/mavis-service/diagnostics/phase2-embedding/` with file hashes.
- Both MLX arms used model batch size one. A direct Qwen 0.6B canary with batch size two produced all-NaN values for a padded input; each passage was finite alone. This is recorded in `qwen-0.6b-probe.*`. The installed 8B adapter used existing local Torch/Torchvision dependencies and cleared incompatible cached text positions between calls; the processor and 16-passage canaries passed without editing the shared oMLX runtime.
- The IRIS bot and its two models were temporarily unloaded for each run and restored. The final handoff receipt reports both original IRIS models loaded, Mavis unloaded, and no restoration errors.
- A separate real-model run verified that vectors stay on their branch, changed-file exact lookup works before reindexing, stale vectors disappear, and rename/deletion update the vector index. Its six steps and independent IRIS restoration receipt are retained as `freshness.json` and `freshness-handoff.json`.
- An independent read-only audit rehashed all 70 corpus files and all 16 expected-file references against the pinned Git revision and recomputed the saved retrieval scores and timing medians. It qualified the comparison as a direct-MLX, fixed-corpus result, not installed Mavis acceptance. Full audit: `/Users/dustinpainter/.Codex/reports/mavis-phase2-embedding-comparison-independent-audit-2026-09-23.md`.

| Arm | Expected file in top 5 | Warm query median | Initial passage index | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| Exact lookup, fixed corpus | 0/16 | 0.091 s | 0.138 s | — |
| Qwen3 Embedding 0.6B | 12/16 | 0.281 s | 10.7 s | 1.96 GB |
| Installed Q3 8B VL Embedding | 13/16 | 0.632 s | 98.2 s | 9.83 GB |

Exact lookup on the full pinned repository also returned 0/16 expected files in the top five, with a 0.959 s median warm query. These questions were deliberately phrased as natural-language descriptions rather than exact substrings.

## Source-evidence audit

The file metric is useful for an agent that opens the retrieved file. It does not prove that the displayed passage contains the answer. After the runs, I marked the implementation line ranges for the 16 questions using the pinned source tree. Only 5/16 Qwen 0.6B top-five passages and 4/16 installed 8B passages overlapped those ranges. Several correct-file hits were module headers or docstrings. This **post-run diagnostic is not a preregistered quality metric**; the answer-span file and both raw hit lists are retained beside the comparison. A vector result is a source candidate for further reading, not a verified final citation.

The smaller model is the provisional choice for the optional index canary: it cleared the 50% file-retrieval floor, was within one question of the 8B model, and used much less memory and index time. It was better on the post-run passage diagnostic, though the small sample does not establish a general quality advantage. Exact lookup remains available without embeddings and after provider failure. No production default has been switched by this comparison.

The bandwidth monitor reported other GPU activity from WindowServer and a virtual graphics process in both model runs. Its readings do not support a strict speedup claim. The measured end-to-end times and memory are retained as observed local costs, with this contamination noted. The local 8B result names a mutable model label rather than a weight hash, and the saved result JSON does not pin the loaded Mavis implementation commit. Neither arm is ready for a reproducible production promotion decision.

## Evidence and open gates

- Frozen inputs: `~/.local-codex/mavis-service/diagnostics/phase2-embedding/cases.json`, `corpus.json`, `fixture-cases.json`.
- Complete results: `exact-baseline.json`, `exact-fixture-baseline.json`, `qwen-0.6b.json`, `installed-8b.json`, `answer_spans.json`, arm stdout/stderr logs, and `handoff-result.json` in the same directory.
- MLX constraint record: `constraint_record.json` in the same directory. Its source-freshness row now passes; task closeout remains partial until installed integration has evidence.
- Independent source review accepted hash-bound passage citations and exact fallback at commit `50e7f68da`; source tests passed 148/148. This is not installed acceptance.
- Next gates: run integrated installed retrieval with embeddings enabled and disabled; prove the helper/librarian work and Phase 2 completed-task benefit.
