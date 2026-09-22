# E0 installed small-repository repair

Date: 2026-09-22. This is one installed candidate's evidence, not full E0 acceptance.

Fixture: `~/.local-codex/mavis-service/evaluations/e0/tasks/78371f290bd84ec594dd28ccfb700a08/manifest.json`.

The frozen test failed before Mavis (`12 != 8`). After the approved sequential model handoff, the installed Desktop launcher ran Mavis in the fixture checkout. Its JSON event stream records reads of the code and tests, a `file_change` for `package/pricing.py`, and a completed turn. Mavis changed `return subtotal + discount` to subtraction. A separate host test command exited zero and reported two passing tests. The pre-existing `user-notes.txt` edit retained its manifest SHA-256; Git showed exactly those two modified tracked files and no untracked file. IRIS's Qwen model was reloaded and Mavis's unloaded after the run.

A separate native Terra CLI review ran read-only, checked the baseline and diff, ran the exact tests, compared the protected note hash, and returned `ACCEPT`. Its full JSON event stream and final review are in the same private task directory.

`E0Evaluator` now rechecks those facts for the small-repository, dirty-work-preservation, and seeded-failure-repair cases. The host `installed-run.json` binds the evidence logs to hashes of the installed core binary, Desktop launcher, and Python service package. A changed installed candidate blocks reuse of this task evidence until the new candidate is run. The other E0 cases remain independent and must pass separately.
