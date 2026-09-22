# Objective gateway verifier forgery fix

Date: 2026-09-22

## Scope

This branch repairs the accepted-transition trust boundary. It does not make a
model call and does not alter the live Mavis or IRIS services.

## Change

`objective verify` still records the legacy verifier document so existing
non-accepted workflows can retain it. That document is no longer evidence for
the `accepted` transition.

`objective gateway-verify OBJECTIVE_ID WORKER_JOB_ID` asks only the configured
`model-gateway` MCP server for `harness_job_status`. Mavis writes its own
gateway-status receipt under the objective evidence root. The receipt retains:

- the exact worker job ID and terminal worker receipt;
- a distinct Terra verifier job and its accepted status;
- gateway target, report, verifier-verdict, and evidence hashes;
- the evidence revision and exact retained host receipt paths and hashes.

Before the accepted transition, Mavis verifies the local receipt hash, each
host receipt, revision, acceptance-check coverage, and the retained receipt
set. It then asks the configured gateway again and requires an identical,
still-accepted status. Missing configuration, process failures, malformed MCP
responses, failed jobs, changed status, and missing distinct Terra verification
all stop acceptance.

The gateway status must also contain a `mavis_binding` projected from its
durable worker assignment. Mavis compares its objective ID, checkout path,
starting and changed revisions, worker owner, owned paths, requirement IDs,
acceptance check IDs, assignment hash, report hash, and target hash against the ObjectiveStore
record and retained host evidence. A completed accepted gateway job for another
objective, checkout, or revision is rejected.

## Tests

`python3 -m unittest discover -s tests -v` passed: 47 tests.

`python3 -m compileall -q mavis` passed.

`git diff --check` passed.

Focused coverage proves that a model-supplied accepted verifier JSON cannot
move an objective to accepted, that a bound gateway/Terra fixture can, that a
worker cannot be its own Terra verifier, and that a gateway outage stops the
accepted transition.

The adversarial gateway fixture also proves that mismatched objective ID,
checkout, revision, and worker owner cannot create a Mavis gateway receipt.

## Boundary

This is source and fixture evidence only. No gateway process or model was
started, and no live worker or Terra job was used. The receipt binds the worker
job returned by the configured gateway to Mavis host evidence; the gateway
remains the authority for the worker-to-Terra native-job chain.

The gateway must ship the matching `mavis_binding` status projection before
this branch can accept any gateway worker. Until then, the new check fails
closed, which is intentional.

Gateway contract dependency verified at `71bc4704daa3c2e08acfc6645e04fd83aa31aaef`.
It persists and validates the Mavis objective fields before exposing the
projection only for an accepted native job.

## Repository state

Worktree: `/Users/dustinpainter/Dev-Projects/local-codex-mavis-objective-gateway-verifier`

Branch: `codex/mavis-objective-gateway-verifier`

Base: `10308567ce9bb5174c4c26c1680b12abc2d7d9a8`

No commit or push was made, pending review instruction.
