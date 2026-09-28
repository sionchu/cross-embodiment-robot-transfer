# Project handoff

## Objective
Complete the owner-approved local P0: UR10e expert -> 10 demonstrations -> canonical
GR00T-compatible dataset -> actual ACT GPU checkpoint -> held-out closed-loop results.

## Scope
AGENTS.md is canonical. E0 is an independent GR00T runtime gate. Preserve its pinned
runner; do not spend time on Colab entitlement or waiting for T4. Keep the aerospace
reference read-only. No public push or release.

## Acceptance criteria
Actual deterministic simulator and rendered verification; at least 10 valid successful
episodes; one GR00T LeRobot v2 dataset; real RTX 4070 Ti ACT training/loss/checkpoint
reload; checkpoint-controlled held-out evaluation; real GR00T adapter contract.
Unavailable GR00T execution must not acquire fabricated metrics.

## Completed
- Existing G0/E0 runner, pinned sources and sanitized evidence retained.
- Owner directly approved switching to independent local work on 2026-09-28.
- Reconciled obsolete serial E0 dependencies in AGENTS.md and README.

## Current checkpoint
P0-A planning reconciliation; simulator implementation is next. Baseline HEAD was
73f1eb2 and the tree was clean before this change. The latest README record reports
OFFICIAL_INFERENCE_FAILED after pinned T4 installation/gated access. The older HF_TOKEN
blocker in this handoff was stale. No E0 PASS artifact is committed; cloud execution
was not revalidated in this planning change.

## Decisions and reasons
- Local ACT establishes real learning evidence within available compute.
- E0 gates actual GR00T runtime claims/fine-tuning only.
- Minimum front RGB + target-relative EEF pose; explicit six-dimensional task-space deltas.
- One canonical dataset feeds both maintained ACT and real N1.7 interfaces.
- Keep exact upstream/model pins and existing E0 failure diagnostics.

## Verification evidence
- Read complete canonical plan, revised owner attachment, README, handoff and E0 runner.
- git status --short: clean before changes; git log -5 confirmed 73f1eb2.
- python -m unittest discover -s tests -v: 9 tests passed (mocked E0 wrapper checks, not model inference).
- git diff --check: passed; full planning diff reviewed.

## Not executed
Local simulator, demonstration generation, dataset export, ACT training and learned
closed-loop evaluation. GR00T fine-tuning and real UR10e deployment. No cloud rerun.

## Blockers
No E0 prerequisite blocks local P0. GR00T execution remains compute-deferred and the
last recorded inference failed. Local simulation/training feasibility must be measured.

## Modified files
AGENTS.md, README.md, HANDOFF.md.

## Next concrete action
Build and run the smallest deterministic official UR10e insertion scene on local Isaac Sim.
