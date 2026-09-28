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
- Reconciled obsolete serial E0 dependencies in AGENTS.md and README (de145e2).
- Implemented official UR10e fixed-peg scene, measured-pose/Jacobian controller, front RGB,
  contact monitoring, geometric success and deterministic seeded reset.
- Actual expert smoke seeds 0, 0, 1 succeeded; repeated results matched exactly.
- Deliberate 15 mm miss terminated COLLISION at 98.3386 N; reviewed actual RGB frames.

## Current checkpoint
G1 local simulator smoke is verified; next is synchronized per-frame demonstration
capture and the single GR00T v2 export. Compact evidence and three inspected frames
are in artifacts/simulation/. This is scripted expert evidence, not learned control.
No E0 PASS artifact is committed; GR00T remains independent and compute-deferred.

## Decisions and reasons
- Local ACT establishes real learning evidence within available compute.
- E0 gates actual GR00T runtime claims/fine-tuning only.
- Minimum front RGB + target-relative EEF pose; explicit six-dimensional task-space deltas.
- One canonical dataset feeds both maintained ACT and real N1.7 interfaces.
- Keep exact upstream/model pins and existing E0 failure diagnostics.

## Verification evidence
- python -m unittest discover -s tests -v: 12 passed (9 E0 wrapper, 3 task predicate/expert checks).
- C:\isaacsim\python.bat scripts\local\insertion.py --seeds 0 0 1 --output artifacts\runtime\insertion-03:
  exit 0; 3 successful expert trials, 50 control steps each; repeated seed exactly reproduced.
- Same runner --seeds 0 --collision-check --output artifacts\runtime\collision-02:
  exit 0; physical COLLISION at step 40 (negative test passed).
- artifacts/simulation/g1_smoke.json records actual environment, config, source digests,
  outcomes and image hashes. Raw-source and LF-normalized digests distinguish Windows line endings.
- Opened and inspected start, success and collision PNGs; no synthetic media used.
- Runtime warns about the official asset's ee_joint initial disjoint transform; the reset
  settles before measurement, and measured repeated-state control succeeded. This warning
  is retained in ignored runtime logs; no hardware or force-control fidelity is claimed.
- Full simulator source and staged documentation diff reviewed; git diff --cached --check passed.
- Staged source digest and committed image digests match the measured evidence.

## Not executed
Synchronized demonstration dataset generation/export, ACT training and learned
closed-loop evaluation. GR00T fine-tuning and real UR10e deployment. No cloud rerun.

## Blockers
No E0 prerequisite blocks local P0. GR00T execution remains compute-deferred and the
last recorded inference failed. Local simulator smoke runs; ML environment/training feasibility is not yet measured.

## Modified files
scripts/local/insertion.py, scripts/local/README.md, tests/test_insertion.py,
artifacts/simulation/g1_smoke.json and inspected PNGs, README.md, HANDOFF.md.

## Next concrete action
Record at least 10 successful episodes with RGB/state/action synchronized at 15 Hz,
export one pinned GR00T LeRobot v2 dataset and validate it before ACT smoke training.
Freeze held-out seeds before model selection/evaluation; seeds 0 and 1 are development
seeds and must not be presented as held-out learned-policy evaluation.
