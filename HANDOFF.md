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
G2 canonical dataset is verified: 10 successful episodes, 487 frames. Pinned official
GR00T loader and action round-trip PASS are in artifacts/datasets/validation.json.
This is scripted expert/data evidence, not learned control. Fixed held-out seeds
1000-1019 are in configs/seeds.json before ACT training; 0 and 1 are development only.
No E0 PASS artifact is committed; GR00T remains independent and compute-deferred.

## Decisions and reasons
- Local ACT establishes real learning evidence within available compute.
- E0 gates actual GR00T runtime claims/fine-tuning only.
- Minimum front RGB + target-relative EEF pose; explicit six-dimensional task-space deltas.
- One canonical dataset feeds both maintained ACT and real N1.7 interfaces.
- Keep exact upstream/model pins and existing E0 failure diagnostics.

## Verification evidence
- scripts/local/setup.ps1: exit 0; locked environment, torch 2.9.0+cu128 and RTX 4070 Ti CUDA.
- C:\isaacsim\python.bat scripts/local/insertion.py --seeds 10 11 12 13 14 15 16 17 18 19 --record --output artifacts/runtime/demos-01:
  exit 0; all 10 SUCCESS, no contact, 487 distinct per-episode RGB frames total.
- .venv/Scripts/python.exe scripts/local/dataset.py --dataset datasets/ur10e_10 --upstream .external/Isaac-GR00T --report artifacts/datasets/validation.json:
  PASS, official_loader PASS, official_action_roundtrip PASS.
- Dataset SHA256: cd7fb1e8ccc08d37fef7cdc405ceb75db22fe7d9f88dde461bbeb7523092e017.
- .venv/Scripts/python.exe -m unittest discover -s tests -v: 22 passed, including
  10 corruption tests against copies of the actual dataset. E0 PASS lines in these
  unit tests are temporary mocked fixtures, not actual GR00T inference evidence.
- Opened a decoded middle video frame; actual UR10e approach and fixture were visible.
- Windows TorchCodec requires shared FFmpeg DLL discovery; pinned 7.1.1 plus a local
  os.add_dll_directory handle resolved decoding without modifying upstream/Isaac Sim.
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
ACT training and learned closed-loop evaluation. GR00T fine-tuning and real UR10e deployment. No cloud rerun.

## Blockers
No E0 prerequisite blocks local P0. GR00T execution remains compute-deferred and the
last recorded inference failed. Local data/ML environment has working CUDA; training feasibility is not yet measured.

## Modified files
G2 slice: recorder, canonical exporter/validator, corruption tests, seed/modality
configs, isolated ML setup and dependency lock, source manifest, dataset evidence,
README/HANDOFF and the root-only raw-dataset ignore rule.

## Next concrete action
Train maintained LeRobot ACT on the single datasets/ur10e_10 dataset using .venv.
Record actual CUDA training/loss, checkpoint and reload before connecting inference
to Isaac Sim. Keep the two Python environments isolated. Then execute fixed held-out
closed-loop evaluation and implement/validate the real GR00T adapter contract.
