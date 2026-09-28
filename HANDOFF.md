# Project handoff

## Objective
Owner-approved local P0: UR10e expert -> 10 demonstrations -> one GR00T-compatible
dataset -> actual ACT GPU checkpoint -> held-out closed-loop results -> real GR00T
adapter contract and truthful comparison. AGENTS.md and the owner's revised scope govern.

## Scope
E0 remains independent. No further Colab entitlement work; preserve its pinned runner.
The aerospace reference remained read-only. No public push, tag or release was performed.

## Acceptance criteria
Real deterministic simulation/rendering; at least 10 valid demos; one canonical v2
dataset; actual CUDA training/checkpoint reload; learned closed-loop held-out results;
real GR00T interface with unavailable execution explicit; reproducible compact evidence.

## Completed
- G1: official UR10e fixed-peg task, measured-pose/Jacobian controller, front RGB,
  contact monitoring, geometric success and deterministic seeded reset.
- G2: 10 successful expert episodes, 487 synchronized frames; official GR00T reader
  and action normalization round-trip passed. No Franka joint replay or separate ACT dataset.
- G3: maintained LeRobot ACT trained from scratch for 1,000 steps on RTX 4070 Ti;
  real checkpoint saved and strictly reloaded with matching finite action chunks.
- G4: actual checkpoint-controlled simulation completed on all 20 frozen held-out seeds.
  Expert: 20/20 successful. ACT: 0/20 successful, all COLLISION. This is a negative
  learned-control result, not a successful insertion or transfer claim.
- GR00TPolicy implements the official N1.7 NEW_EMBODIMENT checkpoint/policy/decoding
  path. Official observation batching and action round-trip passed; model execution
  and target-checkpoint validation remain separately unavailable.
- Comparison CSV/README table/plot generated from recorded episodes; every raw trace
  hash, executed/clipped action, timestamp, terminal outcome and matched initial state checked.
- Opened and inspected the actual held-out failure image, expert terminal image and plot.

## Current checkpoint
The local P0 execution chain is complete. Evidence is under artifacts/datasets/,
artifacts/training/act-10-001/, artifacts/eval/{expert-002,act-10-002}/ and
artifacts/environment/gr00t_local_contract.json. No live evaluation job remains.
Raw datasets, checkpoints, logs and videos are ignored but available locally.

## Decisions and reasons
- Keep Isaac Sim and the pinned ML .venv separate; a local stdio worker carries
  fresh RGB/state to actual ACT inference and returns physical action chunks.
- Execute the first action of each new 16-step chunk; record per-axis clipping.
- Seeds 1000-1019 were committed in 0392ba1 before ACT training. Seeds 0/1 are development.
- ACT's first long evaluation stalled at the seventeenth render reset. Two live
  py-spy stack samples and the unchanged Kit log confirmed SimulationApp.update in reset.
  Preserve act-10-001 as SIMULATOR_RESET_STALL and expert-001 as cancelled recovery evidence.
  Final runs use two fresh processes of 10 episodes for both policies. No model/seed tuning.
- Run metadata records the parent Git HEAD and exact source hashes. The final
  committed evaluator source matches those hashes, including the reset-isolation fix.

## Verification evidence
- scripts/local/setup.ps1: pinned clean ML environment and CUDA verified independently of Isaac Sim.
- C:\isaacsim\python.bat scripts/local/insertion.py --seeds 0 0 1 --output artifacts/runtime/insertion-03:
  repeated seed matched exactly; deliberate miss in collision-02 triggered physical collision.
- C:\isaacsim\python.bat scripts/local/insertion.py --seeds 10 11 12 13 14 15 16 17 18 19 --record --output artifacts/runtime/demos-01:
  10 successful episodes; actual images/physics timestamps remained synchronized at 15 Hz.
- .venv/Scripts/python.exe scripts/local/dataset.py --dataset datasets/ur10e_10 --upstream .external/Isaac-GR00T --report artifacts/datasets/validation.json:
  schema/official loader/official action round-trip PASS.
- .venv/Scripts/python.exe scripts/local/act.py --output artifacts/training/act-10-001:
  1,000 CUDA steps, 81.258 s, real changed weights, reload PASS. Model SHA256:
  957d617972c5ae2d37b3060c57223e330d191dee8becec98a06b00e29e015631.
- C:\isaacsim\python.bat scripts/local/evaluate.py --policy act --split held_out --output artifacts/eval/act-10-002:
  all 20 completed; same command with --policy expert and expert-002 completed 20 successes.
- .venv/Scripts/python.exe scripts/local/groot.py --report artifacts/environment/gr00t_local_contract.json:
  integration PASS; MODEL_COMPUTE_NOT_AVAILABLE; TARGET_CHECKPOINT_NOT_AVAILABLE.
- python scripts/local/compare.py: all 40 trajectories and initial-state matching verified;
  CSV/table/plot and actual 15 Hz video metadata generated from results.
- .venv/Scripts/python.exe -m unittest discover -s tests -v: 33 passed, no skips.
  E0 PASS text in unit-test logs refers to mocked temporary fixtures, not model execution.

## Not executed
Successful learned insertion, GR00T target fine-tuning/inference, cross-embodiment
transfer, FIRe source specialization, real UR10e deployment and fresh-machine validation.
The prior E0 inference failed; no real E0 PASS artifact is committed.

## Blockers
No local P0 execution blocker remains. GR00T runtime needs supported Linux GPU
compute and an adapted target checkpoint; its metrics stay N/A. Current ACT competence
is insufficient: failures are retained rather than relabeled or tuned on held-out results.

## Modified files
Canonical local simulator/data/learning/policy/evaluation scripts, targeted tests,
pinned dependencies/configs, compact execution evidence, README and this handoff.
No source data or model payloads are committed.

## Next concrete action
Owner overrides priority: recover ACT learned success before final media assembly.
0/20 baseline committed as 4204876. Exact-path expert already measured 20/20;
verify replay, tensors, conventions, distributions, then one-episode overfit.
Final media is blocked on learned success >0. FIRe candidate payloads (0,14,29,
three views) and pinned metadata are cached at artifacts/runtime/fire-inspection;
no source bridge or final media claim exists yet.

## Active ACT diagnosis
- Exact same evaluator/controller/detector expert baseline: 20/20.
- Replay seed 10: SUCCESS in 48 steps, state error <=3.65e-9; replay-002.
  replay-001 stopped because Isaac Python has no pandas; raw generator trace is
  used with canonical float32 casting, without installing ML packages in Isaac.
- Real checkpoint audit: 487 demo frames; train/adapter input tensors bit-exact;
  label/expert comparison and normalization inverse pass. Temporal execution K=1.
- All 20 baseline failures classified I with two observed approach patterns;
  exact optimization cause remains under investigation, not asserted.
- Running act-overfit-001: episode 0 only (seed 10), 5000 steps, same architecture
  and full-training-set normalization; no extra data or changed success detector.
