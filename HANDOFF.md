# Project handoff

## Objective
Real FIRe Franka experience → relative EEF representation → UR10e → measured target
sample efficiency. Execute target-only and source-informed ACT at 10/25/50 target demos,
with a real GR00T N1.7 NEW_EMBODIMENT integration path and evidence-linked technical video.
The owner's final 2026-09-29 direction is reconciled in AGENTS.md; it supersedes older P0 scope.

## Scope
Existing canonical dataset, policy API and artifacts remain SSOT. Local RTX 4070 Ti and
Isaac Sim are independent of E0. No Colab entitlement work, extra architectures or final
video before evidence. Aerospace reference stays read-only. No external push/tag/release.

## Acceptance criteria
Pinned actual source footage/data and task audit; verified relative EEF transform;
reproducible learned held-out success; two executable tracks at 10/25/50 with identical
protocol; structured measured curves; truthful GR00T status; inspected source/simulator/
training/rollout/evaluation/video receipts and reproducible claim-linked media.

## Completed
- Official UR10e fixed-peg simulation, deterministic reset, contact detector and front RGB.
- Ten expert demos (seeds 10-19), 487 synchronized frames; canonical GR00T v2 reader validated.
- Real maintained LeRobot ACT CUDA training and strict checkpoint reload.
- Preserved fixed held-out baseline: expert 20/20, ACT 0/20, all collisions.
- Same-path replay seed 10 succeeds in 48 steps; state error <=3.65e-9.
- Actual train/eval input tensor equality, normalization inverse, expert/label comparison,
  action/frame/chunk audit and per-episode baseline failure classification.
- One-episode ACT overfit: ImageNet initialization alone changes same-seed result from
  COLLISION to SUCCESS (48 steps, 0 N contact, 0.309 mm lateral error). Training condition
  only; not held-out success. Actual end image inspected.
- Real GR00TPolicy official observation/action contract validated, model execution NOT_RUN.
- FIRe pinned metadata and candidate episodes 0/14/29, all three views, cached and hash-verified.
  Selected actual wrist clip and generated pose trajectories with reconstruction checks;
  task audit remains ANALOGOUS_TASK, with related-code dimensions explicitly unconfirmed.

## Current checkpoint
Phase-balanced ACT completed5000 steps; development0/1 both collided (1.80/2.18mm final
lateral error). Both initially aligned within tolerance at steps11/12, then small biased
lateral actions accumulated. This is classH approximate behavior with lost alignment;
rotation is within tolerance. No environment/controller/detector change. Next controlled
run act-10-lr-001 is ACTIVE: main optimizer LR1e-4→1e-5, everything else unchanged from
phase-balanced. The current learning-rate hypothesis is not yet a proven correction.
No new held-out evaluation or target demonstrations were generated.
FIRe candidate frames inspected; episode29 wrist selected, pinned hashes and derived
trajectories stored. Tasks remain ANALOGOUS_TASK because physical source geometry/frame
provenance is unresolved. No source-informed training or data-efficiency claim exists.

## Decisions and reasons
- Preserve original 0/20 and every later failed run. Task/controller/detector unchanged.
- Isolated ML .venv and Isaac Python communicate via local stdio worker; no expert fallback.
- K=1 receding horizon at 15 Hz; prediction horizon follows checkpoint (16 or diagnostic 1).
- Held-out seeds 1000-1019 frozen in 0392ba1; development 0/1; never select better seeds.
- Two fresh simulator processes of ten episodes avoid measured seventeenth-reset Kit stall.
- Random ResNet with FrozenBatchNorm produced saturated image features; controlled official
  ImageNet initialization fixed one-episode overfit but did not fix all-ten learning.
- Source Hub names Panda; related FIRe code describes FR3. Verify provenance rather than
  assert exact hardware. Source normalized actions are not physical delta units by default.

## Verification evidence
- artifacts/simulation/g1_smoke.json: repeated reset/render/control and collision evidence.
- artifacts/datasets/validation.json: canonical schema/official loader/action round-trip PASS.
- artifacts/eval/{expert-002,act-10-002}/: complete fixed 20-seed matched baseline.
- artifacts/eval/replay-002/: successful same-path demonstration replay.
- artifacts/diagnostics/act-baseline-001/: actual 487-frame tensor/action/failure audits.
- artifacts/diagnostics/recovery.json: single-factor one-episode correction, no held-out claim.
- artifacts/training/act-overfit-002/ and artifacts/eval/act-overfit-train-003/: learned
  training-condition success, checkpoint 88112479fbe19be29d527e32288dac724c2b765bea18e4738bbfed01278fa8e9.
- artifacts/training/act-10-imagenet-001/ and artifacts/eval/act-10-imagenet-dev-001/: 0/2.
- artifacts/training/act-10-h1-001/ and artifacts/eval/act-10-h1-dev-001/: 5000 CUDA steps,
  reload PASS, 0/2 collisions; checkpoint 090185e98a540e369d44fcc5676f3152f8166d0048e38bdbae42f7ddcd245d0c.
- .venv/Scripts/python.exe -m unittest discover -s tests -v: 35 passed, no skips;
  artifacts/runtime/source-regression-tests.log. E0 PASS strings are mocked fixtures only.
- artifacts/environment/gr00t_local_contract.json: integration PASS, model/target checkpoint NOT_RUN.

## Not executed
Verified source-to-target learning bridge, matched-task changes, source-informed training,
25/50 datasets, learned held-out success, data-efficiency curves, final video, native desktop
verification of latest stages, GR00T target training/inference and fresh-machine reproduction.

## Blockers
Source units/frame/tool geometry and dataset-to-code provenance need verification. ACT general
learning remains unresolved. GR00T runtime requires supported compute and adapted checkpoint;
that does not block local experiments. No real E0 PASS exists.

## Modified files
Canonical AGENTS/README/HANDOFF reconciled; H1 training/evaluation evidence preserved;
open-loop diagnostic plot uses physical action bounds for honest scale comparison.

## Next concrete action
Finish act-10-lr-001, inspect actual predictions and evaluate unchanged development0/1.
If successful, run all20 frozen held-out seeds and repeat a successful seed independently.
Otherwise inspect drift without changing detector or selecting seeds. Regression suite:
35 tests PASS in9.716s (artifacts/runtime/phase-regression-tests.log); E0 fixture outputs
are mocked tests, not actual GR00T inference. Latest measured closed-loop plot inspected
with file viewer; native-desktop verification remains separate and incomplete.
Antigravity was NOT_RUN after auto-review rejected external payload transmission;
no workaround was used. Local diagnostics continue.
Source regeneration: python scripts/local/fire.py --download, then offline command without
--download. Offline clip SHA reproduced; two transform tests and full35-test suite passed.
Native desktop source inspection is NOT_COMPLETED: Computer Use app approval timed out.
