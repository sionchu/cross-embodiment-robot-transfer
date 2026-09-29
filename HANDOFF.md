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
The final50-demo target-only ACT completed20/20 fixed held-out successes, with unchanged
task/controller/detector/seeds and verified raw transition hashes/terminal predicates.
Checkpoint e370cdcbfaf844c15a47770a6212dd7720270e73794737fb8920e93029eaf8fa.
All six training/evaluation conditions completed and verified. Final source-informed50
is0/20; no source-data benefit was observed for this recipe. The canonical six-condition
comparison is COMPLETE; original seeds, controller and detector remain unchanged.
Paired measured results so far:10 demos target-only5/20 vs source-informed0/20;
25 demos0/20 vs0/20. Preserve the original0/20 debugging baseline and every failed run.
The50-demo dataset has2441 frames; original25 episodes are byte-identical prefixes.
Both50-demo checkpoints passed actual2441-frame action audits and representative input
train/adapter tensor equality, plus strict checkpoint reload.
Native desktop verification remains incomplete after app approval timeout. Source footage,
actual data and encoded trim contact sheets have been inspected through file tools.
Media manifest retains fixed capture seeds1000-1007 and1005; it now selects the final50-demo
target-only checkpoint, explicitly documented without changing any experiment or denominator.
All four capture cameras executed; expert and three learned captures succeeded.
Hashes, full decode, unchanged initial conditions and zero-time capture invariance passed.
The first65s main cut rendered and was visually reviewed; targeted final refinements are rendering.

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
- Latest evaluator/media regression:38 tests passed in9.393s, artifacts/runtime/media-regression-tests.log.
- artifacts/eval/data_efficiency.json:COMPLETE, six fixed20-seed conditions.
- artifacts/media/capture_validation.json:four actual captures hash/decode verified.
- artifacts/environment/gr00t_local_contract.json: integration PASS, model/target checkpoint NOT_RUN.

## Not executed
Calibrated matched-task changes,
Final video, native desktop
verification of latest stages, GR00T target training/inference and fresh-machine reproduction.

## Blockers
Source units/frame/tool geometry and dataset-to-code provenance need verification. ACT achieved
reproducible held-out success; broader robustness remains open; the required source-informed comparison is complete with no improvement for this recipe. GR00T runtime requires supported compute and adapted checkpoint;
that does not block local experiments. No real E0 PASS exists.

## Modified files
Canonical AGENTS/README/HANDOFF reconciled; H1 training/evaluation evidence preserved;
open-loop diagnostic plot uses physical action bounds for honest scale comparison.

## Next concrete action
Finish review of the refined65s main cut, then generate and inspect the12s hero/poster.
Four camera captures succeeded; actual main evaluation videos supply the first-eight-seed
wall. Main rendering is reproducible with media/build.py; output files stay ignored.
The original post-close HERO serialization issue is preserved and fixed by saving before
SimulationApp.close.38 regression tests passed. Review the actual encoded final frames
and keep native desktop verification separate: Computer Use app approval timed out.
Antigravity was NOT_RUN after auto-review rejected external payload transmission;
no workaround was used. GR00T actual model execution remains COMPUTE_DEFERRED.
