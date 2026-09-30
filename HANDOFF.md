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
Final paired measured results:10 demos target-only5/20 vs source-informed0/20;
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
Final65s main and12s hero rendered at1920x1080/30fps; poster generated. Full decoding
and representative encoded frames were inspected. Editing package stores source/sim
clips, overlays, captions, evidence, credits and file hashes. Native review remains pending.

### 2026-09-29 source-transfer diagnosis and third track (uncommitted)
- Existing audits showed full-transfer FIRe 25/50 checkpoints output constant actions (pred std ~1e-8).
- diag-fire0-50-001: FIRe path with 0 source updates is bit-identical to act-50-001 (e370cdcb…): code path innocent.
- diag-fire1000-50-001: reproduces act-fire-50-001 bit-identically (b344d21b…). After source training the
  transformer encoder/decoder are input-invariant on UR10e frames, responsive on FIRe frames; backbone/state
  projection stay responsive. Evidence: artifacts/diagnostics/source-collapse-001/. Non-recovery mechanism UNKNOWN.
- act.py `--source-transfer backbone` (default `features` preserves history): keep only source-trained ResNet.
- act-firebb-{10,25,50}-001 trained, open-loop audits not collapsed; held-out: 0/20 each, all COLLISION,
  mean lateral 3.24 / 2.06 / 1.13 mm vs target-only 2.02 / 1.74 / 0.44 mm.
- compare.py supports three tracks and a lateral-error panel; nine-condition study COMPLETE and verified.
- media/edit_manifest_v2.json + build.py `--manifest` render media/output/main_v2.mp4 (70 s) with a diagnosis shot;
  v1 manifest/main/hero/poster unchanged. v2 native desktop review NOT_RUN.
- Stale field: fire_task_audit.json `source_informed_design_status` still reads NOT_IMPLEMENTED.
- media/linkedin.py renders media/output/linkedin_4x5.mp4 (1080x1350, 44 s) + receipt JSON. Ecosystem context
  clips from .external/Isaac-GR00T demo_data (git lfs pulled): DROID, BridgeData V2, Fractal/RT-1, all CC BY 4.0
  (DROID paper; BridgeData site; open_x_embodiment repo), labeled "not trained or run here". LIBERO/SO-101 excluded
  (license unverified). Full decode PASS, 8 encoded frames reviewed; native desktop review NOT_RUN.

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
- Latest full regression:38 tests passed in16.085s, artifacts/runtime/final-media-tests.log.
- artifacts/eval/data_efficiency.json:COMPLETE, six fixed20-seed conditions.
- artifacts/media/capture_validation.json:four actual captures hash/decode verified.
- artifacts/environment/gr00t_local_contract.json: integration PASS, model/target checkpoint NOT_RUN.

## Not executed
Calibrated matched-task changes,
Native desktop verification of latest stages, GR00T target training/inference and fresh-machine reproduction.

## Blockers
Source units/frame/tool geometry and dataset-to-code provenance need verification. ACT achieved
reproducible held-out success; broader robustness remains open; the required source-informed comparison is complete with no improvement for this recipe. GR00T runtime requires supported compute and adapted checkpoint;
that does not block local experiments. No real E0 PASS exists.

## Modified files
Canonical AGENTS/README/HANDOFF reconciled; H1 training/evaluation evidence preserved;
open-loop diagnostic plot uses physical action bounds for honest scale comparison.

## Next concrete action
Native desktop review of the completed main/hero/poster is the remaining media gate.
Files:media/output/main.mp4,hero.mp4,poster.png. Current file review and package hashes
are in media/edit_manifest.json and artifacts/media/delivery_validation.json.
Latest continuation verified live Computer Use app inventory, but reading the existing
Windows Explorer window again returned "Computer Use app approval timed out". The
helper is available; app approval remains the same blocker. Wait for owner availability
to approve the app request before retrying. Do not substitute another transport
or claim native review from FFmpeg/file inspection. No further training is required
for the completed six-condition study. Keep the negative source-informed result.
Antigravity was NOT_RUN after auto-review rejected external payload transmission;
no workaround was used. GR00T actual model execution remains COMPUTE_DEFERRED.
Fresh-machine reproduction and publication are not complete; no push/tag was performed.
