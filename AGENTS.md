# Cross-Embodiment Robot Transfer — Codex Execution Plan

Status: canonical project instruction
Date: 2026-09-29
Owner intent: build an evidence-first public portfolio/research project, not a demo that merely name-drops VLA models.

## New Codex chat bootstrap — read this first

This file is the **single source of truth for a brand-new Codex chat**. Assume the new chat has no access to the prior ChatGPT discussion, no memory of why this project was chosen, and no hidden context beyond files and tools it can inspect.

Do **not** ask the user to restate the previous discussion. Everything required to start P0 is documented here. When something is not documented, verify it from the repository, the local machine, or the official/public source instead of inventing context.

Repository to open and modify:
- `D:\Workspace\01_Robotics\cross-embodiment-robot-transfer`
- This repository starts intentionally minimal. At the initial committed state, `AGENTS.md` is the canonical artifact and implementation files may not exist yet.

Read-only implementation reference:
- `C:\aerospace-olp-ros2-isaacsim`
- It contains already-validated local examples for NVIDIA official UR10e loading, cuMotion/RMPflow control, Isaac Replicator RGB capture, deterministic tests and video recording.
- Inspect and reuse the smallest generic pieces where useful, but do not modify this reference repository and do not create a runtime dependency on its absolute path.

Available execution environments:
- Local Windows workstation: RTX 4070 Ti 12 GB, Isaac Sim 6.0.1 at `C:\isaacsim`, Python 3.12, `uv`, Git, Git LFS, FFmpeg, ROS 2 workspace available.
- Cloud compute: E0 inference requires an authorized Linux x86_64 NVIDIA CUDA GPU with at least 16 GB-class VRAM; the later GR00T fine-tuning stage requires 40 GB+ VRAM. Google AI Pro's bundled Colab CCU benefit is not available to this South Korea account as of 2026-09-28, but free-tier T4 may still satisfy the E0 inference gate when Colab allocates it.
- Hugging Face authentication / gated-model approval may require the user to perform the login or approval step. Do not store secrets in the repository.

### Mandatory first actions in a new Codex chat

1. Read this `AGENTS.md` completely before changing anything.
2. Run `git status --short`, `git log --oneline --decorate -5`, and inspect the current tree.
3. Confirm the repository is still at the expected minimal/bootstrap state; if it has progressed, continue from the actual latest state instead of replaying this plan from the beginning.
4. Re-verify time-sensitive upstream GR00T/LeRobot interfaces against the pinned/current official sources before implementing their schemas or CLI commands.
5. Start at the earliest incomplete local P0 gate below. E0 is independent and must not block target-task work.
6. Make the smallest coherent change, run its verification, review the diff, then commit it. Do not scaffold the whole future folder tree up front.

### Current execution priority (owner final goal, 2026-09-29)

This direction replaces conflicting earlier P0 plans. Use this file as the canonical
plan and HANDOFF.md as the current execution checkpoint; do not append competing plans.
Public identity: **GR00T N1.7 × UR10e — Real-World Cross-Embodiment Robot Learning**.
Technical subtitle: Real Franka demonstrations → Relative EEF representation → New
UR10e embodiment → Closed-loop data-efficiency evaluation.

Execute in this order:
1. Reconcile canonical documents with this goal.
2. Inspect actual FIRe media/data and record pinned source provenance.
3. Audit Franka versus UR10e task equivalence; prefer a documented MATCHED_TASK.
4. Resolve the preserved ACT 0/20 baseline through the debugging protocol below.
5. Pass one-episode overfit and obtain reproducible learned held-out success.
6. Execute the target-only 10-demo baseline and a technically valid source-informed
   10-demo comparison; then execute both tracks at 25 and 50 target demos.
7. Generate measured data-efficiency curves from structured evidence.
8. Validate GR00T N1.7 NEW_EMBODIMENT integration and its actual execution status.
9. Generate evidence-linked assets, assemble the technical video, inspect the actual
   rendered video through Remote Desktop and recheck every public claim.

FIRe inspection and the 10/25/50 comparison are required, not stretch work. Do not
wait for GR00T compute. E0 gates GR00T runtime/fine-tuning claims only. Preserve its
pinned runner; no further Colab entitlement work, quantization or extra architectures.
No external push, tag or publication is authorized by this plan.

### Learned-policy debugging gate

Preserve 0/20 as baseline evidence. Before adding demonstrations: exact same-path
expert → successful demonstration replay → actual observation tensors → action
normalization/frames/conventions/distributions → ACT chunk/replanning audit →
one-episode overfit → same-condition closed-loop test → proven root-cause correction
→ 10-demo retraining. Change one coherent factor at a time and retain failed runs.

Classify every failed episode with evidence: A invalid output; B scaling/normalization;
C coordinate frame; D action convention; E controller tracking; F temporal/chunking;
G observation mismatch; H approximate behavior misses tolerance; I task not learned;
J incorrect success detection. Do not guess a cause from aggregate success alone.
Expert must succeed reliably through the identical adapter/controller/detector before
policy tuning. Replay must approximately reproduce its recorded successful episode.

Use training/development seeds for diagnosis; do not select new evaluation seeds after
seeing results. Never replace policy actions with expert actions, intervene manually,
or relax detection for video. Any task change gets a new protocol identity and applies
to expert, both learning tracks and all data sizes. At least one reproducible held-out
learned success is required before final media; 50%/80% are useful goals, not invented
results. One training-condition success is not a held-out result.

## 0. Mission

Can real-world manipulation experience and an embodiment-independent action
representation reduce the target-robot data required to transfer a precision task?

**Real Franka experience → relative task-space EEF representation → UR10e adapter →
actual training → closed-loop evaluation → measured target-data efficiency.**

Foundation-model extension: how does this pipeline map onto NVIDIA GR00T N1.7's
cross-embodiment / NEW_EMBODIMENT workflow? GR00T remains the foundation-model target;
ACT is the executable local policy for the controlled source-informed comparison.
A valid negative transfer comparison is acceptable after pipeline debugging and the
required experiments; an unresolved 0/20 pipeline is not the final project result.

## 1. Scope decision

This is a **new standalone project**. Do not turn the existing aerospace repository into a generic research sandbox.

Reference-only local repository:
- `C:\aerospace-olp-ros2-isaacsim`
- Reuse only small, generic and clearly attributable implementation ideas if needed: official UR10e asset loading, cuMotion/RMPflow control, Replicator RGB capture, deterministic smoke-test patterns.
- Do not modify that repository as part of this project unless explicitly requested later.
## 2. Clean-v0 rules

- Prefer one clear vertical slice over many half-finished features.
- Core scope is real FIRe source + relative EEF bridge + UR10e + two ACT tracks at 10/25/50 demos + fixed evaluation + GR00T integration.
- Do not add dashboards, agents, RAG, web backends, Kubernetes, ROS 2 orchestration, humanoids, or extra simulators unless a measured requirement appears.
- Do not implement a custom VLA architecture in P0.
- Do not add SmolVLA, π0.5, Diffusion Policy, or Isaac Lab Mimic to P0 merely for technology breadth.
- Do not claim sim-to-real, real UR10e validation, or industrial qualification.
- Do not claim that Franka actions are directly replayed on UR10e.
- Do not call source-data fine-tuning “transfer” until the target embodiment is evaluated in closed loop.
- Every public metric must come from a saved machine-readable result artifact.
- Fixed evaluation seeds must be chosen before final model comparison.
- Keep source datasets, model weights, caches, and secrets out of Git.
- No Hugging Face token, W&B key, Colab credential, or private path should enter committed files.
- New abstractions require evidence that at least two concrete call sites need them.

## 3. Current compute split

### Local Windows workstation
- Windows 11.
- NVIDIA RTX 4070 Ti, 12 GB VRAM.
- NVIDIA Isaac Sim 6.0.1 at `C:\isaacsim`.
- Python 3.12 available.
- `uv`, Git, and Git LFS available.
- Existing ROS 2 Jazzy workspace exists, but ROS 2 is **not required for P0**.

### Cloud compute
- E0 inference: use any authorized Linux x86_64 NVIDIA CUDA environment with at least 16 GB-class VRAM. A free-tier T4 is acceptable if it passes the measured VRAM gate.
- E5 fine-tuning: use at least 40 GB VRAM; A100/L40/H100 or equivalent is appropriate.
- First step must run `nvidia-smi` and record actual GPU model and VRAM.
- GR00T N1.7 official guidance: inference minimum 16 GB+ VRAM; fine-tuning minimum 40 GB+ VRAM. Default fine-tuning trains projector + flow-matching DiT action head and is documented as staying below roughly 35 GB peak VRAM.
- 2026-09-28 infrastructure finding: Google One support reported the bundled Google AI Pro Colab CCU benefit as available to Canada, Mexico and the United States, not South Korea. Treat zero bundled CCU on this account as current regional policy, not a provisioning bug. Do not use VPN/region spoofing or account-country workarounds.
## 4. Primary open model

### NVIDIA Isaac GR00T N1.7
- Repository: https://github.com/NVIDIA/Isaac-GR00T
- Base checkpoint: https://huggingface.co/nvidia/GR00T-N1.7-3B
- DROID checkpoint: https://huggingface.co/nvidia/GR00T-N1.7-DROID
- Gated backbone: https://huggingface.co/nvidia/Cosmos-Reason2-2B
- Hardware guide: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md
- Data preparation: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md
- New embodiment fine-tuning: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
- Policy / embodiment tags: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md
- DROID example: https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/DROID/README.md
- Deployment/server-client: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md

Important current facts to preserve in code/docs:
- N1.7 is approximately 3B parameters.
- N1.7 uses the gated `nvidia/Cosmos-Reason2-2B` VLM backbone.
- The action head remains a flow-matching DiT.
- Base-model pretrain tags include `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` and generic X-DOF variants.
- A custom target robot uses `NEW_EMBODIMENT` plus a modality config.
- Current reference GR00T training format is **GR00T LeRobot v2**, with an additional `meta/modality.json`.
- LeRobot v3 must currently be converted to v2 before GR00T reference training.

Do not silently substitute N1.6-era APIs or data layouts.
## 5. Primary public datasets

### P0-A — Real Franka peg insertion
Use as the canonical real-world source unless direct inspection proves it unsuitable:
- Dataset: https://huggingface.co/datasets/bhe1004/FIRe-VLA-GR00T-PegInsert-Real
- Task: contact-rich peg insertion.
- Domain: real.
- Hub metadata names Franka Emika Panda; verify physical robot provenance rather than inferring it from a label.
- Format: LeRobot / GR00T-oriented.
- Published card reports 30 episodes, 3,840 frames, 15 FPS.
- License shown on Hub: Apache-2.0.

Before using it:
1. inspect its actual `meta/info.json`, `meta/modality.json`, state keys, action keys, camera keys, episode lengths and license;
2. verify whether its action representation is compatible with an existing GR00T embodiment tag;
3. if it is not exactly compatible, treat it as its own source embodiment and fine-tune with an explicit modality config;
4. never force the DROID tag just because both robots are Franka.

### P0-B — DROID sanity/reference data
- LeRobot v3 mirror: https://huggingface.co/datasets/lerobot/droid_1.0.1
- GR00T ships a small `demo_data/droid_sample` specifically for zero-shot/open-loop validation.
- Current LeRobot card reports Franka, 95,658 episodes, 27,630,375 frames, 15 FPS, Apache-2.0.

Use DROID to verify official N1.7 installation and embodiment handling on the independent GR00T track.
### P1/P2 reference datasets — do not pull into P0 unless needed

**Stanford KUKA multimodal peg insertion**
- https://huggingface.co/datasets/lerobot/stanford_kuka_multimodal_dataset
- convenient metadata/reference mirror:
  https://huggingface.co/datasets/THULab/lerobot_stanford_kuka_multimodal_dataset
- KUKA contact-rich peg insertion; 3,000 episodes are reported by the mirror.
- useful for a later multi-source / multi-embodiment study.

**OmniReset → GR00T simulated UR5e peg insertion**
- dataset: https://huggingface.co/datasets/yqi19/gr00t_data_omnireset
- generation docs/code: https://huggingface.co/datasets/yqi19/omnireset_reproduce
- useful as a reference for GR00T-ready relative Cartesian action layout and large synthetic demo generation.
- do not use as the “real source” because it is simulation.

### Source/task matching audit

Inspect actual downloaded episode media, all useful views, RGB/state/action samples
and episode metadata. Select 2-3 candidate clips and one canonical episode/view.
Record dataset ID/revision, license, episode, camera, frame/time range, exact resolved
source file and SHA256. Cache only required media outside Git; retain attribution.
Do not invent MP4 URLs or substitute stock/simulated Franka footage.

Compare source and target peg/hole dimensions, clearance, insertion depth/axis,
initial relative pose, EEF/tool orientation, approach direction, workspace scale and
success condition. Classify SAME_TASK_INSTANCE, MATCHED_TASK, ANALOGOUS_TASK or
DIFFERENT_TASK. Prefer MATCHED_TASK; unknown measurements remain unknown and prevent
stronger claims. Related source code is not proof of the dataset's collection revision.
Derive trajectories from actual poses (or verified kinematics), with explicit SI units,
frames, quaternion ordering and relative rotation composition. Never relabel normalized
actions as metres/radians or project telemetry onto images without camera calibration.

### Source-use policy
- Prefer linking to public sources over copying them into this repository.
- If a public clip is used in the final video, keep it short, attribute it on-screen/end-card, and obey the source license.
- Never imply that public Franka/KUKA footage was collected by this project.
- Keep a machine-readable `references/source_manifest.json` once implementation starts.

## 6. Secondary model/algorithm references

### ACT baseline
- Paper: https://arxiv.org/abs/2304.13705
- Project: https://tonyzhaozh.github.io/aloha/
- Purpose here: a strong target-data-only imitation-learning baseline using action chunks.
- Prefer the maintained LeRobot ACT implementation instead of rewriting ACT from scratch.

### Flow Matching background
- Paper: https://arxiv.org/abs/2210.02747
- Purpose: explain the generative-action principle behind modern continuous action heads.
- Do not build a separate flow model in P0 because GR00T N1.7 already supplies a flow-matching DiT action head.
### SmolVLA — P2 only
- Blog: https://huggingface.co/blog/smolvla
- Paper: https://arxiv.org/abs/2506.01844
- Base model: https://huggingface.co/lerobot/smolvla_base
- LeRobot repository: https://github.com/huggingface/lerobot
- 450M compact VLA using a flow-matching action expert.
- Add only if P0 is finished and an additional low-compute open baseline materially improves the study.

### OpenPI / π0.5 — P2 only
- Repository: https://github.com/Physical-Intelligence/openpi
- Use only as a comparison/future-work reference unless there is time for a clean extra experiment.
- Do not let π0.5 setup displace the GR00T experiment.

### LeRobot standard
- Repository: https://github.com/huggingface/lerobot
- v3 docs: https://huggingface.co/docs/lerobot/lerobot-dataset-v3

Data-format decision:
- **P0 training SSOT:** GR00T-compatible LeRobot v2 because current N1.7 reference implementation requires it.
- **P1 interoperability:** deterministic export/conversion to LeRobot v3.
- Do not maintain separate ACT and GR00T datasets; runtime adapters may read the canonical v2 dataset.

## 7. Target embodiment and task

Target robot:
- NVIDIA official UR10e asset inside Isaac Sim.
- 6 revolute joints.
- New GR00T embodiment: `NEW_EMBODIMENT`.

Target task:
- generic precision **peg insertion** / connector-style insertion.
- do **not** make aerospace geometry the main story.
- peg starts already grasped/fixed to the end-effector in P0.
- P0 focuses on approach → 6D alignment → insertion → verify.
- picking the peg is P1, not P0.
### Target scene requirements
Keep geometry visually clear and physically deterministic:
- table / fixture;
- receptacle with explicit insertion axis;
- cylindrical or keyed peg;
- UR10e;
- simple fixed tool/holder;
- front camera;
- wrist camera if stable in P0; otherwise front camera first and wrist camera P1;
- collision geometry separate from decorative meshes.

Randomize only parameters that support the research question:
- receptacle XYZ offset;
- receptacle orientation offset;
- initial TCP pose;
- modest camera pose jitter;
- modest light intensity/temperature jitter.

Do not randomize textures, entire backgrounds, robot dynamics, friction, camera intrinsics, and lighting all at once in the first pass.

### Success definition
A rollout is successful only if all of the following are satisfied:
- insertion depth exceeds the predefined geometric threshold;
- final lateral/axial pose is within the task tolerance;
- no forbidden collision is observed;
- no timeout occurs;
- policy reaches a terminal success state.

Store the thresholds in versioned config, not scattered constants.

## 8. Canonical action representation

Never transfer Franka joint angles directly to UR10e.

Cross-embodiment bridge:
`source robot → relative task-space end-effector motion → target embodiment adapter/controller`.

P0 target action should follow the closest officially documented N1.7 `NEW_EMBODIMENT` representation after inspecting current examples:
- relative end-effector translation;
- relative orientation representation;
- optional gripper/tool channel only if the task actually uses it.
Document the six-dimensional delta encoding precisely. Verify official current N1.7 modality/action transforms so already-relative actions are not transformed twice.

Target observation candidates:
- `observation.state`: target-relative EEF pose; additional proprioception only when justified by evidence;
- camera: front RGB;
- optional wrist RGB;
- task language;
- any TCP pose used as a model input must be declared explicitly in modality metadata rather than hidden in custom code.

Recommended acquisition rate:
- begin at 15 FPS to align with common public real-data cadence and simplify comparison;
- change only if measured control stability requires it.

Recommended model image resolution:
- use the current N1.7 preprocessing contract; avoid needless high-resolution recording if it is immediately resized.

## 9. Research experiment matrix

### E0 — official installation sanity gate
On an authorized Linux NVIDIA GPU with at least 16 GB-class VRAM:
1. record `nvidia-smi`;
2. clone official Isaac-GR00T with submodules;
3. authenticate to Hugging Face without storing token in notebook output;
4. confirm access to `nvidia/Cosmos-Reason2-2B`;
5. run official DROID sample zero-shot open-loop inference with `nvidia/GR00T-N1.7-3B`;
6. run official DROID checkpoint inference with `nvidia/GR00T-N1.7-DROID`;
7. save environment/package/commit manifest.

E0 gates GR00T fine-tuning and runtime claims only. Local target-task work and ACT proceed independently.

### E1 — inspect real peg-insertion source
Load FIRe real Franka peg-insertion data.
Produce:
- schema report;
- camera montage;
- state/action dimension report;
- action-distribution plot;
- episode-length distribution;
- 2–3 representative success clips;
- exact compatibility assessment with N1.7 embodiment/action representation.
If source-specific fine-tuning is needed, create it as a separate checkpoint lineage:
`GR00T base → Franka real peg source`.

### E2 — build UR10e target environment
Create a generic UR10e insertion environment in Isaac Sim.
Requirements:
- deterministic seed;
- headless smoke test;
- visual smoke test;
- collision-aware expert;
- explicit geometric success predicate;
- failure labels;
- stable reset;
- reproducible camera capture.

Reuse concepts from `C:\aerospace-olp-ros2-isaacsim` only where they reduce risk.
Do not import from that local path at runtime.

### E3 — generate target demonstrations
Use a deterministic/collision-aware expert to create successful UR10e demonstrations.
Required sizes: 10, 25 and 50 successful episodes, expanded incrementally only after one-episode overfit and pipeline sanity pass.

Keep one larger source pool if generation is cheap, but train on exact deterministic subsets.

Every episode must record:
- seed;
- task text;
- randomization parameters;
- source/expert identifier;
- state/action/video;
- success/failure;
- failure reason;
- episode duration;
- dataset/code version identifiers.

Failed episodes belong in evidence/failure analysis but should not silently enter an imitation-learning training split unless the experiment explicitly requires them.
### E4 — executable target-only versus source-informed study

Track A: UR10e target demonstrations → maintained LeRobot ACT → closed-loop UR10e.
Track B: actual FIRe poses → canonical relative EEF representation → a documented
source-informed learning, initialization or conditioning mechanism → the same target
policy/evaluation contract, plus the same UR10e target demonstrations.

Implement both tracks before GR00T fine-tuning. The source must demonstrably influence
learning; visualization or concatenation alone does not establish transfer. Record
source lineage, transform, objective/parameters affected and checkpoint provenance.
Selected mechanism: FIRe pose-motion feature pretraining of the same maintained ACT.
Use pinned inspected episodes0/14/29 and actual wrist RGB. The terminal-EEF transform
is uniformly length-normalized to target task/action bounds; orientation-log magnitude
is normalized to the target angular span, then left-composed deltas are recomputed.
These are observed kinematic transitions, not verified source commands or calibrated
UR10e demonstrations. Source task status remains ANALOGOUS_TASK.

Freeze source pretraining at1000 steps on381 transitions. Use source action statistics
and unchanged target state statistics. Restore the exact original target action head
before target training because output normalization differs; retain learned backbone,
state and transformer features. Restore target RNG and start a fresh target optimizer.
Record feature hashes before/after, head restoration, source budget and final checkpoint.
The frozen target recipe is ImageNet ACT,H1/K1,15Hz,5000 steps,batch8,mainLR1e-4,
backboneLR1e-5,precision-balanced sampling,seed20260929. Apply it to both tracks at
all sizes. Source extra compute/data is reported separately; no compute-efficiency claim.

Run both tracks at 10, 25 and 50 target demonstrations. Freeze architecture, task,
controller, detector, observation/action contract and evaluation seeds for paired
comparisons. Record training budget and source-data budget separately. Use deterministic
nested target subsets and change training factors only in explicit separate diagnostics.
Primary plot: actual closed-loop success versus target demonstration count, with both
curves and actual denominators. Never fabricate monotonic improvement.

### E5 — GR00T N1.7 NEW_EMBODIMENT integration

Implement a real GR00TPolicy using the pinned NVIDIA Isaac-GR00T policy, canonical
LeRobot v2 data, modality.json, six-dimensional relative EEF actions and UR10e adapter.
Validate observation batching, modality/embodiment config, checkpoint identity,
statistics and action decoding. Do not transform already-relative actions twice.
INTEGRATION_READY / COMPUTE_DEFERRED is acceptable until supported compute and a real
adapted checkpoint exist. Do not return fake outputs or claim GR00T TRAINED.

Official base → UR10e fine-tuning and source-specialized GR00T adaptation are optional
compute-dependent extensions after the two local experimental tracks. Use the default
projector + flow-matching DiT path; additional backbone tuning needs measured reason.

## 10. Evaluation protocol

Open-loop action MSE is a diagnostic only.
Final claims require **closed-loop** simulation evaluation.

Before comparing final checkpoints:
- freeze a fixed held-out seed list;
- store it in Git;
- do not tune on it.
The existing fixed protocol is seeds 1000-1019 (20 episodes); training and development
seeds are separate. Report this as a P0 smoke distribution, not a broad benchmark.
All six track/data-size conditions use identical initial conditions. A larger benchmark
requires a prospectively defined separate protocol, never replacement of failed seeds.

Use receding-horizon control: observe → predict chunk → execute K → re-observe/replan.
Record actual observation/prediction/execution horizons, interpolation, control frequency
and policy query frequency. Use the smallest justified K; do not blindly sweep it.

Primary metric:
- task success rate.

Secondary metrics:
- collision rate;
- timeout rate;
- final lateral position error;
- final axial/insertion error;
- final orientation error;
- completion time;
- trajectory path length;
- action smoothness/jerk metric;
- inference latency;
- failure-reason distribution.

Report confidence intervals for success rate where practical.

Optional OOD split:
- target pose and/or orientation range just outside training distribution.
- define it before running the final comparison.

No cherry-picking best-looking episodes for numerical claims.

## 11. Hypothesis and completion boundary

Hypothesis: real-source relative EEF information reduces target demonstrations needed
by the same target policy compared with target-only learning. The result is measured,
not required to be positive. Report source-informed mechanism and task-match limitations.
GR00T cross-embodiment adaptation remains a separately labeled foundation-model extension.

Required outcomes are evidence of actual source use, validated transferable representation,
new embodiment control, real GPU training, reproducible learned held-out success, all
10/25/50 paired experiments, generated data-efficiency curves and verified technical media.
Simulator appearance, checkpoint existence or a single overfit success alone cannot finish
this project. Every public claim must trace to structured results.

## 12. Result artifact contract

Do not hand-edit numbers in README or video.

Expected generated artifacts:
- `artifacts/environment/*.json` — GPU, CUDA, Python, package and upstream commit manifest;
- `artifacts/datasets/*.json` — dataset schema/provenance summaries;
- `artifacts/training/<run_id>/config.json`;
- `artifacts/training/<run_id>/summary.json`;
- `artifacts/eval/<run_id>/episodes.jsonl`;
- `artifacts/eval/<run_id>/summary.json`;
- `artifacts/eval/comparison.csv`;
- `artifacts/figures/*.png`;
- `artifacts/video_manifest.json`.

Large media, checkpoints and raw datasets must be ignored or stored externally.

Each result summary must include:
- git commit;
- upstream GR00T commit;
- model checkpoint ID;
- dataset ID/version;
- seed set;
- environment config hash;
- evaluation episode count.

## 13. Suggested implementation layout

Create only as needed:

```text
AGENTS.md
README.md
pyproject.toml
configs/
  task/
  groot/
  eval/
src/
  sim/
  data/
  policy/
  eval/
scripts/
  local/
  colab/
tests/
artifacts/
media/
references/
```

Do not create empty folder forests. A directory should appear only with a real artifact or source file.
## 14. Cloud GPU workflow

Maintain the existing Colab notebook/Markdown adapter under `scripts/colab/`, but keep the E0 runner provider-agnostic. Colab is only one possible front end; any Linux NVIDIA GPU provider that satisfies the current stage's measured VRAM gate is acceptable. The official `googlecolab/google-colab-cli` is preferred for terminal automation when Colab is used.

First cells/steps:
1. `nvidia-smi`.
2. verify available disk/RAM.
3. install Git LFS and `uv` if missing.
4. clone a pinned official Isaac-GR00T revision.
5. `uv sync`.
6. authenticate with `hf auth login` or an environment secret.
7. verify `nvidia/Cosmos-Reason2-2B` access.
8. run E0 before GR00T runtime validation/fine-tuning; local dataset/ACT work remains independent.

Training rules:
- pin upstream commit IDs in the run manifest;
- use W&B only if configured, never require it for reproducibility;
- save checkpoints outside Git;
- save compact configs/results back to the project;
- export the best/final checkpoint identity, not the full weight blob, into Git evidence;
- do not silently resume an old run with changed data.

If the selected GPU is 40–48 GB-class:
- use official default N1.7 fine-tuning first;
- do not enable `--tune-llm` or `--tune-visual` in P0.

If the selected GPU has materially more VRAM (for example A100/H100 80 GB):
- still keep default fine-tuning for experimental comparability unless a later ablation explicitly tests more tuning.

## 15. Local Isaac Sim workflow

P0 local responsibility:
- build UR10e insertion task;
- generate demonstrations;
- export GR00T-compatible dataset;
- run policy client / closed-loop simulator;
- record evidence video.

Run ACT training on the local 12 GB GPU in an isolated ML environment. Run Isaac Sim and training sequentially when needed for VRAM. Defer GR00T training to supported compute.
A remote-policy architecture is acceptable:
```text
40 GB+ Linux NVIDIA GPU server
GR00T policy server
        |
        | predicted action chunk
        v
Local Windows workstation
Isaac Sim UR10e client
        |
        v
closed-loop rollout + recording
```

However, Colab inbound networking can be awkward. Do not make a fragile live network bridge a P0 requirement.
A safer P0 workflow is:
- train checkpoint on the selected authorized Linux GPU environment;
- download checkpoint;
- run inference where supported;
- or perform evaluation in a compatible Linux GPU environment.

Choose the simplest reproducible route supported by the actual environment; E0 is independent of the local ACT route.

## 16. Required software

### Local Windows
Required:
- NVIDIA Isaac Sim 6.0.1 already installed;
- NVIDIA driver compatible with that install;
- Python 3.12;
- Git;
- Git LFS;
- FFmpeg;
- `uv`.

Optional:
- Rerun for trajectory/data inspection;
- LeRobot tools for dataset visualization/conversion if dependency isolation is clean.

Keep Isaac Sim Python and general ML Python environments separate unless official compatibility proves otherwise.

### Cloud / Linux
Required for GR00T N1.7:
- Linux x86_64;
- E0 inference: NVIDIA CUDA GPU with at least 16 GB-class VRAM;
- E5 fine-tuning: NVIDIA CUDA GPU with at least 40 GB VRAM;
- Python 3.12 per current official N1.7 guidance;
- CUDA 12.6+;
- PyTorch 2.7+;
- Git + Git LFS;
- `uv`;
- official NVIDIA/Isaac-GR00T checkout;
- Hugging Face authentication and approved Cosmos-Reason2-2B access.
Optional:
- Weights & Biases for experiment tracking;
- LeRobot;
- Rerun;
- Jupyter/Colab notebook wrapper.

## 17. Evidence-linked media milestone

Final assembly requires underlying evidence and learned held-out success. Prepare actual
FIRe source, measured trajectory, real-to-sim transition and expert capture beforehand.
Deliver one 55-70 s 1920x1080 16:9 video at 30 fps, understandable muted, followed by a
10-15 s hero clip and poster. Optional social formats wait for canonical-cut acceptance.
Hook: GR00T N1.7 × UR10e / Real-World Cross-Embodiment Robot Learning.
Show actual Franka motion from the first frame, with compact third-party source credit.

Canonical sequence: real Franka video → actual source trajectory → relative EEF abstraction
→ GR00T cross-embodiment context → NEW_EMBODIMENT UR10e / Isaac Sim → 10/25/50 paired
tracks → actual ACT training → multi-UR10e evaluation wall → real Franka versus learned
UR10e split-screen → measured success curves → GR00T N1.7 integration.

The signature transition preserves a measured trajectory/coordinate frame while source
morphology fades and the target scene appears. It must trace to the actual software
transform, not an arbitrary spline. Show Δx Δy Δz ΔRx ΔRy ΔRz; never imply joint replay.
GR00T ecosystem examples (DROID, Franka, WidowX, Google Robot, G1, R1, human video) require
official verification and explicit ecosystem/pretrained-embodiment labels, not project
training claims. ACT TRAINED + EVALUATED and GR00T INTEGRATION READY remain distinct.

Build the multi-UR10e wall from actual evaluation episodes, each identified by seed and
recorded APPROACH/ALIGN/INSERT/SUCCESS/FAIL state. Do not fabricate extra embodiments.
Final payoff: left REAL WORLD / original Franka demonstration; right SIMULATION / learned
UR10e policy, aligned in approach/alignment/insertion where possible. Render ACT identity,
actual target demo count and held-out X/20 from results, then the GR00T integration hook.

## 18. Reproducible assets and verification

Use existing artifacts/, references/ and scripts/ as SSOT; create only needed media
helpers. One edit manifest maps shot, timing, source artifact/type, caption, transition,
claim, evidence reference and truth status. Generated charts/cards consume JSON/CSV,
not manually copied numbers. Keep heavy media/checkpoints/data ignored, with hashes and
regeneration commands. Internal classes: PUBLIC_REAL_DATA, PROJECT_SIM_EXPERT,
PROJECT_SIM_LEARNED, PROJECT_TRAINING, PROJECT_EVALUATION, ARCHITECTURE_GRAPHIC.

Only necessary cameras: HERO, MATCHED_SOURCE, CLOSE_INSERTION, EVALUATION. Store transform,
resolution, seed, task version, policy/expert identity and frame range. Capture must not
advance physics or change policy observations. Use measured loss, real GPU/run receipt,
checkpoint path/hash and actual evaluation. No fake telemetry, terminal output, HUD,
stock robot footage or synthetic success. Any music needs recorded usable rights.

Inspect actual execution environments through Remote Desktop at major stages: source
media/data, derived trajectory/canonical data, launched Isaac Sim scene, GPU training/logs,
checkpoint, actual learned rollout, representative success/failure and final rendered video.
Separate code, runtime, visual, evaluation and media verification. Save only representative
claim-linked receipts: REAL_SOURCE, SOURCE_TRAJECTORY, TARGET_ENV, TRAINING, CHECKPOINT,
LEARNED_ROLLOUT, EVALUATION, FINAL_VIDEO. Generated files alone do not establish visual PASS.

Media PASS requires actual source footage/provenance, architecture-backed transition,
expert and learned control evidence, real training, canonical evaluation numbers and
curves, clear ACT/GR00T and real/sim status, muted comprehension and claim-linked manifest.

## 19. LinkedIn post claim boundaries

Allowed after evidence exists:
- “Fine-tuned GR00T N1.7 for a new simulated UR10e embodiment.”
- “Compared a cross-embodiment pretrained VLA against an ACT target-only baseline.”
- “Evaluated target-data efficiency on fixed held-out simulation episodes.”
- “Used public real-robot data as source/reference training data.”

Do not write:
- “Transferred a Franka policy directly to UR10e” unless that exact experiment exists.
- “Sim-to-real” without physical UR10e tests.
- “Industrial deployment ready.”
- “Foundation model learned aerospace drilling” unless that exact dataset/task is run.
- “General intelligence” or “universal robot intelligence” as a technical claim.

Preferred terms:
- cross-embodiment robot learning;
- generalist robot policy;
- open-weight VLA;
- embodiment adaptation;
- target-data efficiency;
- standardized robot dataset;
- closed-loop evaluation.

## 20. Git / publication gates

No public push or LinkedIn publication before all applicable gates pass.

### G0 — Repository hygiene
- clean working tree;
- no credentials;
- no raw model weights;
- no large raw datasets;
- licenses/references documented;
- `.gitignore` covers checkpoints, HF cache, W&B, datasets, videos not intended for Git, Colab secrets.

### G1 — Simulator
- deterministic headless smoke test passes;
- visual smoke test passes;
- reset is repeatable;
- success/failure predicate is unit-tested;
- fixed seed can reproduce the same initial condition.

### G2 — Dataset
- GR00T v2 directory/schema validator passes;
- `meta/modality.json` is explicit and correct;
- video frame count and parquet timestep count are synchronized;
- no NaN/Inf in state/action;
- action ranges are plausible;
- train/eval episode IDs do not overlap;
- source provenance is recorded.

### G3 — Training
- for GR00T training only: official DROID sanity inference completed;
- for ACT: real local GPU training, loss, checkpoint and reload evidenced;
- exact upstream commit recorded;
- exact model checkpoint recorded;
- training config saved;
- dataset version/hash saved;
- no unexplained resume state.

### G4 — Evaluation
- fixed held-out seeds committed before final comparison;
- evaluated policies use the same episode set; unavailable GR00T is explicitly not evaluated;
- result JSONL saved;
- aggregate CSV/JSON generated by script;
- plots generated from aggregate artifacts;
- at least one failure example retained.

### G5 — Media
- every numerical overlay traceable to an artifact;
- public source footage attributed;
- altered playback speed labeled;
- simulation clearly labeled;
- no unsupported hardware/real-world claims.

### G6 — Release
- README can reproduce the main experiment from a clean checkout plus external datasets/models;
- fresh-machine or clean-environment smoke test documented;
- diff reviewed for dead files and duplicated config;
- Git LFS used only where intentionally necessary;
- tag a first evidence-backed release only after G0–G5.

Suggested tag after final validation:
`v0.1.0-cross-embodiment-evidence`
## 21. Commit strategy

Prefer small coherent commits, for example:
1. `chore: scaffold cross-embodiment research workspace`
2. `feat: add deterministic ur10e insertion environment`
3. `feat: export groot-compatible ur10e demonstrations`
4. `feat: add colab groot n1.7 finetune workflow`
5. `feat: add act target-only baseline`
6. `feat: add fixed-seed closed-loop evaluation`
7. `docs: publish evidence-backed results and media package`

Do not commit generated checkpoints.
Do not commit source datasets.
Do not squash evidence-generating implementation and post-hoc result edits into one opaque commit.

## 22. Codex operating procedure

At the start of every work session:
1. read this `AGENTS.md`;
2. inspect `git status`, recent log and actual source tree;
3. read the latest canonical design/status files if they exist;
4. inspect code before proposing new abstractions;
5. identify the current gate and work only on the smallest coherent scope needed to pass it.

When implementing:
- preserve an evidence-first boundary between expert control, learned policy and evaluation;
- keep dataset transforms deterministic and testable;
- isolate simulator-specific code from model-training code;
- validate real outputs rather than relying on import success;
- after changes, run targeted tests first, then broader gates;
- reread the diff and delete accidental bloat before committing.

If a planned technology becomes incompatible with the actual runtime, prefer the smallest evidence-preserving substitution and document the deviation rather than stacking workaround layers.

## 23. First execution order

Follow the current execution priority at the top. Read HANDOFF.md, inspect actual state,
and continue the earliest incomplete required step. Preserve G0 throughout. Do not redo
completed baseline runs or treat compute-deferred GR00T as a local experiment blocker.

## 24. Decision log / non-goals for P0

Explicitly deferred:
- real UR10e hardware;
- aerospace-specific cell;
- humanoid embodiment;
- ROS 2 network orchestration;
- Isaac Lab Mimic/SkillGen;
- KUKA multi-source training;
- SmolVLA benchmark;
- π0.5 benchmark;
- TensorRT deployment;
- LeRobot v3 native training;
- multi-GPU scaling;
- force/torque contact sensor modeling beyond what the task requires.
These may become P1/P2 only after the core transfer study is complete.

## 25. Definition of done

A reviewer must trace the complete causal chain: real robot experience → verified relative
EEF representation → new UR10e embodiment → actual source-informed and target-only training
→ matched fixed-seed closed-loop evaluation → measured 10/25/50 data efficiency.

The repository must identify exact source/license/episode/camera/files/hashes, task-match
classification, transform, algorithm/upstream versions, configurations, budgets, checkpoint
hashes, horizons/frequencies, seeds and all outcomes. It must reproduce each public number
and media claim. Include real-source/learned-target visual evidence and the inspected final
video. GR00T N1.7 NEW_EMBODIMENT integration must be real; runtime/fine-tuning status may
remain explicitly compute-deferred. Physical UR10e deployment is outside scope.

G0-G6 are evidence/release gates, not a substitute for this definition. G4 includes both
tracks at all three sizes; G5 includes the media acceptance above. Tagging/publication still
requires owner authorization. No completion claim may depend on undocumented session memory.

## 26. One-sentence project description

> Real Franka demonstrations inform a relative end-effector representation for a new
> simulated UR10e embodiment, tested through controlled target-data-efficiency experiments
> and a GR00T N1.7 NEW_EMBODIMENT integration path.

## 27. Video hook

> **REAL FRANKA → TASK-SPACE MOTION → UR10e**

Show real data, real training and real closed-loop evidence with GR00T N1.7 prominent.
