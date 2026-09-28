# Cross-Embodiment Robot Transfer — Codex Execution Plan

Status: canonical project instruction
Date: 2026-09-28
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

### Current owner-approved execution priority (2026-09-28)

The owner explicitly approved this scope change after the Colab availability limitation.
Do not spend further time on Colab entitlement or waiting for T4. Preserve the existing
pinned E0 runner. E0 gates actual GR00T runtime claims and GR00T fine-tuning only;
it does not gate the simulator, demonstrations, canonical dataset, ACT or ACT evaluation.

Execute this local P0 in order:
1. Reconcile the canonical planning/status files in place.
2. Build the smallest deterministic official UR10e fixed-peg insertion task in Isaac Sim.
   Verify reset, repeated seeds, geometric success, collisions and actual rendered output.
3. Generate at least 10 successful expert demonstrations.
4. Export one GR00T LeRobot v2 dataset with explicit `meta/modality.json`. Validate
   modalities, dimensions, finite values, timestamps, episode continuity, action limits
   and video/state synchronization. ACT and GR00T consume this same canonical data.
5. Train maintained LeRobot ACT on the RTX 4070 Ti in an isolated ML environment.
   Record actual GPU use/loss and a real checkpoint; reload it.
6. Execute checkpoint inference in closed-loop UR10e simulation on fixed held-out seeds.
   Save every episode outcome and generated aggregate results, including failures.
7. Implement real ACTPolicy and GR00TPolicy paths mapping observation to action chunks.
   Validate N1.7 embodiment/modality/checkpoint/action decoding before model execution.
   GR00T may explicitly report MODEL_COMPUTE_NOT_AVAILABLE; never substitute fake outputs.
8. Publish a compact local result table for Expert, ACT and GR00T with missing GR00T
   metrics as unavailable. No external push/publication is authorized by this plan.

Minimum input: front RGB and target-relative EEF pose. Extra velocity, cameras, depth
or gripper channels require a measured task need. Canonical action is six-dimensional
relative task-space `[dx, dy, dz, dRx, dRy, dRz]`; document units, reference frame,
rotation convention and composition explicitly and verify the N1.7 transformation contract.
No Franka joint replay or separate ACT dataset.

GR00T N1.7 remains the primary foundation-model integration target. Until execution
is evidenced, use compatible/ready/integration language; ACT metrics belong to ACT.
The minimum milestone requires actual simulation, 10 valid demos, GPU training,
checkpoint reload, learned closed-loop held-out results and the real GR00T adapter
contract. Unit tests or a planning update alone cannot complete it.

After that milestone only: 25/50/100 demos, FIRe source inspection/mapping,
source-informed comparison, GR00T inference on available >=16 GB compute and
fine-tuning on >=40 GB. Quantization is not P0. Media polish waits until evidence exists.

### P0 priority override: recover learned-policy success (2026-09-29)

The preserved 0/20 ACT result is a debugging baseline, not the final conclusion.
Immediate order: exact-path expert -> recorded demonstration replay -> actual
observation tensors -> normalization/frame/convention audit -> action distributions
-> one-episode overfit -> same-condition closed-loop test -> fix proven cause ->
retrain 10-demo ACT -> existing fixed 20-seed evaluation. Expand to 25/50 only if
needed. Classify each failure A-J with evidence. No GR00T compute, quantization,
alternative architecture or final editing during this loop. Final assembly requires
at least one reproducible learned-policy held-out success; preserve all failures.
Do not change seeds, substitute expert actions, intervene or relax only the video.
Any justified task simplification must be deterministic, documented and shared by
expert and ACT with a new protocol identity. Debug on training/development cases;
held-out results confirm rather than drive tuning. Report actual denominators.

## 0. Mission

Build and validate a reproducible cross-embodiment robot-learning pipeline:

**public real-robot manipulation data → open-weight GR00T N1.7 prior → new UR10e embodiment in NVIDIA Isaac Sim → closed-loop evaluation.**

Primary research question:

> Does cross-embodiment pretraining reduce the amount of target-robot demonstration data required to learn a precision insertion task?

The project is successful if the experiment is reproducible and honestly measured, even if transfer does **not** outperform a from-scratch baseline.

## 1. Scope decision

This is a **new standalone project**. Do not turn the existing aerospace repository into a generic research sandbox.

Reference-only local repository:
- `C:\aerospace-olp-ros2-isaacsim`
- Reuse only small, generic and clearly attributable implementation ideas if needed: official UR10e asset loading, cuMotion/RMPflow control, Replicator RGB capture, deterministic smoke-test patterns.
- Do not modify that repository as part of this project unless explicitly requested later.
## 2. Clean-v0 rules

- Prefer one clear vertical slice over many half-finished features.
- P0 is GR00T N1.7 + UR10e peg insertion + ACT baseline + fixed evaluation.
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
Use as the most task-aligned real-world source reference:
- Dataset: https://huggingface.co/datasets/bhe1004/FIRe-VLA-GR00T-PegInsert-Real
- Task: contact-rich peg insertion.
- Domain: real.
- Robot: Franka Emika Panda.
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
Initial target: 10 successful episodes. Increase to 25/50/100 only after the first checkpoint trains and executes in closed loop.

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
### E4 — ACT target-data-only baseline
Train ACT from scratch or standard initialization using only UR10e target demonstrations.
P0 trains on the first 10 successful episodes. Larger data-size comparisons are stretch work after checkpoint execution.

Keep architecture/training choices constant across data-size comparisons.

### E5 — GR00T base → UR10e adaptation
Fine-tune `nvidia/GR00T-N1.7-3B` using the UR10e `NEW_EMBODIMENT` dataset.
Use the same target-data sizes where compute permits.

Default fine-tuning should remain the official projector + flow-matching DiT action-head path.
Do not enable full LLM or visual-backbone tuning without a measured reason.

### E6 — source-specialized transfer (stretch but high value)
If E0–E5 are stable:
1. fine-tune GR00T base on the real Franka peg-insertion source;
2. use that checkpoint as the initialization for UR10e target adaptation;
3. compare against direct base → UR10e adaptation.

This is the clearest experiment matching the narrative:
**real Franka insertion experience → new UR10e embodiment**.

Do not assume sequential source fine-tuning helps. Catastrophic forgetting or modality mismatch is a valid negative result.

## 10. Evaluation protocol

Open-loop action MSE is a diagnostic only.
Final claims require **closed-loop** simulation evaluation.

Before comparing final checkpoints:
- freeze a fixed held-out seed list;
- store it in Git;
- do not tune on it.
Recommended final held-out evaluation:
- 100 episodes per final condition if runtime permits;
- minimum 20 for smoke/iteration only;
- identical target poses/randomization seeds across compared policies.

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

## 11. Research hypotheses and result goals

H1:
A cross-embodiment GR00T prior will require fewer UR10e target demonstrations than ACT trained only on target data to reach a comparable success rate.

H2:
A GR00T checkpoint additionally exposed to real Franka peg-insertion data may adapt more efficiently to UR10e peg insertion than the generic GR00T base.

These are hypotheses, **not required outcomes**.

Minimum project success:
- target simulator and expert are reproducible;
- 10 successful demonstrations form one validated GR00T-compatible dataset;
- local ACT training produces and reloads a real checkpoint;
- held-out learned closed-loop evaluation completes;
- a real GR00T adapter contract is validated, with unavailable model execution explicit;
- all published charts are generated from committed result JSON/CSV files.

Strong-result target, not a gate:
- GR00T at 25–50 target episodes matches or exceeds ACT using materially more target data;
- or source-specialized transfer improves low-data success by a meaningful margin.

A negative result is publishable if methodology and evidence are sound.
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

## 17. Owner-approved real-source/media milestone (2026-09-29)

The owner extends local P0 with an evidence-linked video. Begin FIRe inspection now;
do not wait for GR00T cloud compute. This section supersedes the older storyboard.
The completed ACT checkpoint remains target-only and scored 0/20 held-out success.
Do not retrospectively claim that it learned from FIRe or succeeded at insertion.

Canonical source: `bhe1004/FIRe-VLA-GR00T-PegInsert-Real`. Inspect the card, actual
media, views, license and state/action contract. Select 2-3 candidate episodes and
one canonical episode/view. Pin revision, exact paths and hashes; cache only needed
payloads outside Git. Keep third-party attribution and required license material.

Order: source inspection/provenance -> measured source trajectory -> verified
Franka-to-task-space transform -> trajectory transition -> matched UR10e capture ->
expert capture -> actual training evidence -> learned rollout -> measured result
card -> architecture -> canonical cut -> claim/visual review -> hero and poster.
Missing physical units, frames or camera calibration must remain explicit; never
fabricate projected telemetry. A source-informed experiment needs its own lineage.

Deliver a 55-70 s 1920x1080 16:9 cut at 30 fps, understandable muted, then a 10-15 s
hero and poster. Optional social formats wait for acceptance of the canonical cut.
Hook: `GR00T N1.7 x UR10e / Real-World Cross-Embodiment Robot Learning`.
Show actual Franka motion in the first frame; no black intro or unrelated footage.

Canonical storyboard (seconds):
- 0-4 real Franka hook; 4-9 episode/view and evidence-derived source overlays.
- 9-17 morphology fades while actual task trajectory/frame persists; show six
  translation/rotation deltas, then reveal the UR10e scene. No joint replay claim.
- 17-24 analogous real/sim task; 24-31 source adapter/canonical dataset/modalities.
- 31-37 ACT TRAINED + EVALUATED versus GR00T INTEGRATION READY, shared policy API.
- 37-44 actual RTX 4070 Ti training/loss/checkpoint; 44-54 actual learned rollout.
- 54-59 measured P0 smoke metrics; 59-65 whole pipeline and GR00T headline.

A learned failure remains a failure; INSERT SUCCESS is allowed only for the actual
policy that succeeded. GR00T TRAINED requires a real fine-tuning checkpoint.
Do not connect source data to target-only training as if that transfer was executed.

## 18. Reproducible media implementation and acceptance

Use the smallest scripts that acquire selected source media, derive trajectories,
capture simulation and render charts/cards from actual JSON/CSV. One canonical
edit manifest maps shot/time/source/caption/transition/claim/evidence/truth status.
Heavy output remains ignored with hashes and regeneration instructions. Internal
classes: PUBLIC_REAL_DATA, PROJECT_SIM_EXPERT, PROJECT_SIM_LEARNED,
PROJECT_TRAINING, PROJECT_EVALUATION, ARCHITECTURE_GRAPHIC.

Define only needed HERO, MATCHED_SOURCE, CLOSE_INSERTION and EVALUATION cameras;
record transform, resolution, seed, task episode, policy and frame range. Rendering
must not change policy observations or advance physics. Never substitute expert
footage for learned control. Altered playback speed must be labeled.

Keep overlays restrained and legible: actual video, trajectory, coordinate frames,
real loss and generated results. No fake HUD, stock robots or invented sounds.
Audio is optional; any music must have a recorded usable license.

G5 PASS requires real FIRe footage and provenance; an actual architecture-backed
Real -> task-space -> UR10e transition; expert and learned rollout evidence; real
ACT training; generated evaluation values; explicit ACT/GR00T status distinction;
no unsupported transfer/fine-tuning claim; clear real/simulation identity; muted
comprehensibility; and an edit manifest linking every claim to evidence.
Final assembly waits only on genuinely missing evidence, not cloud entitlement.

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

Follow the owner-approved local P0 order at the top of this file. Preserve G0 hygiene
throughout. G1 simulator -> G2 canonical dataset -> G3 ACT training -> G4 evaluation
is independent of E0. FIRe source specialization, larger dataset comparisons and
GR00T fine-tuning are stretch experiments. No portfolio polish before the first real
checkpoint and closed-loop result.

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

The project is done for first public release when a reviewer can answer all of these from the repository:

1. What real public robot data was used, and under what license?
2. What open-weight model/checkpoint was used?
3. What exactly changed between Franka-source and UR10e-target embodiment representations?
4. How was the UR10e target data generated?
5. How many target demonstrations were used?
6. What baseline was compared?
7. Were both policies evaluated on identical held-out closed-loop episodes?
8. What were success rate and failure modes?
9. Can every graph and video number be traced to result artifacts?
10. What is simulation-only and what, if anything, was physically validated?

If any answer depends on memory, manual explanation, or an uncommitted notebook state, the release is not ready.

## 26. One-sentence project description

> A GR00T N1.7-compatible UR10e robot-learning pipeline with a canonical task-space dataset, local ACT training and closed-loop simulation evaluation; foundation-model transfer remains an experiment requiring actual compute and results.

## 27. Video hook

> **Can robot intelligence learned from a real Franka transfer to a UR10e it has never seen?**

That question—not aerospace, not a list of frameworks—is the project narrative.
