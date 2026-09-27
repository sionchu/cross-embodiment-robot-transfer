# Cross-Embodiment Robot Transfer — Codex Execution Plan

Status: canonical project instruction
Date: 2026-09-27
Owner intent: build an evidence-first public portfolio/research project, not a demo that merely name-drops VLA models.

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

### Cloud training
- Google Colab with NVIDIA A100 available.
- First cell must run `nvidia-smi` and record actual GPU model and VRAM.
- GR00T N1.7 official guidance: 40 GB+ VRAM for fine-tuning; default fine-tuning trains projector + flow-matching DiT action head and is documented as staying below roughly 35 GB peak VRAM.
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

Use DROID primarily to verify that the official N1.7 installation and embodiment handling are correct before custom work.
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
- Do not build dual-format support before the first GR00T fine-tune succeeds.

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
The exact rotation encoding and normalization must be copied from an official current N1.7 modality/action example, not invented.

Target observation candidates:
- `observation.state`: J1–J6 positions, plus only additional proprioception justified by the modality config;
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
On Colab A100:
1. record `nvidia-smi`;
2. clone official Isaac-GR00T with submodules;
3. authenticate to Hugging Face without storing token in notebook output;
4. confirm access to `nvidia/Cosmos-Reason2-2B`;
5. run official DROID sample zero-shot open-loop inference with `nvidia/GR00T-N1.7-3B`;
6. run official DROID checkpoint inference with `nvidia/GR00T-N1.7-DROID`;
7. save environment/package/commit manifest.

Do not proceed to custom fine-tuning until E0 is green.

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
Initial target sizes:
- 10 episodes;
- 25 episodes;
- 50 episodes;
- 100 episodes.

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
Train at minimum on:
- 10;
- 25;
- 50;
- 100 episodes.

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
- source-data inspection is reproducible;
- target simulator and expert are reproducible;
- GR00T target fine-tuning completes;
- ACT baseline completes;
- fixed closed-loop comparison completes;
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
## 14. Colab A100 workflow

Create a reproducible notebook or Python/Markdown workflow under `scripts/colab/`.

First cells/steps:
1. `nvidia-smi`.
2. verify available disk/RAM.
3. install Git LFS and `uv` if missing.
4. clone a pinned official Isaac-GR00T revision.
5. `uv sync`.
6. authenticate with `hf auth login` or an environment secret.
7. verify `nvidia/Cosmos-Reason2-2B` access.
8. run E0 before any custom dataset work.

Training rules:
- pin upstream commit IDs in the run manifest;
- use W&B only if configured, never require it for reproducibility;
- save checkpoints outside Git;
- save compact configs/results back to the project;
- export the best/final checkpoint identity, not the full weight blob, into Git evidence;
- do not silently resume an old run with changed data.

If Colab provides A100 40 GB:
- use official default N1.7 fine-tuning first;
- do not enable `--tune-llm` or `--tune-visual` in P0.

If A100 80 GB:
- still keep default fine-tuning for experimental comparability unless a later ablation explicitly tests more tuning.

## 15. Local Isaac Sim workflow

P0 local responsibility:
- build UR10e insertion task;
- generate demonstrations;
- export GR00T-compatible dataset;
- run policy client / closed-loop simulator;
- record evidence video.

Keep model training off the local 12 GB GPU if it creates avoidable memory workarounds.
A remote-policy architecture is acceptable:
```text
A100 / Colab or other GPU server
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
- train checkpoint on Colab;
- download checkpoint;
- run inference where supported;
- or perform evaluation in a Linux GPU environment with a compatible simulator/client.

Choose the simplest reproducible route supported by the actual environment after E0.

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

### Colab / Linux
Required for GR00T N1.7:
- Linux;
- NVIDIA A100;
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

## 17. Video / LinkedIn narrative

Working title:
**Can Robot Intelligence Transfer Across Robots? From Real Franka Data to an Isaac Sim UR10e**

Technical alternate:
**Cross-Embodiment Robot Learning: Adapting GR00T N1.7 to a New UR10e Embodiment**

Do not title it as aerospace unless an aerospace-specific experiment is actually performed.

Recommended final video:
- 16:9, 1920×1080.
- 30 fps export.
- 55–75 seconds.
- English on-screen technical captions.
- Korean explanation belongs primarily in the LinkedIn post body if desired.
- No cinematic intro longer than ~2 seconds.

### Storyboard

**0–4 s — Hook**
Visual: split-screen real Franka insertion source vs Isaac Sim UR10e.
Caption:
`Can robot intelligence transfer across robots?`

Sub-caption:
`Real Franka demonstrations → Open-weight VLA → UR10e`

**4–12 s — Real source evidence**
Visual: licensed/public FIRe/DROID clip or dataset viewer capture.
Caption:
`Public real-robot data | Franka Panda`
`Precision insertion demonstrations`
Small source attribution on screen:
`Source: Hugging Face public dataset — see project references`

**12–20 s — Representation bridge**
Visual: freeze frame; overlay source EEF frame, target EEF frame, Δpose arrow.
Captions:
`Different joints. Shared task-space motion.`
`Relative end-effector action`

Show a simple transformation graphic:
`Franka → canonical EEF action → UR10e`

**20–31 s — Open-weight model**
Visual: clean architecture card, not a wall of logos.
Captions:
`GR00T N1.7 — open-weight cross-embodiment VLA`
`Cosmos-Reason2-2B + flow-matching DiT action head`
`Target robot registered as NEW_EMBODIMENT`

**31–44 s — Target simulator**
Visual: UR10e target randomization montage and one complete insertion.
Captions:
`Isaac Sim | UR10e precision insertion`
`Randomized target pose + fixed held-out seeds`
State callouts:
`APPROACH → ALIGN → INSERT → VERIFY`

**44–57 s — Experiment**
Visual: data-size cards and training/eval plots.
Captions:
`10 / 25 / 50 / 100 target demonstrations`
`GR00T adaptation vs ACT from scratch`
`Same held-out episodes. Closed-loop evaluation.`

**57–68 s — Results**
Visual: final success-vs-data curve plus one matched pair of rollout clips.
Caption template:
`Target-data efficiency`
`ACT: <measured>`
`GR00T: <measured>`
Never place placeholder numbers into the final render.

**68–75 s — Engineering boundary/end card**
Caption:
`Transfer is measured, not assumed.`
`Public real data → reproducible embodiment adaptation`
Small footer:
`Simulation target only. No real UR10e validation claimed.`

## 18. Premiere Pro editing guidance

Use Premiere only to improve comprehension, never to manufacture evidence.

Recommended effects:
- split-screen Crop for source-vs-target comparison;
- Match Cut at the moment both robots enter insertion alignment;
- keyframed Transform/Scale for 5–10% push-ins on important pose details;
- freeze frame for the embodiment/action explanation;
- simple tracked line/arrow graphics for EEF Δpose;
- Linear Wipe or mask reveal for metric curves;
- subtle Time Remapping only for non-measurement B-roll;
- clean lower thirds for model/dataset IDs;
- hard cuts or 4–8 frame dissolves; avoid flashy transitions;
- very light UI click/servo sound design only if it does not obscure narration.

Do not:
- use fake HUD numbers;
- speed up a measured rollout without an explicit `2×` / `4×` badge;
- crop away failures when presenting quantitative results;
- add synthetic robot sounds that imply hardware execution;
- show another dataset’s robot footage while labeling it as UR10e.

Suggested visual language:
- engineering-white/neutral background;
- one consistent accent color;
- monospace labels for model/dataset IDs;
- large result numbers only when sourced from final JSON.
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
- official DROID sanity inference completed;
- exact upstream commit recorded;
- exact model checkpoint recorded;
- training config saved;
- dataset version/hash saved;
- no unexplained resume state.

### G4 — Evaluation
- fixed held-out seeds committed before final comparison;
- ACT and GR00T use the same episode set;
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

P0 sequence:
1. scaffold repo and hygiene files;
2. create reference/source manifest;
3. build Colab E0 GR00T sanity notebook/script;
4. verify source FIRe real peg dataset schema and compatibility;
5. implement deterministic UR10e insertion environment;
6. implement expert rollout + target dataset recording;
7. validate GR00T v2 export;
8. train ACT target-only baseline;
9. fine-tune GR00T base on target embodiment;
10. run fixed closed-loop evaluation;
11. only then attempt source-specialized real-Franka → UR10e transfer;
12. generate results figures;
13. record matched source/target video;
14. produce LinkedIn edit package;
15. pass release gates.

Do not start with Premiere, README cosmetics, or a custom UI.

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

> An evidence-first cross-embodiment robot-learning study that adapts an open-weight GR00T N1.7 prior and public real-world manipulation experience to a new simulated UR10e precision-insertion embodiment, then measures target-data efficiency against an ACT target-only baseline.

## 27. Video hook

> **Can robot intelligence learned from a real Franka transfer to a UR10e it has never seen?**

That question—not aerospace, not a list of frameworks—is the project narrative.
