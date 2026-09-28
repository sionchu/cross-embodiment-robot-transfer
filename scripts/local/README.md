# Local UR10e insertion

Run from the repository root using the installed Isaac Sim 6.0.1 Python environment:

```powershell
& C:\isaacsim\python.bat scripts\local\insertion.py --seeds 0 0 1 --output artifacts\runtime\expert-smoke
& C:\isaacsim\python.bat scripts\local\insertion.py --seeds 0 --collision-check --output artifacts\runtime\collision-smoke
```

Each output directory must be new. The runner saves JSON outcomes, per-step state/action
traces and actual start/end RGB frames. The collision test deliberately misses the hole
by 15 mm: its test PASS requires a COLLISION episode, not insertion success. Repeated
seeds must reproduce identical recorded episode results. Raw output remains ignored;
review compact evidence before committing it.

The official UR10e USD is loaded from NVIDIA's versioned Isaac 6.0 asset endpoint.
The fixed square peg is part of the wrist rigid body. Four collidable walls define
an 8 mm peg / 10 mm hole, with a 50 mm bore. Joint drives use the simulated robot's
measured wrist pose and Jacobian for damped differential IK; only reset teleports
joint state. Normal actions update drive targets and advance four 60 Hz physics
steps per 15 Hz control step. Physics contact reports monitor moving robot links,
including the peg, against fixture, pedestal and floor; self-collision is enabled
and net contact forces are also checked. The fixed base's expected ground contact
is excluded. This is an idealized insertion fixture, not a validated force-control model.

Observation state is `[px, py, pz, rx, ry, rz]`: peg-tip displacement from the hole
entrance in the hole frame (metres), followed by the rotation vector of
`R_hole^-1 R_wrist` (radians). Hole +Z points into the bore. RGB is one fixed front
camera. Geometry/config thresholds are versioned in `insertion.py:CONFIG`.

Actions are `[dx, dy, dz, dRx, dRy, dRz]` in the same hole frame. Translation is
added in that frame; rotation is left-composed as
`R_world_next = R_hole Exp(dR) R_hole^-1 R_world_current`.
Limits are per component: 3 mm and 0.025 rad per control step. Commands are actual
expert commands, not measured joint differences. Dataset/model adapters must preserve
these conventions and avoid applying a second relative-action conversion.

Geometric success requires 25–35 mm depth, <=0.7 mm lateral error, <=0.01 rad
orientation error and no recorded contact exceeding 1 N. Timeout is 180 control
steps. These thresholds are simulation task definitions, not hardware tolerances.

Sources: [NVIDIA articulation API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html),
[NVIDIA manipulator configuration](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_configure_manipulator.html).
Local implementation was checked against installed 6.0.1 API source. The separate
Apache-2.0 aerospace reference supplied examples of asset loading, SimulationApp
startup and Replicator capture; no runtime import or modification of that project occurs.

## Canonical dataset and isolated ML environment

Run `./scripts/local/setup.ps1` with Python 3.12, uv, Git, FFmpeg/ffprobe and 7-Zip
installed. It installs the pinned Windows ML dependencies in `.venv`, checks CUDA,
checks out the pinned GR00T data reader and downloads hash-verified shared FFmpeg
7.1.1 into `.external`. TorchCodec uses these DLLs through a process-local directory
handle. Isaac Sim and the Linux E0 dependency pins remain separate.

```powershell
& C:\isaacsim\python.bat scripts/local/insertion.py --seeds 10 11 12 13 14 15 16 17 18 19 --record --output artifacts/runtime/demos-new
.venv/Scripts/python.exe scripts/local/dataset.py --raw artifacts/runtime/demos-new --dataset datasets/ur10e_10 --upstream .external/Isaac-GR00T --report artifacts/datasets/validation.json
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Both output directories must be new. To validate an existing dataset, omit `--raw`.
The recorded run produced 10 successful episodes and 487 synchronized frames. The
training, development and held-out seed lists are frozen in `configs/seeds.json`.
Corruption tests require this real dataset; a fresh checkout without it reports
those tests as skipped, not passed.

Recording captures RGB before each action and checks that rendering neither advances
physics nor changes the corresponding state. Actual simulation timestamps and video
PTS are checked against the 15 Hz control cadence. The single v2.1 directory contains
Parquet, MP4, modality/task metadata, statistics and generation provenance; ACT reads
this same dataset through a runtime adapter.

`configs/groot_ur10e.py` registers NEW_EMBODIMENT with a 16-step action horizon.
`ABSOLUTE` is the official preprocessing bypass for already-encoded task-frame deltas,
not a world-absolute pose command. The pinned official episode loader read all 487
frames and its action processor round-tripped every action. This verifies the data
contract; it does not establish GR00T model execution or target-policy competence.


## ACT training

```powershell
.venv/Scripts/python.exe scripts/local/act.py --output artifacts/training/act-10-001
```

Use a new run name; this command never resumes or overwrites a checkpoint. It checks
the canonical dataset hash and installed LeRobot Git revision, then trains the
maintained ACT implementation on CUDA for 1,000 steps with batch size 8 and seed
20260929. The ResNet18/ACT model starts from scratch, predicts 16 actions and uses
one new observation per control step. The architecture retains upstream defaults
except the explicit input dimensions, chunk length and learning rates (1e-4).

RGB is resized from 640x480 to 320x240 with PIL bilinear interpolation and normalized
with ImageNet channel constants; state/action use training-set mean/std with a 1e-6
standard-deviation floor. Future action chunks are padded only within their own
episode and masked in the loss. No development or held-out observations enter training.

The run writes model `config.json`, provenance `run.json`, every-step `loss.csv` and
`summary.json`. The real weights and normalization metadata are saved under ignored
`checkpoints/<run-name>/`. The summary identifies every checkpoint file by SHA256
and requires a strict reload to reproduce a finite action chunk. Training PASS does
not imply closed-loop task success. `ACTPolicy.predict` returns physical task-frame
deltas; the simulator applies its declared action limits and must record clipping.


## Checkpoint closed-loop evaluation

```powershell
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy act --split development --output artifacts/runtime/act-development-new
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy act --split held_out --output artifacts/eval/act-10-002
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy expert --split held_out --output artifacts/eval/expert-002
```

Each output/run name must be new. The worker runs `.venv` ACT inference over a local
stdio pipe; Isaac Sim stays in its own Python process. Every control step sends a
fresh front RGB frame and measured target-relative pose, predicts a 16-step chunk,
and executes its first action. Both policies use the same scene and declared bounds.
Raw predictions and clipped commands are retained separately. `clipped_steps` counts
any floating-point clipping, including tiny rounding excess at a bound.

Episode JSONL records all successes/failures; summary JSON reports outcome counts,
Wilson confidence intervals and measured error/time/path/latency statistics. Runner
`execution_status=PASS` means evaluation finished, not that insertion succeeded.
The first episode video, all final/start images and action traces remain in ignored
`artifacts/runtime/eval-<run-name>/`. Development seeds 0 and 1 are never counted as
held-out results. No tuning is performed on held-out outcomes.


A measured Kit 6.0.1 render initialization stall occurred at the seventeenth scene
reset. Evaluation therefore uses fresh simulator processes for blocks of 10 seeds,
then verifies and combines all 20 episodes. Both policies use the same isolation.
Interrupted runs remain recorded as failures and are excluded from the comparison;
no model parameters or held-out seeds change during this infrastructure recovery.

## GR00T N1.7 integration

```powershell
.venv/Scripts/python.exe scripts/local/groot.py --report artifacts/environment/gr00t_local_contract.json
```

This checks the pinned upstream, hashes of the actual canonical data, official
observation batching and official action normalization/physical decoding. It does
not load model weights. The report keeps runtime availability and target-checkpoint
validation separate: on this Windows 12 GB machine, runtime is
`MODEL_COMPUTE_NOT_AVAILABLE`; target checkpoint validation is `NOT_RUN` because no
UR10e-adapted GR00T checkpoint exists.

On a supported Linux GPU environment prepared by the pinned E0 workflow, the real
adapter is `GR00TPolicy(checkpoint, upstream, e0_evidence)` in `groot.py`. It checks
N1.7 model/backbone identifiers, NEW_EMBODIMENT modalities, statistics, embodiment ID,
action horizon, weight files and a matching E0 PASS before loading the official
`Gr00tPolicy`. The same `predict({'state': state6, 'front': rgb})` interface returns
a physical `(16,6)` action chunk. The official policy has already decoded physical
units; the adapter never applies a second relative transform. A base/DROID checkpoint
without the target embodiment is rejected. This runtime branch awaits real target
fine-tuning and supported compute; contract PASS is not model-runtime PASS.

`python scripts/local/compare.py` regenerates the CSV, README table and plot from
complete matched evaluation artifacts. Plotting uses the separately installed
Matplotlib 3.11.1; it does not alter the isolated training dependency lock. On a clean
machine, install `matplotlib==3.11.1` in the Python environment used for this plotting
command. Raw trajectory videos remain ignored; a compact failure image is retained.


## Reproduce without overwriting committed evidence

The commands above identify the recorded runs. Their output folders already exist
in a checkout containing evidence. For a new training/evaluation run after generating
the canonical dataset, use new names:

```powershell
$runStamp = Get-Date -Format yyyyMMdd-HHmmss
$actRun = "act-reproduction-$runStamp"
$expertRun = "expert-reproduction-$runStamp"
.venv/Scripts/python.exe scripts/local/act.py --output "artifacts/training/$actRun"
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy act --split held_out --checkpoint "checkpoints/$actRun" --output "artifacts/eval/$actRun"
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy expert --split held_out --output "artifacts/eval/$expertRun"
python scripts/local/compare.py --act-run $actRun --expert-run $expertRun
```

The last command regenerates the comparison/README table from those named runs.
The original run folders and checkpoint metadata remain intact. Inspect the generated
`summary.json` execution status; a shell exit code alone is insufficient when the
Isaac Sim Windows launcher is terminated externally.

## ACT success recovery diagnostics

The original 0/20 checkpoint remains `act-10-001`. Do not overwrite it.
`diagnose_act.py` checks actual model input tensors, action distributions and
open-loop predictions; `--features` records projected features and attention.
Open-loop error and training PASS do not establish task success.

```powershell
.venv/Scripts/python.exe scripts/local/diagnose_act.py --baseline --output artifacts/diagnostics/audit-new
python scripts/local/diagnose_act.py --plot --output artifacts/diagnostics/audit-new
.venv/Scripts/python.exe scripts/local/diagnose_act.py --features --output artifacts/diagnostics/audit-new
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy replay --split train --seed-count 1 --output artifacts/eval/replay-new
.venv/Scripts/python.exe scripts/local/act.py --episodes 0 --steps 5000 --output artifacts/training/overfit-new
& C:\isaacsim\python.bat scripts/local/evaluate.py --policy act --split train --seed-count 1 --checkpoint checkpoints/overfit-new --output artifacts/eval/overfit-new
```

Replay uses the original recorded trace cast to canonical float32; every recorded
state/action was verified bit-exact against the exported Parquet. It uses the same
clipping, controller, 15 Hz execution and success detector as ACT. The reference
trace is in the ignored `artifacts/runtime/demos-01` run generated above.

`--episodes 0` trains only the first episode without copying a second dataset.
Normalization remains the full canonical training statistics to isolate the factor.
The first 5000-step one-episode checkpoint also failed. Its visual feature magnitude
and nearly saturated attention motivate a controlled `--backbone-weights imagenet`
experiment: official ResNet18 visual initialization only; robot-action learning still
uses target demonstrations. All other hyperparameters and task thresholds stay fixed.
The exact image weights are pinned in `artifacts/diagnostics/backbone_initialization.json`.
