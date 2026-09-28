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
