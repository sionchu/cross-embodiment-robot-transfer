# Verified GR00T N1.7 contract

Verified 2026-09-28 against commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
These are source checks; a successful A100 run is still required.

## Environment

The pinned [pyproject.toml](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/pyproject.toml)
requires Python >=3.12,<3.13 and pins PyTorch 2.9.0, torchvision 0.24.0,
transformers 4.57.3, flash-attn 2.8.3 and torchcodec 0.8.0. Linux torch wheels
come from cu128. The [installation guide](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/README.md)
requires FFmpeg 4-7 for this torchcodec version. Use the upstream lock unchanged.
This supersedes AGENTS.md's older minimum-version notes for this pinned E0 run.
Driver compatibility and available A100 VRAM must be measured in Colab.

## Data and policy

The [data guide](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/data_preparation.md)
still requires GR00T-flavored LeRobot v2 with `meta/modality.json`; v3 inputs
must be converted before reference training. No v3 exporter is needed for E0.

The pinned bundled DROID sample reports v2.1, three episodes, 844 frames, 15 FPS,
two 180x320 RGB cameras and 17-dimensional stored state/action arrays in
[info.json](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/demo_data/droid_sample/meta/info.json).
[modality.json](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/demo_data/droid_sample/meta/modality.json)
splits both arrays into `eef_9d[0:9]`, `gripper_position[9:10]`, and
`joint_position[10:17]`. Language maps to `task_index`.

The [registered policy config](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/configs/data/embodiment_configs.py)
uses `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, relative XYZ_ROT6D EEF,
absolute gripper and relative joints. Relative action handling is a model/data
transform contract; the stored array shape alone does not establish its units
or authorize replay on UR10e. Video indices are [-15,0], state index [0],
action indices 0..39. Raw camera keys `exterior_1_left` and `wrist_left` are
mapped to policy keys `exterior_image_1_left` and `wrist_image_left`.

Use the official [standalone inference entry point](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/scripts/deployment/standalone_inference_script.py),
PyTorch mode, trajectories 1 and 2, seed 42, 200 steps maximum per trajectory,
and execution horizon 8. Episode 0 can have empty language. Run both pinned
base and DROID checkpoints. Open-loop MSE/MAE verifies the inference path;
it is not a transfer or closed-loop task-success metric.

`NEW_EMBODIMENT` requires an explicit modality configuration. The current
[custom embodiment example](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/finetune_new_embodiment.md)
is SO100 joint-space, so it does not by itself settle a UR10e task-space adapter.
That decision remains deferred until E0 succeeds.

## Provenance and licenses

`source_manifest.json` is the pin source. Official code is Apache-2.0;
the pinned [base model card](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/README.md)
and [DROID model card](https://huggingface.co/nvidia/GR00T-N1.7-DROID/blob/05e7cc97e40dbd33b0890c35cc0214fcb0547ab5/README.md)
state NVIDIA Open Model License Agreement. Do not infer weight licensing from
the repository's Apache license or its broader README claims.
The [DROID dataset card](https://huggingface.co/datasets/lerobot/droid_1.0.1/blob/0eabc778f959c54b8c5aa3626cc1128d2d2e54d4/README.md)
reports Apache-2.0. Cosmos access requires the owner to accept the terms on
[its model page](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
This project links to external material and does not redistribute weights/data.
No project-wide license has been selected on the owner's behalf.
