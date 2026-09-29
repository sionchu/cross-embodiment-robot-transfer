"""Pinned FIRe candidate acquisition and pose-derived media evidence (no policy claims)."""
import argparse
import hashlib
import io
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DATASET = "bhe1004/FIRe-VLA-GR00T-PegInsert-Real"
REVISION = "ad5e895d53254caa61b7535e10a8a7790536451e"
EPISODES = (0, 14, 29)
VIEWS = ("left_view", "right_view", "wrist_view")
CACHE = ROOT / "artifacts/runtime/fire-inspection"
REPORT = ROOT / "artifacts/datasets/fire_source.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def candidate_paths():
    paths = ["README.md", "meta/info.json", "meta/modality.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]
    for episode in EPISODES:
        paths.append(f"data/chunk-000/episode_{episode:06d}.parquet")
        paths.extend(f"videos/chunk-000/observation.images.{view}/episode_{episode:06d}.mp4" for view in VIEWS)
    return paths


def acquire(download=False):
    """Resolve inventory from pinned Hub API; never accept an invented media path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    inventory = CACHE / "pinned_hub.json"
    if download:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}?blobs=true", timeout=60) as r:
            inventory.write_bytes(r.read())
    if not inventory.exists():
        raise RuntimeError("Run --download to resolve the pinned public inventory first")
    hub = json.loads(inventory.read_text(encoding="utf-8"))
    assert hub["sha"] == REVISION
    available = {x["rfilename"]: x for x in hub["siblings"]}
    prior = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    expected = {x["source_file"]: x["sha256"] for x in prior.get("files", [])}
    files = []
    for name in candidate_paths():
        if name not in available:
            raise RuntimeError(f"Path absent from pinned Hub inventory: {name}")
        local = CACHE / name
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
        if not local.exists():
            if not download:
                raise FileNotFoundError(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as r:
                local.write_bytes(r.read())
        digest = sha(local)
        remote = available[name]
        if "lfs" in remote:
            if digest != remote["lfs"]["sha256"]:
                raise ValueError(f"Pinned LFS hash mismatch: {name}")
        else:
            payload = local.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if git_hash != remote["blobId"]:
                raise ValueError(f"Pinned Git blob mismatch: {name}")
        if name in expected and digest != expected[name]:
            raise ValueError(f"Cached source hash mismatch: {name}")
        files.append({"source_file": name, "url": url, "sha256": digest, "bytes": local.stat().st_size, "pinned_upstream_hash_verified": True})
    return hub, files


def relative_trajectory(state):
    """Measured motion in terminal-EEF coordinates, NOT calibrated hole coordinates.

    Position remains in recorded units. Quaternion wxyz is explicitly converted to
    scipy xyzw. Spatial rotation deltas compose on the LEFT in this fixed frame.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 9 or not np.isfinite(state).all():
        raise ValueError("Expected finite Nx9 state")
    norms = np.linalg.norm(state[:, 3:7], axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        raise ValueError("Source quaternion is not unit length")
    rot = Rotation.from_quat(state[:, [4, 5, 6, 3]])
    anchor = rot[-1]
    position = anchor.inv().apply(state[:, :3] - state[-1, :3])
    local_rot = anchor.inv() * rot
    dpos = np.diff(position, axis=0)
    drot = (local_rot[1:] * local_rot[:-1].inv()).as_rotvec()
    # Reconstruct the measured sequence to detect body/spatial composition errors.
    reconstructed = [local_rot[0]]
    for delta in drot:
        reconstructed.append(Rotation.from_rotvec(delta) * reconstructed[-1])
    r_error = max((a.inv() * b).magnitude() for a, b in zip(reconstructed, local_rot))
    p_error = np.max(np.abs(position[0] + np.vstack([np.zeros(3), np.cumsum(dpos, axis=0)]) - position))
    assert p_error < 1e-10 and r_error < 1e-10
    return position, local_rot, dpos, drot, {"position_max_error_recorded_units": float(p_error), "rotation_max_error_rad": float(r_error)}



def motion_prior_arrays(source_states, task):
    """Task-normalized measured motion; not calibrated source commands or hole poses.

    Uniform translation scaling cancels the unknown recorded length unit. Rotation
    log magnitude is reduced to the target's initial angular span. Recompute LEFT
    deltas after retargeting; no clipping, joint replay or fabricated terminal action.
    """
    trajectories = [relative_trajectory(state) for state in source_states]
    radius = max(np.linalg.norm(p, axis=1).max() for p, *_ in trajectories)
    peak_delta = max(np.abs(dp).max() for _, _, dp, _, _ in trajectories)
    if radius <= 0 or peak_delta <= 0:
        raise ValueError("SOURCE_TRANSLATION_REQUIRED")
    position_scale = min(2 * task['hole_length_m'] / radius,
                         task['translation_limit_m'] / peak_delta) * (1 - 1e-8)
    angle = max(rot.magnitude().max() for _, rot, *_ in trajectories)
    rotation_scale = min(1., np.linalg.norm(task['rotation_randomization_rad']) / angle) if angle else 1.
    goal = np.array([0., 0., (task['min_depth_m'] + task['max_depth_m']) / 2])
    results = []
    for pos, rot, *_ in trajectories:
        mapped_pos = pos * position_scale + goal
        mapped_rot = Rotation.from_rotvec(rot.as_rotvec() * rotation_scale)
        action = np.column_stack([np.diff(mapped_pos, axis=0),
            (mapped_rot[1:] * mapped_rot[:-1].inv()).as_rotvec()])
        limits = np.array([task['translation_limit_m']] * 3 + [task['rotation_limit_rad']] * 3)
        if not np.isfinite(action).all() or np.any(np.abs(action) > limits + 1e-10):
            raise ValueError("RETARGETED_MOTION_EXCEEDS_ACTION_BOUNDS")
        state = np.column_stack([mapped_pos, mapped_rot.as_rotvec()])
        np.testing.assert_allclose(state[-1], np.r_[goal, np.zeros(3)], atol=1e-10)
        results.append((state, action))
    return results, {'position_scale_m_per_recorded_unit': float(position_scale),
        'rotation_log_scale': float(rotation_scale), 'target_goal_position_m': goal.tolist(),
        'maximum_position_span_m': 2 * task['hole_length_m'],
        'formula': 'p_target = s_p R_final^-1(p-p_final) + goal; R_target = Exp(s_r Log(R_final^-1 R))',
        'action': '[p_target[t+1]-p_target[t], Log(R_target[t+1] R_target[t]^-1)]',
        'timing_hz': 15, 'clipping': False, 'interpolation': False,
        'meaning': 'scaled observed EEF motion prior; NOT measured source commands, calibrated hole coordinates or successful UR10e demonstrations',
        'source_length_unit': 'unverified; normalization is invariant to a uniform change of recorded length unit'}


def build_motion_prior(task, download=False):
    _, files = acquire(download)
    source = [np.stack(pd.read_parquet(CACHE / f'data/chunk-000/episode_{e:06d}.parquet')['observation.state']) for e in EPISODES]
    mapped, transform = motion_prior_arrays(source, task)
    folder = ROOT / 'artifacts/datasets/fire_motion_prior'
    folder.mkdir(parents=True, exist_ok=True)
    state_columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    action_columns = ['dx', 'dy', 'dz', 'dRx', 'dRy', 'dRz']
    episodes = []
    for episode, (state, action) in zip(EPISODES, mapped, strict=True):
        columns = {'source_frame': np.arange(len(state)), 'time_s': np.arange(len(state))/15}
        columns.update({key: state[:, i] for i, key in enumerate(state_columns)})
        columns.update({key: np.r_[action[:, i], np.nan] for i, key in enumerate(action_columns)})
        path = folder / f'episode_{episode:06d}.csv'
        pd.DataFrame(columns).to_csv(path, index=False, lineterminator='\n')
        video = f'videos/chunk-000/observation.images.wrist_view/episode_{episode:06d}.mp4'
        episodes.append({'episode': episode, 'frames': len(state), 'supervised_transitions': len(action),
            'trajectory': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'camera': 'wrist_view', 'source_video': video, 'source_video_sha256': sha(CACHE / video),
            'source_frame_range_half_open': [0, len(state)], 'source_parquet': f'data/chunk-000/episode_{episode:06d}.parquet'})
    from dataset import stats
    import scipy
    report = {'dataset': DATASET, 'revision': REVISION, 'license': 'Apache-2.0',
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'source_type': 'DERIVED_PUBLIC_REAL_POSE_PRIOR',
        'source_files': [f for f in files if f['source_file'].startswith('data/') or '.wrist_view/' in f['source_file']],
        'state_columns': state_columns, 'action_columns': action_columns,
        'episodes': episodes, 'transform': transform, 'target_task': task,
        'task_equivalence': 'ANALOGOUS_TASK', 'source_commands_used': False,
        'action_statistics': stats(np.concatenate([a for _, a in mapped])),
        'last_action': 'absent; final pose retained for visualization, excluded from supervised transitions',
        'script_sha256': sha(Path(__file__)), 'policy_training': 'NOT_RUN by this generator',
        'credit': 'references/FIRe-source-credit.md'}
    dump(folder / 'manifest.json', report)
    return report


def frame(path, index):
    encoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
        f"select=eq(n\\,{index})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
    return Image.open(io.BytesIO(encoded)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument('--motion-prior', action='store_true', help='Generate bounded task-normalized measured pose transitions.')
    args = parser.parse_args()
    if args.motion_prior:
        from insertion import CONFIG
        report = build_motion_prior(CONFIG, args.download)
        print(json.dumps({'episodes': len(report['episodes']), 'transform': report['transform']}))
        return
    hub, files = acquire(args.download)
    info = json.loads((CACHE / "meta/info.json").read_text(encoding="utf-8"))
    candidates = []
    canvas = Image.new("RGB", (1024, 9 * 282), "#161b22")
    draw = ImageDraw.Draw(canvas)
    trajectory_dir = ROOT / "artifacts/datasets/fire_trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for e_index, episode in enumerate(EPISODES):
        data = pd.read_parquet(CACHE / f"data/chunk-000/episode_{episode:06d}.parquet")
        state = np.stack(data["observation.state"])
        actions = np.stack(data["action"])
        times = data["timestamp"].to_numpy()
        assert len(data) == 128 and np.allclose(np.diff(times), 1 / 15, atol=1e-6)
        assert np.isfinite(actions).all() and actions.shape == (128, 7)
        pos, rot, dp, dr, errors = relative_trajectory(state)
        columns = {"frame": np.arange(len(data)), "timestamp_s": times}
        for i, k in enumerate(("x", "y", "z")):
            columns[k + "_recorded_units"] = pos[:, i]
            columns["rotation_" + k + "_rad"] = rot.as_rotvec()[:, i]
            columns["delta_" + k + "_recorded_units"] = np.r_[dp[:, i], np.nan]
            columns["delta_rotation_" + k + "_rad"] = np.r_[dr[:, i], np.nan]
        path = trajectory_dir / f"episode_{episode:06d}.csv"
        pd.DataFrame(columns).to_csv(path, index=False, lineterminator="\n")
        views = []
        for v_index, view in enumerate(VIEWS):
            name = f"videos/chunk-000/observation.images.{view}/episode_{episode:06d}.mp4"
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(CACHE / name)]))
            video = next(x for x in probe["streams"] if x["codec_type"] == "video")
            assert int(video["nb_frames"]) == len(data) and video["r_frame_rate"] == "15/1"
            assert video["width"] == 256 and video["height"] == 256
            views.append({"camera": view, "source_file": name, "frames": 128, "fps": "15/1", "resolution": [256, 256]})
            row = e_index * 3 + v_index
            for col, f in enumerate((0, 42, 84, 127)):
                canvas.paste(frame(CACHE / name, f), (col * 256, row * 282 + 26))
                draw.text((col * 256 + 4, row * 282 + 6), f"E{episode:02} {view} frame {f}", fill="white")
        candidates.append({"episode": episode, "frames": len(data), "views": views,
            "source_label": {"final_reward": float(data["next.reward"].iloc[-1]), "final_done": bool(data["next.done"].iloc[-1]),
                "final_validity": bool(data["annotation.human.validity"].iloc[-1]), "meaning": "publisher label, not independent geometric success verification"},
            "sample_first_state": state[0].tolist(), "sample_first_action": actions[0].tolist(), "state_shape": list(state.shape), "recorded_action_shape": list(actions.shape),
            "recorded_action_min": actions.min(0).tolist(), "recorded_action_max": actions.max(0).tolist(),
            "trajectory": str(path.relative_to(ROOT)).replace("\\", "/"), "trajectory_sha256": sha(path), "reconstruction": errors})
    canvas.save(CACHE / "candidate_contact_sheet.jpg", quality=94)
    output = ROOT / "media/output/source"
    output.mkdir(parents=True, exist_ok=True)
    selected = CACHE / "videos/chunk-000/observation.images.wrist_view/episode_000029.mp4"
    clip = output / "real_franka_e29_wrist_f000-111.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(selected), "-vf", "trim=start_frame=0:end_frame=112,setpts=PTS-STARTPTS",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip)], check=True)
    report = {"dataset": DATASET, "revision": REVISION, "source_type": "PUBLIC_REAL_DATA",
        "license": hub.get("cardData", {}).get("license"), "license_evidence": "pinned dataset card; no standalone LICENSE or NOTICE in Hub inventory",
        "credit": "FIRe real robot source dataset, bhe1004/FIRe-VLA-GR00T-PegInsert-Real (third-party research data)",
        "robot_card_label": info.get("robot_type"), "robot_hardware_verification": "UNRESOLVED: related deployment describes FR3, Hub says Panda",
        "files": files, "candidates": candidates,
        "selection": {"episode": 29, "camera": "wrist_view", "frame_range_half_open": [0, 112], "time_range_s": [0, 112/15],
            "reason": "clearest peg/hole interaction; little terminal yaw relative to other inspected candidates; external right view retained for robot context",
            "local_clip": str(clip.relative_to(ROOT)).replace("\\", "/"), "local_media_sha256": sha(clip),
            "source_file": str(selected.relative_to(CACHE)).replace("\\", "/"), "source_sha256": sha(selected)},
        "transform": {"state": "xyz + quaternion wxyz + two finger positions", "frame": "terminal EEF coordinates derived from final recorded pose; hole pose is unavailable",
            "position": "R_final^-1 (p_t - p_final)", "rotation": "R_final^-1 R_t", "relative_action": "[p_local[t+1]-p_local[t], Log(R_local[t+1] R_local[t]^-1)]",
            "rotation_composition": "left, fixed terminal-EEF frame", "position_units": "recorded units; SI metres suggested by related ROS pose code, not established in dataset metadata",
            "action_status": "observed pose increments, NOT recorded commands, controller targets or verified UR10e labels",
            "last_action": "absent because no next observation; blank CSV entries, not fabricated zero action",
            "camera_projection": "NOT_AVAILABLE; no calibrated trajectory-on-video overlay"},
        "metadata_issue": "info splits train 0:1 conflicts with 30 episode entries; no split semantics inferred",
        "visual_verification": {"candidate_frame_sheet": "inspected 0/42/84/127 for all nine videos", "native_desktop": "NOT_COMPLETED: Computer Use app approval timed out"},
        "transfer_training": "NOT_RUN", "task_equivalence": "ANALOGOUS_TASK"}
    report["generator"] = {"script_sha256": sha(Path(__file__)), "numpy": np.__version__, "pandas": pd.__version__,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}
    frame(selected, 0).save(ROOT / "artifacts/datasets/fire_real_source.png")
    dump(REPORT, report)
    # Actual measured trajectory; no invented hole region or image-space projection.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111, projection="3d")
    for episode in EPISODES:
        d = pd.read_csv(trajectory_dir / f"episode_{episode:06d}.csv")
        xyz = d[[k + "_recorded_units" for k in "xyz"]].to_numpy()
        ax.plot(*xyz.T, label=f"Episode {episode}")
        ax.scatter(*xyz[0], marker="o", s=30)
    ax.scatter(0, 0, 0, marker="x", s=60, c="black", label="Recorded terminal EEF anchor")
    ax.set(xlabel="x (recorded units)", ylabel="y (recorded units)", zlabel="z (recorded units)", title="FIRe measured EEF trajectories / terminal-pose frame")
    all_xyz = np.concatenate([pd.read_csv(trajectory_dir / f"episode_{e:06d}.csv")[[k + "_recorded_units" for k in "xyz"]].to_numpy() for e in EPISODES])
    center = (all_xyz.min(axis=0) + all_xyz.max(axis=0)) / 2
    radius = max(np.ptp(all_xyz, axis=0).max(), 1e-6) * .56
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 1))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(trajectory_dir / "trajectory.png", dpi=150); plt.close(fig)
    print(json.dumps({"report": str(REPORT), "episodes": len(candidates), "videos_verified": 9, "selected_clip_sha256": sha(clip)}))


if __name__ == "__main__":
    main()
