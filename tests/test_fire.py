"""Independent checks for source pose conventions and corrupted source rejection."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local"))
from fire import relative_trajectory, motion_prior_arrays
from insertion import CONFIG


class SourceTransformTests(unittest.TestCase):
    def test_noncommuting_spatial_rotation_and_fixed_frame_translation(self):
        r0 = Rotation.from_euler("xyz", [.4, -.2, .7])
        step = Rotation.from_rotvec([.15, -.08, .04])
        r1 = step * r0
        rotations = Rotation.from_quat(np.stack([r0.as_quat(), r1.as_quat()]))
        state = np.zeros((2, 9))
        state[:, :3] = [[.2, -.1, .8], [.21, -.12, .77]]
        state[:, 3:7] = rotations.as_quat()[:, [3, 0, 1, 2]]
        pos, local, dp, dr, errors = relative_trajectory(state)
        np.testing.assert_allclose(pos[-1], 0, atol=1e-14)
        np.testing.assert_allclose(dp[0], r1.inv().apply([.01, -.02, -.03]), atol=1e-14)
        np.testing.assert_allclose(dr[0], r1.inv().apply(step.as_rotvec()), atol=1e-14)
        self.assertLess(errors["rotation_max_error_rad"], 1e-14)
        # Sign-equivalent quaternions must not create a false 2*pi action.
        state[0, 3:7] *= -1
        np.testing.assert_allclose(relative_trajectory(state)[3], dr, atol=1e-14)

    def test_retargeting_is_unit_and_world_frame_invariant(self):
        state = np.zeros((4, 9))
        state[:, :3] = [[.2, .1, .4], [.203, .099, .402], [.204, .101, .405], [.205, .102, .407]]
        rot = Rotation.from_rotvec([[.02, -.01, .1], [.01, -.02, .07], [.01, 0, .02], [0, 0, 0]])
        state[:, 3:7] = rot.as_quat()[:, [3, 0, 1, 2]]
        mapped, _ = motion_prior_arrays([state], CONFIG)
        changed = state.copy()
        world = Rotation.from_euler('xyz', [.3, -.5, .7])
        changed[:, :3] = world.apply(state[:, :3] * 1000) + [100, -50, 20]
        changed[:, 3:7] = (world * rot).as_quat()[:, [3, 0, 1, 2]]
        equivalent, _ = motion_prior_arrays([changed], CONFIG)
        for x, y in zip(mapped[0], equivalent[0]):
            np.testing.assert_allclose(x, y, atol=1e-12)
        poses, actions = mapped[0]
        np.testing.assert_allclose(poses[:-1, :3] + actions[:, :3], poses[1:, :3], atol=1e-12)
        predicted = Rotation.from_rotvec(actions[:, 3:]) * Rotation.from_rotvec(poses[:-1, 3:])
        self.assertLess(np.max((predicted.inv() * Rotation.from_rotvec(poses[1:, 3:])).magnitude()), 1e-12)
        self.assertLessEqual(np.abs(actions[:, :3]).max(), CONFIG['translation_limit_m'])
        self.assertLessEqual(np.abs(actions[:, 3:]).max(), CONFIG['rotation_limit_rad'])
        np.testing.assert_allclose(poses[-1], [0, 0, .03, 0, 0, 0], atol=1e-12)

    def test_invalid_source_state_is_rejected(self):
        bad = np.zeros((2, 9))
        with self.assertRaises(ValueError): relative_trajectory(bad)
        bad[:, 3] = 1
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError): relative_trajectory(bad)


if __name__ == "__main__":
    unittest.main()
