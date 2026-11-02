import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.geometry import backproject, voxel_representatives


def pose(position=(0, 0, 0), rotation=None):
    x, y, z, w = (rotation or Rotation.identity()).as_quat()
    return {"position": dict(zip("xyz", position)), "orientation_wxyz": dict(zip("wxyz", (w, x, y, z)))}


def camera(camera_pose):
    return {"width": 3, "height": 3, "intrinsics": [[1., 0, 1], [0, 1., 1], [0, 0, 1.]],
            "rgb_response_pose_world_ned": camera_pose, "depth_response_pose_world_ned": camera_pose,
            "depth_semantics": "ray_distance"}


class GeometryTests(unittest.TestCase):
    def test_zero_depth_stays_invalid_when_minimum_is_zero(self):
        xyz, _, _ = backproject(np.zeros((3, 3), np.float32), camera(pose()), pose(), min_depth=0)
        self.assertEqual(xyz.shape, (0, 3))

    def test_range_is_normalized_and_invalid_pixels_are_removed(self):
        depth = np.full((3, 3), 10., dtype=np.float32)
        depth[0, 0] = np.nan
        depth[0, 1] = 0
        depth[0, 2] = 100
        xyz, v, u = backproject(depth, camera(pose()), pose())
        self.assertEqual(len(xyz), 6)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 10, atol=1e-6)
        np.testing.assert_allclose(xyz[(v == 1) & (u == 1)], [[10, 0, 0]])
        np.testing.assert_allclose(xyz[(v == 2) & (u == 2)], np.array([[1, 1, 1]]) * 10 / np.sqrt(3), atol=1e-6)

    def test_five_camera_axes_offsets_and_rotated_drone(self):
        body_rotation = Rotation.from_euler("xyz", [12, -8, 73], degrees=True)
        body_position = np.array([300., -120., -35.])
        body = pose(body_position, body_rotation)
        fixtures = [(0, 0, [1, 0, 0], [11, 0, 0]),
                    (-90, 0, [0, -1, 0], [0, -11, 0]),
                    (90, 0, [0, 1, 0], [0, 11, 0]),
                    (180, 0, [-1, 0, 0], [-11, 0, 0]),
                    (0, -90, [0, 0, 0], [0, 0, 10])]
        depth = np.full((3, 3), np.nan, dtype=np.float32)
        depth[1, 1] = 10
        for yaw, pitch, offset, expected in fixtures:
            r = body_rotation * Rotation.from_euler("xyz", [0, pitch, yaw], degrees=True)
            p = body_position + body_rotation.apply(offset)
            xyz, _, _ = backproject(depth, camera(pose(p, r)), body)
            np.testing.assert_allclose(xyz, [expected], atol=1e-5)

    def test_voxel_preserves_classes_and_strongest_point(self):
        xyz = np.array([[.01, 0, 0], [.02, 0, 0], [.03, 0, 0]])
        chosen = voxel_representatives(xyz, np.array([1, 1, 2]), np.array([.3, .8, .4]), .1)
        np.testing.assert_array_equal(chosen, [1, 2])

    def test_misaligned_rgb_depth_is_rejected(self):
        info = camera(pose())
        info["rgb_response_pose_world_ned"] = pose([1, 0, 0])
        with self.assertRaisesRegex(ValueError, "poses differ"):
            backproject(np.ones((3, 3)), info, pose())

    def test_voxel_boundary_is_stable_under_coordinate_roundoff(self):
        local = np.array([[0., 0., 1.], [.05, 0., 1.]], np.float32)
        transformed = local.copy()
        transformed[:, 1] = [-1e-14, 1e-14]
        for xyz in (local, transformed):
            chosen = voxel_representatives(xyz, np.ones(2), np.array([.9, .8]), .15)
            np.testing.assert_array_equal(chosen, [0])
        # Adjacent float32 representations of the same mathematical grid plane.
        boundary = np.full((2, 3), .15, np.float32)
        boundary[0] = np.nextafter(boundary[0], np.float32(0))
        chosen = voxel_representatives(boundary, np.ones(2), np.array([.8, .9]), .15)
        np.testing.assert_array_equal(chosen, [1])


if __name__ == "__main__":
    unittest.main()
