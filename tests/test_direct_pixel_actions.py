"""Direct scalar geometry agrees with a separate full-cloud oracle, without voxel thinning."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.actions import action_membership, pool_action_samples, query_actions
from src.geometry import action_depth_samples, backproject, body_to_action_rotation
from src.openuav.dataset import pose_dict


@pytest.mark.parametrize('semantics', ['ray_distance', 'z_depth'])
@pytest.mark.parametrize('stride', [1, 3])
def test_direct_ranges_and_action_pixels_match_full_geometry_oracle(semantics, stride):
    rng = np.random.default_rng(27)
    body_rotation = Rotation.from_euler('xyz', [23, -17, 65], degrees=True)
    position = np.array([300., -100., -20.])
    body = pose_dict(position, body_rotation)
    rotation = body_to_action_rotation(body)
    for yaw, pitch, offset in ((0, 0, [1, 0, 0]), (0, -90, [0, 0, 0]), (-90, 0, [0, -1, 0])):
        pose = pose_dict(position+body_rotation.apply(offset),
                         body_rotation*Rotation.from_euler('xyz', [0, pitch, yaw], degrees=True))
        camera = dict(width=65, height=49, intrinsics=[[24, 0, 32], [0, 24, 24], [0, 0, 1]],
                      rgb_response_pose_world_ned=pose, depth_response_pose_world_ned=pose,
                      depth_semantics=semantics)
        depth = rng.uniform(.6, 25, (49, 65)).astype(np.float32)
        depth[0, 0], depth[1, 1] = np.nan, 0
        semantic = rng.integers(0, 3, depth.shape)
        sample = action_depth_samples(depth, camera, body, stride=stride, action_rotation=rotation)
        # This full back-projection is only a test oracle; production uses scalar samples.
        xyz, v, u = backproject(depth, camera, body, stride=stride)
        xyz = xyz@rotation
        membership = action_membership(xyz)
        selected = membership.any(axis=1)
        np.testing.assert_array_equal(sample['v'], v[selected])
        np.testing.assert_array_equal(sample['u'], u[selected])
        np.testing.assert_array_equal(sample['membership'], membership[selected])
        np.testing.assert_allclose(sample['distance_m'], np.linalg.norm(xyz[selected], axis=1), atol=2e-6)
        direct = pool_action_samples(sample['distance_m'], semantic[sample['v'], sample['u']], sample['membership'], 3)
        reference = query_actions(xyz, semantic[v, u], 3)
        np.testing.assert_array_equal(direct['semantic_votes'], reference['semantic_votes'])
        np.testing.assert_allclose(direct['depth_m'], reference['depth_m'], atol=3e-6, equal_nan=True)
        assert 'xyz' not in sample


def test_camera_offset_is_used_for_uav_range_and_15m_boundary():
    body = pose_dict([0, 0, 0], Rotation.identity())
    pose = pose_dict([-1, 0, 0], Rotation.identity())
    camera = dict(width=1, height=1, intrinsics=np.eye(3).tolist(),
                  rgb_response_pose_world_ned=pose, depth_response_pose_world_ned=pose)
    sample = action_depth_samples(np.array([[16.]], np.float32), camera, body)
    assert sample['distance_m'].tolist() == [15.]
    assert sample['membership'].tolist() == [[True, False, False, False]]
    assert action_depth_samples(np.array([[16.01]], np.float32), camera, body)['selected_depth_pixels'] == 0


def test_custom_direction_ranges_and_membership_match_independent_full_geometry():
    from src.actions import direction_vectors
    body_rotation = Rotation.from_euler('xyz', [17, -13, 72], degrees=True)
    body = pose_dict([200, -90, -25], body_rotation)
    camera_pose = pose_dict([200.3, -90.2, -24.9], body_rotation)
    camera = dict(width=35, height=27, intrinsics=[[13, 0, 17], [0, 13, 13], [0, 0, 1]],
                  rgb_response_pose_world_ned=camera_pose, depth_response_pose_world_ned=camera_pose,
                  depth_semantics='z_depth')
    queries = [dict(name='a', yaw_deg=-17, elevation_deg=23), dict(name='b', yaw_deg=21, elevation_deg=-12)]
    axes = direction_vectors(queries)
    depth = np.random.default_rng(11).uniform(1, 18, (27, 35)).astype(np.float32)
    rotation = body_to_action_rotation(body)
    sample = action_depth_samples(depth, camera, body, window_deg=31, action_rotation=rotation, directions_frd=axes)
    xyz, v, u = backproject(depth, camera, body)
    xyz = xyz@rotation
    distance = np.linalg.norm(xyz, axis=1)
    membership = ((distance[:, None] <= 15) & (distance[:, None] > 0) &
                  (xyz@axes.T >= distance[:, None]*np.cos(np.radians(15.5))))
    keep = membership.any(axis=1)
    np.testing.assert_array_equal(sample['v'], v[keep])
    np.testing.assert_array_equal(sample['u'], u[keep])
    np.testing.assert_array_equal(sample['membership'], membership[keep])
    np.testing.assert_allclose(sample['distance_m'], distance[keep], atol=3e-6)
