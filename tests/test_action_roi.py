"""The sparse projection equals independently filtering a full unthinned cloud."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.actions import action_membership
from src.geometry import backproject, body_to_action_rotation, select_action_pixels
from src.openuav.dataset import pose_dict


@pytest.mark.parametrize('semantics', ['ray_distance', 'z_depth'])
@pytest.mark.parametrize('stride', [1, 3])
def test_scalar_roi_matches_full_reference_with_offsets_and_attitude(semantics, stride):
    rng = np.random.default_rng(42)
    body_r = Rotation.from_euler('xyz', [23, -17, 65], degrees=True)
    body_p = np.array([300., -100., -20.])
    body = pose_dict(body_p, body_r)
    for yaw, pitch, offset in [(0, 0, [1, 0, 0]), (-90, 0, [0, -1, 0]),
                               (90, 0, [0, 1, 0]), (0, -90, [0, 0, 0])]:
        pose = pose_dict(body_p+body_r.apply(offset), body_r*Rotation.from_euler('xyz', [0, pitch, yaw], degrees=True))
        camera = dict(width=65, height=49, intrinsics=[[24, 0, 32], [0, 24, 24], [0, 0, 1]],
                      rgb_response_pose_world_ned=pose, depth_response_pose_world_ned=pose, depth_semantics=semantics)
        depth = rng.uniform(.6, 25, (49, 65)).astype(np.float32)
        depth[0, 0], depth[1, 1] = np.nan, 0
        T = body_to_action_rotation(body, True)
        xyz, v, u = backproject(depth, camera, body, stride=stride)
        expected = action_membership(xyz @ T).any(axis=1)
        mask, stats = select_action_pixels(depth, camera, body, stride=stride, action_rotation=T)
        assert mask[v[expected], u[expected]].all()  # No target pixel may be lost before projection.
        sparse, sv, su = backproject(depth, camera, body, stride=stride, pixel_mask=mask)
        keep = action_membership(sparse @ T).any(axis=1)
        np.testing.assert_array_equal(sv[keep], v[expected])
        np.testing.assert_array_equal(su[keep], u[expected])
        np.testing.assert_allclose(sparse[keep], xyz[expected], atol=1e-6)
        assert stats['selected_depth_pixels'] < stats['valid_depth_pixels']


def test_body_range_includes_points_beyond_15m_camera_depth():
    body = pose_dict([0, 0, 0], Rotation.identity())
    pose = pose_dict([-1, 0, 0], Rotation.identity())
    camera = dict(width=1, height=1, intrinsics=np.eye(3).tolist(),
                  rgb_response_pose_world_ned=pose, depth_response_pose_world_ned=pose, depth_semantics='ray_distance')
    mask, _ = select_action_pixels(np.array([[16.]], np.float32), camera, body)
    assert mask[0, 0]  # Camera depth 16 m is body-centered range 15 m.


def test_pixel_selection_does_not_generate_scene_xyz(monkeypatch):
    body = pose_dict([0, 0, 0], Rotation.identity())
    camera = dict(width=1, height=1, intrinsics=np.eye(3).tolist(),
                  rgb_response_pose_world_ned=body, depth_response_pose_world_ned=body)
    def forbidden(*args, **kwargs):
        raise AssertionError('Full backprojection during pixel selection')
    monkeypatch.setattr('src.geometry.backproject', forbidden)
    mask, _ = select_action_pixels(np.array([[3.]], np.float32), camera, body)
    assert mask[0, 0]
