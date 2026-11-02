"""AirSim rays -> world NED -> drone body FRD, in meters."""
import numpy as np
from functools import lru_cache
from scipy.spatial.transform import Rotation


def pose_arrays(pose):
    p, q = pose["position"], pose["orientation_wxyz"]
    position = np.array([p[k] for k in "xyz"], dtype=np.float64)
    quat = np.array([q[k] for k in "xyzw"], dtype=np.float64)
    if not np.isfinite(position).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-10:
        raise ValueError("Invalid camera/vehicle pose")
    return position, Rotation.from_quat(quat).as_matrix()


def check_registration(camera, position_tolerance=.005, angle_tolerance=.005):
    if "rgb_response_pose_world_ned" not in camera:
        raise ValueError("Missing RGB exposure pose; RGB/depth registration cannot be established")
    p_rgb, r_rgb = pose_arrays(camera["rgb_response_pose_world_ned"])
    p_depth, r_depth = pose_arrays(camera["depth_response_pose_world_ned"])
    angle = Rotation.from_matrix(r_rgb.T @ r_depth).magnitude()
    if np.linalg.norm(p_rgb - p_depth) > position_tolerance or angle > angle_tolerance:
        raise ValueError("RGB/depth exposure poses differ; capture registered images while paused")


@lru_cache(maxsize=32)
def _pixel_rays(width, height, stride, intrinsics, semantics):
    """Cache calibration rays, never depth-dependent 3D scene points."""
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    v, u = v.ravel(), u.ravel()
    K = np.asarray(intrinsics).reshape(3, 3)
    optical = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(K).T
    rays = optical[:, [2, 0, 1]]
    if semantics == 'ray_distance':
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    for value in (v, u, rays):
        value.setflags(write=False)
    return v, u, rays


def calibrated_rays(depth, camera, stride):
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('Require an integer stride >= 1')
    if depth.shape != (camera['height'], camera['width']):
        raise ValueError('Depth shape differs from camera metadata')
    semantics = camera.get('depth_semantics', 'ray_distance')
    if semantics not in ('ray_distance', 'z_depth'):
        raise ValueError('Unsupported depth semantics')
    K = np.asarray(camera['intrinsics'], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-10:
        raise ValueError('Invalid camera intrinsics')
    return _pixel_rays(camera['width'], camera['height'], stride, tuple(K.ravel()), semantics)


def body_to_action_rotation(body_pose, gravity_aligned=True):
    """Row-vector transform; default up/down follow gravity, front follows yaw."""
    if not gravity_aligned:
        return np.eye(3)
    _, rotation = pose_arrays(body_pose)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    heading = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    return rotation.T @ heading


def action_depth_samples(depth, camera, body_pose, *, stride=1, min_depth=.5,
                         max_depth=80., window_deg=20., action_rotation=None,
                         directions_frd=None, _tolerance=1e-10):
    """Return pixel indices, UAV-centered scalar ranges and direction memberships.

    With camera ray r, depth d, camera origin t and rotation R, a scene location
    would be p=t+d*r*R. Compute only ||p||^2 and p dot each action direction,
    without ever constructing p or a point cloud. Static calibration rays are
    cached. Camera offsets, attitude, range-depth and z-depth are supported.
    """
    from src.actions import DIRECTION_FRD, RADIUS_M
    directions = DIRECTION_FRD if directions_frd is None else np.asarray(directions_frd, float)
    if (directions.ndim != 2 or directions.shape[1] != 3 or not len(directions) or
            not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions, axis=1), 1)):
        raise ValueError('directions_frd must contain unit Nx3 direction vectors')
    if not np.isfinite(window_deg) or not 0 < window_deg <= 180:
        raise ValueError('Require finite 0 < window_deg <= 180')
    if not 0 <= min_depth < max_depth:
        raise ValueError('Require 0 <= min_depth < max_depth')
    check_registration(camera)
    v, u, rays = calibrated_rays(depth, camera, stride)
    depth_values = depth[v, u].astype(float)
    valid = np.isfinite(depth_values) & (depth_values > 0) & (depth_values >= min_depth) & (depth_values < max_depth)
    candidate_count = int(valid.sum())
    p_camera, r_camera = pose_arrays(camera['depth_response_pose_world_ned'])
    p_body, r_body = pose_arrays(body_pose)
    transform = np.eye(3) if action_rotation is None else np.asarray(action_rotation, float)
    origin = (p_camera-p_body) @ r_body @ transform
    ray_rotation = r_camera.T @ r_body @ transform
    norm2 = np.einsum('ij,ij->i', rays, rays)
    valid &= depth_values*depth_values*norm2 <= (RADIUS_M+np.linalg.norm(origin)+_tolerance)**2
    selected = np.flatnonzero(valid)
    r, d = rays[selected], depth_values[selected]
    def dot(vector):
        return r[:, 0]*vector[0] + r[:, 1]*vector[1] + r[:, 2]*vector[2]
    distance2 = d*d*norm2[selected] + 2*d*dot(ray_rotation @ origin) + origin @ origin
    near = (distance2 > 1e-16) & (distance2 <= (RADIUS_M+_tolerance)**2)
    ranges = np.sqrt(np.maximum(distance2, 0))
    threshold = ranges*np.cos(np.radians(window_deg/2))
    membership = np.empty((len(selected), len(directions)), bool)
    for index, direction in enumerate(directions):
        projection = d*dot(ray_rotation @ direction) + origin @ direction
        membership[:, index] = near & (projection >= threshold-_tolerance)
    chosen = membership.any(axis=1)
    selected = selected[chosen]
    return dict(v=v[selected], u=u[selected], distance_m=np.minimum(ranges[chosen], RADIUS_M),
                membership=membership[chosen], valid_depth_pixels=candidate_count,
                selected_depth_pixels=len(selected))


def select_action_pixels(depth, camera, body_pose, **kwargs):
    """Compatibility ROI mask for callers that explicitly need sparse point clouds."""
    samples = action_depth_samples(depth, camera, body_pose, _tolerance=1e-5, **kwargs)
    mask = np.zeros(depth.shape, bool)
    mask[samples['v'], samples['u']] = True
    return mask, {key: samples[key] for key in ('valid_depth_pixels', 'selected_depth_pixels')}


def backproject(depth, camera, body_pose, stride=1, min_depth=.5, max_depth=80., pixel_mask=None):
    if stride < 1 or not 0 <= min_depth < max_depth:
        raise ValueError("Require stride >= 1 and 0 <= min_depth < max_depth")
    if depth.shape != (camera["height"], camera["width"]):
        raise ValueError("Depth shape differs from camera metadata")
    if camera.get("depth_semantics", "ray_distance") not in ("ray_distance", "z_depth"):
        raise ValueError("Unsupported depth semantics")
    check_registration(camera)
    K = np.asarray(camera["intrinsics"], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-10:
        raise ValueError("Invalid camera intrinsics")
    v, u, rays = calibrated_rays(depth, camera, stride)
    d = depth[v, u]
    valid = np.isfinite(d) & (d > 0) & (d >= min_depth) & (d < max_depth)
    if pixel_mask is not None:
        if pixel_mask.shape != depth.shape or pixel_mask.dtype != bool:
            raise ValueError('Expected a boolean pixel mask matching depth')
        valid &= pixel_mask[v, u]
    rays = rays[valid]
    v, u, d = v[valid], u[valid], d[valid]
    camera_points = rays * d[:, None]
    p_cam, r_cam = pose_arrays(camera["depth_response_pose_world_ned"])
    p_body, r_body = pose_arrays(body_pose)
    world = camera_points @ r_cam.T + p_cam
    body = (world - p_body) @ r_body
    return body.astype(np.float32), v, u


def voxel_representatives(xyz, semantic, confidence, voxel_size):
    if voxel_size < 0:
        raise ValueError("voxel_size must be >= 0")
    if voxel_size == 0 or not len(xyz):
        return np.arange(len(xyz))
    # Preserve class boundaries; retain a real highest-confidence point per class/voxel.
    scaled = np.asarray(xyz, dtype=np.float64) / voxel_size
    nearest = np.rint(scaled)
    # Back-projected coordinates are float32. Cancellation near grid planes
    # (especially +/-0 after world -> body) must not split the same surface.
    tolerance = 2 * np.finfo(np.float32).eps * np.maximum(1, np.abs(scaled))
    scaled = np.where(np.abs(scaled - nearest) <= tolerance, nearest, scaled)
    keys = np.column_stack((np.floor(scaled).astype(np.int64), semantic))
    order = np.argsort(-confidence, kind="stable")
    _, first = np.unique(keys[order], axis=0, return_index=True)
    return np.sort(order[first])
