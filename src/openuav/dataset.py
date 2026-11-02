"""Normalize an existing TravelUAV frame; explicitly mark quantized depth."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from src.config import ROOT
from src.openuav.capture import CameraCapture, OPENUAV_CAMERAS, intrinsics_from_horizontal_fov


def pose_dict(position, rotation):
    x, y, z, w = rotation.as_quat()
    return {"position": dict(zip("xyz", map(float, position))),
            "orientation_wxyz": dict(zip("wxyz", map(float, (w, x, y, z))))}


def load_dataset_frame(trajectory, frame, fov_deg=90, layout_path=None, allow_missing_pose=False, cameras=OPENUAV_CAMERAS):
    trajectory = Path(trajectory)
    frame = str(frame).zfill(6)
    log_path = trajectory / "log" / f"{frame}.json"
    world_pose_known = log_path.is_file()
    if world_pose_known or not allow_missing_pose:
        state = json.loads(log_path.read_text())["sensors"]["state"]
        # TravelUAV logs are [x,y,z,w], matching AirSim orientation.to_numpy_array().
        rotation = Rotation.from_quat(state["orientation"])
        position = np.asarray(state["position"], dtype=float)
    else:
        # Only relative camera-to-body geometry is needed for body-FRD output.
        # This virtual reference does not claim a measured world position/yaw.
        rotation, position = Rotation.identity(), np.zeros(3)
    layout_path = Path(layout_path or ROOT / "config/openuav_cameras.json")
    layout = json.loads(layout_path.read_text())["cameras"]
    captures = []
    for camera in cameras:
        name = camera.lower()
        rgb = np.array(Image.open(trajectory / name / f"{frame}.png").convert("RGB"))
        encoded = np.array(Image.open(trajectory / f"{name}_depth" / f"{frame}.png"))
        if encoded.ndim != 2 or encoded.dtype != np.uint8 or encoded.shape != rgb.shape[:2]:
            raise ValueError(f"Expected registered single-channel uint8 depth for {camera}")
        depth = encoded.astype(np.float32) * (100.0 / 255.0)
        depth[(encoded == 0) | (encoded == 255)] = np.nan
        extrinsic = layout[camera]
        camera_rotation = rotation * Rotation.from_euler("xyz", extrinsic["rpy_deg"], degrees=True)
        camera_position = position + rotation.apply(extrinsic["position"])
        height, width = depth.shape
        metadata = {
            "camera_name": camera, "width": width, "height": height, "fov_deg": fov_deg,
            "intrinsics": intrinsics_from_horizontal_fov(width, height, fov_deg).tolist(),
            "depth_response_pose_world_ned": pose_dict(camera_position, camera_rotation),
            "rgb_response_pose_world_ned": pose_dict(camera_position, camera_rotation),
            "depth_semantics": "ray_distance", "depth_unit": "meter",
            "world_frame": "NED" if world_pose_known else "body_reference",
            "color_order": "RGB", "calibration_source": str(layout_path),
            "extrinsics_assumed_from_openuav_settings": True,
        }
        captures.append(CameraCapture(camera, rgb, depth, metadata))
    metadata = {
        "schema_version": 1, "source": "traveluav_dataset", "trajectory": str(trajectory),
        "frame": frame, "depth_quality": "quantized_uint8_0_100m",
        "depth_quantization_step_m": 100.0 / 255.0,
        "saturated_depth_discarded": True, "pose_synchronized_by_pause": False,
        "vehicle_pose_world_ned": pose_dict(position, rotation),
        "world_pose_known": world_pose_known,
        "vehicle_pose_reference": "dataset_log" if world_pose_known else "body_origin_identity",
        "calibration_assumption": "Standard OpenUAV camera layout and equal Scene/depth FOV",
    }
    return captures, metadata
