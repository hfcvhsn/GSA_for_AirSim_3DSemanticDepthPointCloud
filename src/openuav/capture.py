from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import json
import math

import airsim
import numpy as np


OPENUAV_CAMERAS = (
    "FrontCamera",
    "LeftCamera",
    "RightCamera",
    "RearCamera",
    "DownCamera",
)


@dataclass(frozen=True)
class CameraCapture:
    name: str
    rgb: np.ndarray
    depth_perspective_m: np.ndarray
    metadata: Dict


def _vector3_to_dict(v) -> Dict[str, float]:
    return {"x": float(v.x_val), "y": float(v.y_val), "z": float(v.z_val)}


def _quaternion_to_dict(q) -> Dict[str, float]:
    return {
        "w": float(q.w_val),
        "x": float(q.x_val),
        "y": float(q.y_val),
        "z": float(q.z_val),
    }


def _projection_matrix_to_list(proj_mat) -> List[List[float]]:
    matrix = getattr(proj_mat, "matrix", None)
    if matrix is None:
        return []
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 16:
        arr = arr.reshape(4, 4)
    return arr.tolist()


def intrinsics_from_horizontal_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Build a pinhole intrinsic matrix from AirSim's horizontal field of view."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("fov_deg must be in (0, 180)")

    fov_rad = math.radians(float(fov_deg))
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


class OpenUAVFiveViewCapture:
    """Faithful OpenUAV/TravelUAV five-camera RGB-D capture wrapper.

    The camera names and request ordering mirror TravelUAV's
    ``AirVLNSimulatorClientTool.getImageResponses``: for each camera, request an
    uncompressed Scene image followed by a floating-point DepthPerspective image
    in one ``simGetImages`` call.
    """

    def __init__(
        self,
        client: airsim.MultirotorClient,
        vehicle_name: str = "",
        cameras: Iterable[str] = OPENUAV_CAMERAS,
    ) -> None:
        self.client = client
        self.vehicle_name = vehicle_name
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("at least one camera is required")

    def _requests(self) -> List[airsim.ImageRequest]:
        requests: List[airsim.ImageRequest] = []
        for camera_name in self.cameras:
            requests.append(
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False,
                )
            )
            requests.append(
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=True,
                    compress=False,
                )
            )
        return requests

    def capture(self, pause_sim: bool = True) -> List[CameraCapture]:
        """Capture one five-view observation and preserve metric depth.

        AirSim recommends pausing physics when image/pose synchronization matters.
        The default therefore pauses the simulator around image and camera-info
        queries, then restores normal simulation in ``finally``.
        """
        if pause_sim:
            self.client.simPause(True)

        try:
            responses = self.client.simGetImages(
                self._requests(), vehicle_name=self.vehicle_name
            )
            expected = 2 * len(self.cameras)
            if len(responses) != expected:
                raise RuntimeError(
                    f"expected {expected} AirSim image responses, got {len(responses)}"
                )

            captures: List[CameraCapture] = []
            for index, camera_name in enumerate(self.cameras):
                rgb_resp = responses[2 * index]
                depth_resp = responses[2 * index + 1]

                if rgb_resp.width <= 0 or rgb_resp.height <= 0:
                    raise RuntimeError(f"invalid RGB response for {camera_name}")
                if depth_resp.width <= 0 or depth_resp.height <= 0:
                    raise RuntimeError(f"invalid depth response for {camera_name}")
                if (rgb_resp.width, rgb_resp.height) != (depth_resp.width, depth_resp.height):
                    raise RuntimeError(f"RGB/depth resolution mismatch for {camera_name}")

                rgb = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
                expected_rgb_values = rgb_resp.height * rgb_resp.width * 3
                if rgb.size != expected_rgb_values:
                    raise RuntimeError(
                        f"unexpected RGB byte count for {camera_name}: "
                        f"expected {expected_rgb_values}, got {rgb.size}"
                    )
                rgb = rgb.reshape(rgb_resp.height, rgb_resp.width, 3)

                depth = airsim.list_to_2d_float_array(
                    depth_resp.image_data_float,
                    depth_resp.width,
                    depth_resp.height,
                ).astype(np.float32, copy=False)

                camera_info = self.client.simGetCameraInfo(
                    camera_name,
                    vehicle_name=self.vehicle_name,
                )
                K = intrinsics_from_horizontal_fov(
                    rgb_resp.width,
                    rgb_resp.height,
                    float(camera_info.fov),
                )

                metadata = {
                    "camera_name": camera_name,
                    "image_type_rgb": "Scene",
                    "image_type_depth": "DepthPerspective",
                    "depth_unit": "meter",
                    "world_frame": "NED",
                    "width": int(rgb_resp.width),
                    "height": int(rgb_resp.height),
                    "fov_deg": float(camera_info.fov),
                    "intrinsics": K.tolist(),
                    "camera_info_pose_world_ned": {
                        "position": _vector3_to_dict(camera_info.pose.position),
                        "orientation_wxyz": _quaternion_to_dict(camera_info.pose.orientation),
                    },
                    "rgb_response_pose_world_ned": {
                        "position": _vector3_to_dict(rgb_resp.camera_position),
                        "orientation_wxyz": _quaternion_to_dict(rgb_resp.camera_orientation),
                    },
                    "depth_response_pose_world_ned": {
                        "position": _vector3_to_dict(depth_resp.camera_position),
                        "orientation_wxyz": _quaternion_to_dict(depth_resp.camera_orientation),
                    },
                    "projection_matrix": _projection_matrix_to_list(camera_info.proj_mat),
                    "rgb_timestamp_ns": int(rgb_resp.time_stamp),
                    "depth_timestamp_ns": int(depth_resp.time_stamp),
                }
                captures.append(
                    CameraCapture(
                        name=camera_name,
                        rgb=rgb,
                        depth_perspective_m=depth,
                        metadata=metadata,
                    )
                )
            return captures
        finally:
            if pause_sim:
                self.client.simPause(False)


def save_capture(captures: Iterable[CameraCapture], output_dir: Path) -> None:
    """Persist one observation without quantizing metric depth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_metadata = {"cameras": {}}
    for capture in captures:
        camera_dir = output_dir / capture.name
        camera_dir.mkdir(parents=True, exist_ok=True)

        # Match AirSim/OpenUAV's raw uncompressed image-array convention.  No
        # geometric flip is introduced here, so RGB and depth stay pixel-aligned.
        airsim.write_png(str(camera_dir / "rgb.png"), capture.rgb)
        np.save(camera_dir / "depth_perspective_m.npy", capture.depth_perspective_m)

        with (camera_dir / "camera.json").open("w", encoding="utf-8") as f:
            json.dump(capture.metadata, f, indent=2)

        frame_metadata["cameras"][capture.name] = capture.metadata

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(frame_metadata, f, indent=2)
