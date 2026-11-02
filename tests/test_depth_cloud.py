import hashlib
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from src.cloud import export, fuse, render, save_depth_cloud
from src.config import ROOT, write_json
from src.openuav.capture import CameraCapture, OPENUAV_CAMERAS, save_capture


SETTINGS = {"stride": 1, "min_depth": .5, "max_depth": 80, "voxel_size": 0, "render_max_points": 100}


def pose(position, rotation):
    quaternion = rotation.as_quat()
    return {"position": dict(zip("xyz", map(float, position))),
            "orientation_wxyz": dict(zip("xyzw", map(float, quaternion)))}


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "capture"
    body_rotation = Rotation.from_euler("xyz", [12, -8, 73], degrees=True)
    body_position = np.array([300., -120., -35.])
    layout = [(0, 0, [1, 0, 0]), (-90, 0, [0, -1, 0]), (90, 0, [0, 1, 0]),
              (180, 0, [-1, 0, 0]), (0, -90, [0, 0, 0])]
    observations = []
    for name, (yaw, pitch, offset) in zip(OPENUAV_CAMERAS, layout):
        camera_pose = pose(body_position + body_rotation.apply(offset),
                           body_rotation * Rotation.from_euler("xyz", [0, pitch, yaw], degrees=True))
        info = {"width": 3, "height": 1, "intrinsics": [[1., 0, 1.], [0, 1., 0], [0, 0, 1.]],
                "rgb_response_pose_world_ned": camera_pose, "depth_response_pose_world_ned": camera_pose,
                "depth_semantics": "ray_distance"}
        observations.append(CameraCapture(name, np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8),
                                          np.array([[0., 10., 20.]], np.float32), info))
    save_capture(observations, root, {"vehicle_pose_world_ned": pose(body_position, body_rotation),
                                     "source": "test_fixture", "depth_quality": "metric_float"})
    return root


def read_ply(path):
    content = path.read_bytes()
    header, body = content.split(b"end_header\n", 1)
    return header.decode("ascii"), body


def test_standard_depth_export_has_correct_body_coordinates_colors_and_artifacts(capture):
    summary = save_depth_cloud(capture, SETTINGS)
    root = capture / "depth_cloud"
    assert summary["points"] == 10 and summary["status"] == "generated"
    data = np.load(root / "cloud.npz")
    assert set(data.files) == {"xyz", "rgb", "view_id"}
    assert data["xyz"].dtype == np.float32 and data["rgb"].dtype == np.uint8
    expected = [[11, 0, 0], [0, -11, 0], [0, 11, 0], [-11, 0, 0], [0, 0, 10]]
    np.testing.assert_allclose(data["xyz"][::2], expected, atol=1e-5)
    np.testing.assert_array_equal(data["rgb"], np.tile([[0, 255, 0], [0, 0, 255]], (5, 1)))
    np.testing.assert_array_equal(data["view_id"], np.repeat(np.arange(5), 2))
    header, body = read_ply(root / "rgb.ply")
    assert "element vertex 10" in header and "semantic_id" not in header
    assert len(body) == 10 * (12 + 3 + 1)
    metadata = json.loads((root / "meta.json").read_text())
    assert metadata["kind"] == "rgb_depth" and metadata["frame"] == "drone_body_FRD"
    assert metadata["source_capture"] == str(capture.resolve())
    assert "classes" not in metadata
    assert "RGB depth cloud" in (root / "cloud.html").read_text()
    Image.open(root / "cloud.png").verify()


@pytest.mark.parametrize("value,status,points", [(1., "depth_suspect", 15), (np.nan, "empty", 0)])
def test_bad_depth_still_has_standard_diagnostic_outputs(capture, value, status, points):
    for name in OPENUAV_CAMERAS:
        np.save(capture / name / "depth_perspective_m.npy", np.full((1, 3), value, np.float32))
    summary = save_depth_cloud(capture, SETTINGS)
    root = capture / "depth_cloud"
    assert summary["status"] == status and summary["points"] == points
    assert summary["depth_suspect"] is True
    assert np.load(root / "cloud.npz")["xyz"].shape == (points, 3)
    assert f"element vertex {points}" in read_ply(root / "rgb.ply")[0]
    assert "SUSPECT DEPTH" in (root / "cloud.html").read_text()
    Image.open(root / "cloud.png").verify()


def test_semantic_export_remains_compatible_with_depth_mode(capture, tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    segmentation = {"classes": ["unknown", "building"], "colors": [[128, 128, 128], [12, 34, 56]],
                    "capture": str(capture), "gsa_commit": "fixture",
                    "capture_meta_sha256": hashlib.sha256((capture / "meta.json").read_bytes()).hexdigest()}
    write_json(masks / "meta.json", segmentation)
    for name in OPENUAV_CAMERAS:
        (masks / name).mkdir()
        for field, values in (("semantic", np.ones((1, 3), np.uint16)),
                              ("instances", np.full((1, 3), 3, np.uint16)),
                              ("confidence", np.full((1, 3), .9, np.float32))):
            np.save(masks / name / f"{field}.npy", values)
    data, metadata, seg, views, counts = fuse(capture, masks, SETTINGS)
    assert data["xyz"].shape == (10, 3)
    np.testing.assert_array_equal(data["semantic_id"], np.ones(10))
    output = tmp_path / "semantic_cloud"
    export(output, data, metadata, seg, views, counts, SETTINGS)
    render(output, data, seg, 100, metadata["cloud_diagnostics"])
    header, body = read_ply(output / "semantic.ply")
    assert "property ushort semantic_id" in header and len(body) == 10 * 24
    assert json.loads((output / "meta.json").read_text())["classes"] == segmentation["classes"]
    Image.open(output / "cloud.png").verify()


def test_cli_accepts_existing_capture_without_masks(capture, tmp_path):
    output = tmp_path / "cloud"
    command = [sys.executable, "scripts/fuse_cloud.py", "--config", "config/resources.example.json",
               "--input", str(capture), "--output", str(output), "--stride", "1", "--voxel-size", "0", "--no-render"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "meta.json").read_text())["points"] == 10
    result = subprocess.run(command + ["--labeled-only", "--output", str(tmp_path / "invalid")], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "--labeled-only requires --masks" in result.stderr
