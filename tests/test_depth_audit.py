import json

import numpy as np
from PIL import Image
import torch

from scripts.audit_depth import audit_trajectory, summarize


def test_audit_checks_all_images_and_reports_missing_and_corrupt_files(tmp_path):
    trajectory = tmp_path / "Scene" / "trajectory"
    rgb, depth = trajectory / "frontcamera", trajectory / "frontcamera_depth"
    rgb.mkdir(parents=True)
    depth.mkdir()
    for frame in (1, 2, 3, 4):
        Image.new("RGB", (2, 2)).save(rgb / f"{frame:06}.png")
    Image.fromarray(np.full((2, 2), 2, np.uint8)).save(depth / "000001.png")
    Image.fromarray(np.array([[0, 1], [2, 50]], np.uint8)).save(depth / "000002.png")
    (depth / "000003.png").write_bytes(b"corrupt PNG")
    torch.save(torch.tensor([float("nan"), 1., 4.]), trajectory / "depth_imgs.tensor")
    row = audit_trajectory(trajectory, check_tensors=True)
    view = row["views"]["FrontCamera"]
    assert view["depth_files"] == 3 and view["decoded"] == 2
    assert view["constant_2_images"] == 1
    assert view["missing_for_rgb"] == ["000004.png"]
    assert view["nontrivial_candidates"] == ["000002.png"]
    assert len(row["errors"]) == 1
    summary = summarize([row])["Scene"]
    assert summary["missing_directories"] == 4
    assert summary["histogram"] == {"0": 1, "1": 1, "2": 5, "50": 1}
    assert summary["tensor_min"] == 1 and summary["tensor_max"] == 4
    assert summary["nonfinite_tensors"] == 1
    json.dumps(summary, allow_nan=False)


def test_all_nan_tensor_can_be_reported_without_nonstandard_json(tmp_path):
    trajectory = tmp_path / "Scene" / "trajectory"
    trajectory.mkdir(parents=True)
    torch.save(torch.tensor([float("nan")]), trajectory / "depth_imgs.tensor")
    summary = summarize([audit_trajectory(trajectory, check_tensors=True)])["Scene"]
    assert summary["tensors_checked"] == summary["nonfinite_tensors"] == 1
    assert summary["tensor_min"] is None and summary["tensor_max"] is None
    json.dumps(summary, allow_nan=False)
