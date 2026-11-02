from copy import deepcopy
import json
from types import SimpleNamespace as NS

import airsim
import numpy as np
from PIL import Image
import pytest

from scripts.recapture_depth import load_records, recapture
from src.geometry import backproject
from src.openuav.capture import OPENUAV_CAMERAS


class ReplayClient:
    def __init__(self, paused=False, failure=False, constant=False, wrong_pose=False, drift_steps=0):
        self.paused, self.failure = paused, failure
        self.constant, self.wrong_pose = constant, wrong_pose
        self.state = airsim.KinematicsState()
        self.state.position = airsim.Vector3r(8, 9, -10)
        self.state.linear_velocity = airsim.Vector3r(1, 2, 3)
        self.original = deepcopy(self.state)
        self.steps = 0
        self.requests = []
        self.drift_steps = drift_steps

    def simIsPause(self):
        return self.paused

    def simPause(self, paused):
        self.paused = paused

    def simGetGroundTruthKinematics(self, **kwargs):
        return deepcopy(self.state)

    def simSetKinematics(self, state, **kwargs):
        self.state = deepcopy(state)

    def simContinueForFrames(self, frames):
        assert frames == 1 and self.paused
        assert self.state.linear_velocity.get_length() == 0
        self.steps += frames
        if self.steps <= self.drift_steps:
            self.state.position.z_val += 27

    def simGetVehiclePose(self, **kwargs):
        if self.wrong_pose:
            return airsim.Pose(airsim.Vector3r(100, 100, 100), airsim.Quaternionr())
        return airsim.Pose(self.state.position, self.state.orientation)

    def simGetCameraInfo(self, *args, **kwargs):
        return NS(fov=90., pose=self.simGetVehiclePose(), proj_mat=NS(matrix=np.eye(4)))

    def simGetImages(self, requests, **kwargs):
        if self.failure:
            raise RuntimeError("RPC failure")
        assert self.paused
        self.requests.extend(requests)
        common = dict(width=2, height=2, camera_position=self.state.position,
                      camera_orientation=self.state.orientation, time_stamp=100)
        responses = []
        for request in requests:
            if request.image_type == airsim.ImageType.Scene:
                responses.append(NS(**common, image_data_uint8=bytes([0, 0, 255] * 4)))
            else:
                values = [2.] * 4 if self.constant else [1., 2., 3., 4.]
                responses.append(NS(**common, image_data_float=values))
        return responses


@pytest.fixture
def trajectory(tmp_path):
    root = tmp_path / "Scene" / "trajectory"
    (root / "log").mkdir(parents=True)
    (root / "frontcamera").mkdir()
    for frame in (5, 10, 15):
        Image.new("RGB", (2, 2), color="blue").save(root / "frontcamera" / f"{frame:06}.png")
        (root / "log" / f"{frame:06}.json").write_text(json.dumps({"sensors": {"state": {
            "position": [1., 2., -3.], "orientation": [0., 0., 2., 2.]}}}))
    return root


def run_capture(client, trajectory, output, **kwargs):
    return recapture(client, load_records(trajectory, [5]), output, "Drone_1",
                     {"raw_color_order": "BGR", "flip_vertical": False}, trajectory,
                     cloud_settings={"stride": 1, "min_depth": .5, "max_depth": 80, "voxel_size": 0, "render_max_points": 100},
                     render_cloud=False, **kwargs)


def assert_restored(client, paused):
    assert client.paused == paused
    serialize = lambda state: json.dumps(state, default=lambda obj: obj.to_msgpack(), sort_keys=True)
    assert serialize(client.state) == serialize(client.original)


def test_replay_saves_fresh_registered_rgb_and_float_depth_with_planar(trajectory, tmp_path):
    client = ReplayClient()
    output = tmp_path / "capture"
    summary = run_capture(client, trajectory, output, check_planar=True)
    assert summary["status"] == "completed"
    assert client.steps == 2
    assert_restored(client, False)
    frame = output / "000005"
    metadata = json.loads((frame / "meta.json").read_text())
    assert metadata["pose_synchronized_by_pause"] is True
    assert list(metadata["cameras"]) == list(OPENUAV_CAMERAS)
    assert metadata["pose_error_m"] == metadata["pose_error_deg"] == 0
    assert (frame / "source_log.json").read_bytes() == (trajectory / "log/000005.json").read_bytes()
    for name in OPENUAV_CAMERAS:
        depth = np.load(frame / name / "depth_perspective_m.npy")
        assert depth.dtype == np.float32
        np.testing.assert_array_equal(depth, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(np.asarray(Image.open(frame / name / "rgb.png"))[0, 0], [255, 0, 0])
        assert (frame / name / "depth_planar_m.npy").is_file()
        # The diagnostic metadata must remain valid input for the existing fusion.
        xyz, _, _ = backproject(depth, metadata["cameras"][name], metadata["vehicle_pose_world_ned"])
        assert xyz.shape == (4, 3) and np.isfinite(xyz).all()
    assert any(r.image_type == airsim.ImageType.DepthPlanar for r in client.requests)
    assert summary["frames"][0]["depth_cloud"]["points"] == 20
    cloud = np.load(frame / "depth_cloud/cloud.npz")
    assert set(cloud.files) == {"xyz", "rgb", "view_id"}
    assert set(cloud["view_id"]) == set(range(5))
    assert (frame / "depth_cloud/rgb.ply").is_file()


@pytest.mark.parametrize("paused", [False, True])
def test_rpc_failure_restores_original_state_and_pause(trajectory, tmp_path, paused):
    client = ReplayClient(paused=paused, failure=True)
    output = tmp_path / "capture"
    with pytest.raises(RuntimeError, match="RPC failure"):
        run_capture(client, trajectory, output)
    assert_restored(client, paused)
    assert json.loads((output / "recapture.json").read_text())["status"] == "failed"


def test_constant_raw_depth_is_saved_and_flagged(trajectory, tmp_path):
    client = ReplayClient(paused=True, constant=True)
    summary = run_capture(client, trajectory, tmp_path / "capture")
    assert summary["status"] == "depth_suspect"
    assert summary["frames"][0]["all_perspective_views_suspect"] is True
    assert summary["frames"][0]["depth_cloud"]["status"] == "depth_suspect"
    assert_restored(client, True)


def test_pose_mismatch_aborts_and_restores(trajectory, tmp_path):
    client = ReplayClient(wrong_pose=True)
    with pytest.raises(RuntimeError, match="did not reach recorded pose"):
        run_capture(client, trajectory, tmp_path / "capture")
    assert_restored(client, False)
    report = json.loads((tmp_path / "capture/recapture.json").read_text())
    assert report["last_pose_check"]["attempt"] == 3
    assert report["last_pose_check"]["observed_position"] == [100, 100, 100]


def test_slow_initial_frame_drift_does_not_accumulate(trajectory, tmp_path):
    client = ReplayClient(drift_steps=1)
    summary = run_capture(client, trajectory, tmp_path / "capture")
    assert summary["status"] == "completed"
    assert summary["frames"][0]["pose_error_m"] == 0
    assert_restored(client, False)


def test_pose_mismatch_retries_without_relaxing_tolerance(trajectory, tmp_path):
    client = ReplayClient(drift_steps=2)
    summary = run_capture(client, trajectory, tmp_path / "capture")
    assert summary["status"] == "completed"
    assert summary["last_pose_check"]["attempt"] == 2
    assert client.steps == 4
    assert_restored(client, False)


def test_empty_failed_attempt_can_be_retried_with_history(trajectory, tmp_path):
    output = tmp_path / "capture"
    with pytest.raises(RuntimeError, match="RPC failure"):
        run_capture(ReplayClient(failure=True), trajectory, output)
    summary = run_capture(ReplayClient(), trajectory, output)
    assert summary["status"] == "completed"
    assert summary["previous_attempts"][0]["error"] == "RPC failure"
    before = (output / "000005/FrontCamera/rgb.png").read_bytes()
    with pytest.raises(ValueError, match="contains capture files"):
        run_capture(ReplayClient(), trajectory, output)
    assert (output / "000005/FrontCamera/rgb.png").read_bytes() == before


def test_cloud_export_failure_still_restores_simulator(trajectory, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("cloud export failed")
    monkeypatch.setattr("scripts.recapture_depth.save_depth_cloud", fail)
    client = ReplayClient()
    with pytest.raises(RuntimeError, match="cloud export failed"):
        run_capture(client, trajectory, tmp_path / "capture")
    assert_restored(client, False)
    assert (tmp_path / "capture/000005/FrontCamera/depth_perspective_m.npy").is_file()


def test_frame_selection_normalizes_quaternion_and_subsamples_saved_frames(trajectory):
    records = load_records(trajectory, every=2)
    assert [r["frame"] for r in records] == ["000005", "000015"]
    np.testing.assert_allclose(records[0]["orientation_xyzw"], [0, 0, 2**-.5, 2**-.5])
    assert load_records(trajectory, ["5"])[0]["frame"] == "000005"
    with pytest.raises(ValueError, match="distinct"):
        load_records(trajectory, ["5", "000005"])
    state_path = trajectory / "log/000005.json"
    state_path.write_text(json.dumps({"sensors": {"state": {"position": [1, 2, 3], "orientation": [0, 0, 0, 0]}}}))
    with pytest.raises(ValueError, match="zero quaternion"):
        load_records(trajectory, [5])
