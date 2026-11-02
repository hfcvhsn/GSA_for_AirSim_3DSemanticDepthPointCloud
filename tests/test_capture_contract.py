import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import airsim
import numpy as np
from PIL import Image

from src.openuav.capture import OpenUAVFiveViewCapture, save_capture


class FakeClient:
    def __init__(self, paused=False, fail=False):
        self.paused, self.fail, self.pause_calls = paused, fail, []

    def simIsPause(self):
        return self.paused

    def simPause(self, state):
        self.paused = state
        self.pause_calls.append(state)

    def simGetVehiclePose(self, **kwargs):
        return airsim.Pose(airsim.Vector3r(2, 3, -4), airsim.Quaternionr())

    def simGetImages(self, requests, **kwargs):
        if self.fail:
            raise RuntimeError("RPC capture failure")
        common = dict(width=2, height=1, camera_position=airsim.Vector3r(),
                      camera_orientation=airsim.Quaternionr(), time_stamp=10)
        # Raw BGR: red, then blue. This catches channel swaps in saved PNGs.
        return [NS(**common, image_data_uint8=bytes([0, 0, 255, 255, 0, 0])),
                NS(**common, image_data_float=[2.5, 4.5])]

    def simGetCameraInfo(self, *args, **kwargs):
        return NS(fov=90., pose=airsim.Pose(), proj_mat=NS(matrix=np.eye(4)))


class CaptureContractTests(unittest.TestCase):
    def test_color_and_metric_depth_roundtrip(self):
        client = FakeClient()
        capture = OpenUAVFiveViewCapture(client, cameras=["FrontCamera"])
        observations = capture.capture(pause_sim=True)
        self.assertEqual(client.pause_calls, [True, False])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_capture(observations, root, capture.frame_metadata)
            np.testing.assert_array_equal(np.array(Image.open(root / "FrontCamera/rgb.png")),
                                          [[[255, 0, 0], [0, 0, 255]]])
            depth = np.load(root / "FrontCamera/depth_perspective_m.npy")
            self.assertEqual(depth.dtype, np.float32)
            np.testing.assert_allclose(depth, [[2.5, 4.5]])
            self.assertEqual(json.loads((root / "meta.json").read_text())["vehicle_pose_world_ned"]["position"]["z"], -4)

    def test_existing_pause_is_preserved(self):
        client = FakeClient(paused=True)
        OpenUAVFiveViewCapture(client, cameras=["FrontCamera"]).capture(pause_sim=True)
        self.assertTrue(client.paused)
        self.assertEqual(client.pause_calls, [])

    def test_pause_restored_on_rpc_failure(self):
        client = FakeClient(fail=True)
        with self.assertRaises(RuntimeError):
            OpenUAVFiveViewCapture(client).capture(pause_sim=True)
        self.assertFalse(client.paused)
        self.assertEqual(client.pause_calls, [True, False])


if __name__ == "__main__":
    unittest.main()
