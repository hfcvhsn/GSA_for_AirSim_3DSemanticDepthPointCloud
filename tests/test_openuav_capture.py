import math

import numpy as np

from src.openuav.capture import OPENUAV_CAMERAS, intrinsics_from_horizontal_fov


def test_openuav_camera_order_matches_traveluav():
    assert OPENUAV_CAMERAS == (
        "FrontCamera",
        "LeftCamera",
        "RightCamera",
        "RearCamera",
        "DownCamera",
    )


def test_intrinsics_for_openuav_default_camera():
    K = intrinsics_from_horizontal_fov(256, 256, 90.0)
    expected = np.array(
        [
            [128.0, 0.0, 128.0],
            [0.0, 128.0, 128.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(K, expected, rtol=0.0, atol=1e-12)


def test_intrinsics_reject_invalid_fov():
    for fov in (0.0, 180.0, -1.0):
        try:
            intrinsics_from_horizontal_fov(256, 256, fov)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for FOV={fov}")
