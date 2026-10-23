from __future__ import annotations

import argparse
from pathlib import Path
import sys

import airsim

# Allow running directly from the repository root without installing a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.openuav.capture import OPENUAV_CAMERAS, OpenUAVFiveViewCapture, save_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one OpenUAV-compatible five-view RGB-D observation from AirSim."
    )
    parser.add_argument("--ip", default="127.0.0.1", help="AirSim server IP")
    parser.add_argument("--port", type=int, default=41451, help="AirSim RPC port")
    parser.add_argument("--vehicle-name", default="", help="AirSim vehicle name")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the captured frame",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help=(
            "Temporarily pause physics during capture. Disabled by default to "
            "match TravelUAV/OpenUAV getImageResponses(), which does not alter "
            "the caller's pause state."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()

    capture = OpenUAVFiveViewCapture(
        client=client,
        vehicle_name=args.vehicle_name,
        cameras=OPENUAV_CAMERAS,
    )
    observations = capture.capture(pause_sim=args.pause)
    save_capture(observations, args.output)

    print(f"Saved {len(observations)} OpenUAV camera views to {args.output}")


if __name__ == "__main__":
    main()
