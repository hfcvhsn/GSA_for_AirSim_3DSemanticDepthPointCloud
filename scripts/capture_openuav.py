from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running directly from the repository root without installing a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.openuav.capture import OPENUAV_CAMERAS, OpenUAVFiveViewCapture, save_capture
from src.config import load_config
from src.cloud import save_depth_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture or import any nonempty subset of the five OpenUAV RGB-D views."
    )
    parser.add_argument("--config")
    parser.add_argument("--ip", help="AirSim server IP")
    parser.add_argument("--port", type=int, help="AirSim RPC port")
    parser.add_argument("--vehicle-name", help="AirSim vehicle name")
    parser.add_argument("--from-dataset", action="store_true", help="Import a saved TravelUAV frame")
    parser.add_argument("--trajectory", type=Path, help="Absolute path, or relative to dataset_root")
    parser.add_argument("--frame", help="Dataset frame number, e.g. 000005")
    parser.add_argument("--camera-layout", type=Path, help="Override dataset camera extrinsics JSON")
    parser.add_argument("--cameras", nargs='+', choices=OPENUAV_CAMERAS, default=OPENUAV_CAMERAS,
                        help="Selected views; default all five. Only selected dataset folders are required.")
    parser.add_argument("--allow-quantized-depth", action="store_true", help="Accept approximate 8-bit dataset depth")
    parser.add_argument("--raw-color-order", choices=["BGR", "RGB"])
    parser.add_argument("--flip-vertical", action="store_true", default=None)
    parser.add_argument("--no-render", action="store_true", help="Export the standard depth cloud without HTML/PNG previews")
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
    args = parser.parse_args()
    if len(set(args.cameras)) != len(args.cameras):
        parser.error('Choose distinct cameras')
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}; choose a new frame directory")
    if args.from_dataset:
        if config["dataset"]["depth_encoding"] != "uint8_0_100m":
            raise SystemExit("Dataset importer currently supports only uint8_0_100m depth PNGs")
        if not args.allow_quantized_depth:
            raise SystemExit("Dataset PNG depth is quantized, not original metric depth; pass --allow-quantized-depth for smoke runs")
        from src.openuav.dataset import load_dataset_frame
        trajectory = args.trajectory or Path(config["dataset"]["trajectory"])
        if not trajectory.is_absolute():
            trajectory = Path(config["paths"]["dataset_root"]) / trajectory
        observations, metadata = load_dataset_frame(
            trajectory, args.frame or config["dataset"]["frame"],
            config["dataset"]["fov_deg"], args.camera_layout, allow_missing_pose=True, cameras=args.cameras)
        save_capture(observations, args.output, metadata)
        save_depth_cloud(args.output, config["cloud"], not args.no_render)
        print(f"Imported {len(observations)} views with APPROXIMATE quantized depth: {args.output}")
        return
    from src.openuav.client import OpenUAVClient
    settings = config["capture"]
    client = OpenUAVClient(ip=args.ip or settings["ip"], port=args.port or settings["port"], timeout_value=30)
    client.confirmConnection()

    capture = OpenUAVFiveViewCapture(
        client=client,
        vehicle_name=args.vehicle_name if args.vehicle_name is not None else settings["vehicle_name"],
        cameras=args.cameras,
        raw_color_order=args.raw_color_order or settings["raw_color_order"],
        flip_vertical=args.flip_vertical if args.flip_vertical is not None else settings["flip_vertical"],
    )
    observations = capture.capture(pause_sim=args.pause)
    save_capture(observations, args.output, capture.frame_metadata)
    save_depth_cloud(args.output, config["cloud"], not args.no_render)
    if not capture.frame_metadata["pose_synchronized_by_pause"]:
        print("Vehicle pose was sampled separately; use --pause for synchronized body-frame geometry.")

    print(f"Saved {len(observations)} OpenUAV camera views to {args.output}")


if __name__ == "__main__":
    main()
