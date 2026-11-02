"""Replay recorded TravelUAV poses in a running AirSim scene and capture fresh RGB-D."""
import argparse
import json
from pathlib import Path
import sys
import time

import airsim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, write_json
from src.cloud import depth_statistics, save_depth_cloud
from src.openuav.capture import OPENUAV_CAMERAS, OpenUAVFiveViewCapture, save_capture
from src.openuav.client import OpenUAVClient


def load_records(trajectory, frames=None, every=1):
    if every < 1:
        raise ValueError("--every must be positive")
    if frames is None:
        files = sorted((trajectory / "frontcamera").glob("*.png"))
        frames = [p.stem for p in files][::every]
    else:
        frames = [str(int(frame)).zfill(6) for frame in frames]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("Select a nonempty set of distinct recorded frames")
    records = []
    for frame in frames:
        path = trajectory / "log" / f"{frame}.json"
        state = json.loads(path.read_text())["sensors"]["state"]
        position, quaternion = np.asarray(state["position"], float), np.asarray(state["orientation"], float)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all():
            raise ValueError(f"Invalid recorded pose: {path}")
        if np.linalg.norm(quaternion) < 1e-8:
            raise ValueError(f"Invalid zero quaternion: {path}")
        quaternion /= np.linalg.norm(quaternion)
        records.append({"frame": frame, "source_log": str(path.resolve()),
                        "position": position.tolist(), "orientation_xyzw": quaternion.tolist()})
    return records


def advance_paused(client, timeout=15):
    client.simContinueForFrames(1)
    deadline = time.monotonic() + timeout
    while not client.simIsPause():
        if time.monotonic() > deadline:
            raise TimeoutError("Simulator did not pause after advancing one frame")
        time.sleep(.01)


def check_output(output):
    """Retry a failed attempt only when it produced no capture files."""
    if not output.exists():
        return None
    if not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    entries = {p.name for p in output.iterdir()}
    if not entries:
        return None
    if entries == {"recapture.json"}:
        previous = json.loads((output / "recapture.json").read_text())
        if previous.get("status") in ("failed", "interrupted") and previous.get("frames") == []:
            return previous
    raise ValueError(f"Output contains capture files; choose a new output directory: {output}")


def pose_errors(actual, record):
    measured = np.array([actual["position"][key] for key in "xyz"])
    quaternion = np.array([actual["orientation_wxyz"][key] for key in "xyzw"])
    if not np.isfinite(np.r_[measured, quaternion]).all() or np.linalg.norm(quaternion) < 1e-8:
        raise RuntimeError("Simulator returned an invalid rendered vehicle pose")
    quaternion /= np.linalg.norm(quaternion)
    angle = np.degrees(2 * np.arccos(np.clip(abs(np.dot(quaternion, record["orientation_xyzw"])), 0, 1)))
    return float(np.linalg.norm(measured - record["position"])), float(angle), measured.tolist()


def recapture(client, records, output, vehicle_name, capture_settings, trajectory, check_planar=False, warmup_frames=2,
              cloud_settings=None, render_cloud=True):
    previous = check_output(output)
    if warmup_frames < 1:
        raise ValueError("At least one warmup frame is required")
    original_pause = client.simIsPause()
    original_state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    capture = OpenUAVFiveViewCapture(client, vehicle_name=vehicle_name,
                                     raw_color_order=capture_settings["raw_color_order"],
                                     flip_vertical=capture_settings["flip_vertical"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {"source_trajectory": str(trajectory.resolve()), "expected_scene": trajectory.parent.name,
               "scene_verification": "User must launch the same scene version and NED origin as the recorded data",
               "vehicle_name": vehicle_name, "depth_type": "DepthPerspective", "frames": [], "status": "running"}
    if previous:
        summary["previous_attempts"] = previous.pop("previous_attempts", []) + [previous]
    try:
        client.simPause(True)
        # Initialize render targets/shaders while physics is paused, before moving.
        capture.capture()
        for record in records:
            # Match TravelUAV's simSetKinematics replay and remove residual velocities.
            state = airsim.KinematicsState()
            state.position = airsim.Vector3r(*record["position"])
            state.orientation = airsim.Quaternionr(*record["orientation_xyzw"])
            for attempt in range(1, 4):
                for _ in range(warmup_frames):
                    # A cold/slow render frame can leave physics running longer
                    # than expected. Reapply the requested pose before every step.
                    client.simSetKinematics(state, ignore_collision=True, vehicle_name=vehicle_name)
                    advance_paused(client)
                    capture.capture()
                observations = capture.capture()
                actual = capture.frame_metadata["vehicle_pose_world_ned"]
                error_m, angle_deg, measured = pose_errors(actual, record)
                summary["last_pose_check"] = {"frame": record["frame"], "attempt": attempt,
                                              "requested_position": record["position"], "observed_position": measured,
                                              "pose_error_m": error_m, "pose_error_deg": angle_deg}
                if error_m <= .25 and angle_deg <= 2:
                    break
                print(f"{record['frame']}: pose attempt {attempt}/3: {error_m:.3f} m / {angle_deg:.3f} degrees; repositioning", flush=True)
            else:
                raise RuntimeError(f"Vehicle did not reach recorded pose after 3 attempts: {error_m:.3f} m / {angle_deg:.3f} degrees error; "
                                   f"requested={record['position']}, observed={measured}")
            frame_dir = output / record["frame"]
            statistics = {view.name: depth_statistics(view.depth_perspective_m) for view in observations}
            metadata = {**capture.frame_metadata, "recapture": record, "source_trajectory": str(trajectory.resolve()),
                        "expected_scene": trajectory.parent.name, "pose_error_m": error_m,
                        "pose_error_deg": angle_deg, "cameras": {view.name: view.metadata for view in observations},
                        "depth_statistics": {"DepthPerspective": statistics},
                        "rgb_depth_pairing": "Both RGB and depth freshly acquired; do not pair with old dataset RGB"}
            all_suspect = all(info["suspect"] for info in statistics.values())
            if all_suspect:
                metadata["depth_quality"] = "metric_float_all_views_suspect"
            save_capture(observations, frame_dir, metadata)
            (frame_dir / "source_log.json").write_bytes(Path(record["source_log"]).read_bytes())
            if check_planar:
                responses = client.simGetImages([
                    airsim.ImageRequest(name, airsim.ImageType.DepthPlanar, True, False)
                    for name in OPENUAV_CAMERAS], vehicle_name=vehicle_name)
                if len(responses) != len(OPENUAV_CAMERAS):
                    raise RuntimeError("Incomplete DepthPlanar diagnostic responses")
                planar = {}
                for name, response in zip(OPENUAV_CAMERAS, responses):
                    if response.width <= 0 or response.height <= 0:
                        raise RuntimeError(f"Invalid diagnostic DepthPlanar response: {name}")
                    depth = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height).astype(np.float32)
                    if capture_settings["flip_vertical"]:
                        depth = depth[::-1]
                    np.save(frame_dir / name / "depth_planar_m.npy", depth)
                    planar[name] = {**depth_statistics(depth), "timestamp_ns": int(response.time_stamp),
                                    "note": "Diagnostic camera-plane depth; not substituted for DepthPerspective"}
                metadata["depth_statistics"]["DepthPlanar"] = planar
                write_json(frame_dir / "meta.json", metadata)
            cloud = save_depth_cloud(frame_dir, cloud_settings, render_cloud)
            row = {"frame": record["frame"], "capture": str(frame_dir), "pose_error_m": error_m, "pose_error_deg": angle_deg,
                   "all_perspective_views_suspect": all_suspect, "depth_statistics": metadata["depth_statistics"], "depth_cloud": cloud}
            summary["frames"].append(row)
            write_json(output / "recapture.json", summary)
            print(f"{record['frame']}: saved fresh five-view RGB-D; all depth views suspect={all_suspect}", flush=True)
        summary["status"] = "depth_suspect" if any(r["all_perspective_views_suspect"] for r in summary["frames"]) else "completed"
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        # Restore kinematics and pause state even if capture or file writing fails.
        try:
            client.simSetKinematics(original_state, ignore_collision=True, vehicle_name=vehicle_name)
        finally:
            client.simPause(original_pause)
            write_json(output / "recapture.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--trajectory", type=Path, required=True, help="Absolute path or relative to dataset_root")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", nargs="+", help="Recorded frame numbers, e.g. 000210")
    group.add_argument("--all-frames", action="store_true", help="Every saved FrontCamera frame with its matching pose log")
    parser.add_argument("--every", type=int, default=1, help="Subsample saved frames with --all-frames")
    parser.add_argument("--ip")
    parser.add_argument("--port", type=int)
    parser.add_argument("--vehicle-name")
    parser.add_argument("--output", type=Path, help="New/empty directory; a failed attempt with only recapture.json may be retried")
    parser.add_argument("--check-planar", action="store_true", help="Also save raw DepthPlanar for diagnosis")
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--no-render", action="store_true", help="Export standard depth clouds without HTML/PNG previews")
    parser.add_argument("--dry-run", action="store_true", help="Print the replay plan without connecting to AirSim")
    args = parser.parse_args()
    config = load_config(args.config)
    trajectory = args.trajectory if args.trajectory.is_absolute() else Path(config["paths"]["dataset_root"]) / args.trajectory
    records = load_records(trajectory, args.frames, args.every)
    if args.every != 1 and not args.all_frames:
        parser.error("--every only applies to --all-frames")
    if args.warmup_frames < 1:
        parser.error("--warmup-frames must be positive")
    settings = config["capture"]
    connection = {"ip": args.ip or settings["ip"], "port": args.port or settings["port"],
                  "vehicle_name": args.vehicle_name if args.vehicle_name is not None else settings["vehicle_name"]}
    if args.dry_run:
        print(json.dumps({"trajectory": str(trajectory), "connection": connection, "frames": records}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for capture")
    try:
        check_output(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    client = OpenUAVClient(ip=connection["ip"], port=connection["port"], timeout_value=30)
    try:
        client.confirmConnection()
        summary = recapture(client, records, args.output, connection["vehicle_name"], settings, trajectory,
                            args.check_planar, args.warmup_frames, config["cloud"], not args.no_render)
    finally:
        client.client.close()
    return 2 if summary["status"] == "depth_suspect" else 0


if __name__ == "__main__":
    raise SystemExit(main())
