"""Read every five-view depth PNG in selected TravelUAV scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, write_json
from src.openuav.capture import OPENUAV_CAMERAS


def audit_trajectory(trajectory, check_tensors=False):
    result = {"scene": trajectory.parent.name, "trajectory": trajectory.name, "views": {}, "errors": []}
    total_histogram = np.zeros(256, dtype=np.int64)
    for camera in OPENUAV_CAMERAS:
        view = camera.lower()
        rgb_dir, depth_dir = trajectory / view, trajectory / f"{view}_depth"
        rgb = {p.name for p in rgb_dir.glob("*.png")}
        depth = {p.name for p in depth_dir.glob("*.png")}
        histogram = np.zeros(256, dtype=np.int64)
        info = {"rgb_files": len(rgb), "depth_files": len(depth), "decoded": 0,
                "missing_directory": not depth_dir.is_dir(), "missing_for_rgb": sorted(rgb - depth),
                "extra_depth_frames": sorted(depth - rgb), "constant_images": 0,
                "constant_2_images": 0, "max_le_2_images": 0, "nontrivial_candidates": []}
        for name in sorted(depth):
            try:
                with Image.open(depth_dir / name) as image:
                    if image.mode != "L":
                        raise ValueError(f"expected single-channel uint8 PNG, got {image.mode}")
                    counts = np.array(image.histogram(), dtype=np.int64)
                    nonzero = np.flatnonzero(counts)
                assert len(counts) == 256 and len(nonzero)
                histogram += counts
                info["decoded"] += 1
                info["constant_images"] += int(len(nonzero) == 1)
                info["constant_2_images"] += int(len(nonzero) == 1 and nonzero[0] == 2)
                info["max_le_2_images"] += int(nonzero[-1] <= 2)
                if np.any((nonzero > 2) & (nonzero < 255)) and len(info["nontrivial_candidates"]) < 5:
                    info["nontrivial_candidates"].append(name)
            except Exception as error:
                result["errors"].append({"file": f"{view}_depth/{name}", "error": str(error)})
        info["histogram"] = {str(i): int(histogram[i]) for i in np.flatnonzero(histogram)}
        result["views"][camera] = info
        total_histogram += histogram
    result["histogram"] = {str(i): int(total_histogram[i]) for i in np.flatnonzero(total_histogram)}
    tensor_path = trajectory / "depth_imgs.tensor"
    result["tensor"] = {"exists": tensor_path.is_file()}
    if check_tensors and tensor_path.is_file():
        try:
            import torch
            tensor = torch.load(tensor_path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"expected Tensor, got {type(tensor).__name__}")
            finite_mask = torch.isfinite(tensor)
            finite = bool(finite_mask.all())
            values = tensor if finite else tensor[finite_mask]
            result["tensor"].update(bytes=tensor_path.stat().st_size, shape=list(tensor.shape),
                                     dtype=str(tensor.dtype), min=float(values.min()) if values.numel() else None,
                                     max=float(values.max()) if values.numel() else None, finite=finite)
        except Exception as error:
            result["tensor"]["error"] = str(error)
            result["errors"].append({"file": "depth_imgs.tensor", "error": str(error)})
    result["other_depth_files"] = [p.name for p in trajectory.iterdir()
                                    if p.is_file() and "depth" in p.name.lower() and p.name != "depth_imgs.tensor"]
    return result


def summarize(results):
    scenes = {}
    for row in results:
        scene = scenes.setdefault(row["scene"], {
            "trajectories": 0, "png_files": 0, "decoded": 0, "missing_directories": 0,
            "missing_for_rgb": 0, "extra_depth_frames": 0, "constant_images": 0,
            "constant_2_images": 0, "max_le_2_images": 0, "nontrivial_candidate_images_examples": [],
            "histogram": {}, "errors": 0, "tensors_present": 0, "tensors_checked": 0,
            "tensor_min": None, "tensor_max": None, "nonfinite_tensors": 0,
            "tensor_read_errors": 0, "other_depth_files": 0})
        scene["trajectories"] += 1
        scene["errors"] += len(row["errors"])
        scene["other_depth_files"] += len(row["other_depth_files"])
        for camera, view in row["views"].items():
            scene["png_files"] += view["depth_files"]
            for key in ("decoded", "constant_images", "constant_2_images", "max_le_2_images"):
                scene[key] += view[key]
            scene["missing_directories"] += int(view["missing_directory"])
            scene["missing_for_rgb"] += len(view["missing_for_rgb"])
            scene["extra_depth_frames"] += len(view["extra_depth_frames"])
            for name in view["nontrivial_candidates"]:
                if len(scene["nontrivial_candidate_images_examples"]) < 10:
                    scene["nontrivial_candidate_images_examples"].append(f"{row['trajectory']}/{camera.lower()}_depth/{name}")
        for value, count in row["histogram"].items():
            scene["histogram"][value] = scene["histogram"].get(value, 0) + count
        tensor = row["tensor"]
        scene["tensors_present"] += int(tensor["exists"])
        scene["tensor_read_errors"] += int("error" in tensor)
        if "min" in tensor:
            scene["tensors_checked"] += 1
            if tensor["min"] is not None:
                scene["tensor_min"] = tensor["min"] if scene["tensor_min"] is None else min(scene["tensor_min"], tensor["min"])
                scene["tensor_max"] = tensor["max"] if scene["tensor_max"] is None else max(scene["tensor_max"], tensor["max"])
            scene["nonfinite_tensors"] += int(not tensor["finite"])
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--scenes", nargs="+", default=["BrushifyCountryRoads", "NordicHarbour"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-tensors", action="store_true", help="Also read every preprocessed depth tensor")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty or absent")
    base = args.dataset_root or Path(load_config(args.config)["paths"]["dataset_root"])
    trajectories = []
    for name in args.scenes:
        scene = base / name
        if not scene.is_dir():
            parser.error(f"Scene directory is missing: {scene}")
        trajectories.extend(sorted(p for p in scene.iterdir() if p.is_dir()))
    if args.check_tensors:
        import torch
        torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output / "trajectories.jsonl").open("w") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_trajectory, p, args.check_tensors) for p in trajectories]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if len(results) % 20 == 0 or len(results) == len(trajectories):
                print(f"Checked {len(results)}/{len(trajectories)} trajectories", flush=True)
    results.sort(key=lambda r: (r["scene"], r["trajectory"]))
    summary = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "dataset_root": str(base),
               "scope": "Every PNG in the five standard depth directories of every immediate trajectory; optional full depth_imgs.tensor checks",
               "check_tensors": args.check_tensors, "scenes": summarize(results)}
    write_json(args.output / "summary.json", summary)
    with (args.output / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "trajectory", "png_files", "decoded", "constant_2", "max_le_2", "pixel_values", "tensor_min", "tensor_max", "errors"])
        for row in results:
            writer.writerow([row["scene"], row["trajectory"], sum(v["depth_files"] for v in row["views"].values()),
                             sum(v["decoded"] for v in row["views"].values()), sum(v["constant_2_images"] for v in row["views"].values()),
                             sum(v["max_le_2_images"] for v in row["views"].values()), ",".join(sorted(row["histogram"], key=int)),
                             row["tensor"].get("min"), row["tensor"].get("max"), len(row["errors"])])
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(r["errors"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
