"""Five-view depth -> drone-centered cloud; optional masks add semantics."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config
from src.cloud import export, fuse, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--input", type=Path, required=True, help="Step 1 capture")
    parser.add_argument("--masks", type=Path, help="Optional step 2 output; omit for an RGB depth cloud")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--voxel-size", type=float, help="Meters; 0 disables voxel thinning")
    parser.add_argument("--labeled-only", action="store_true")
    parser.add_argument("--allow-unsynchronized", action="store_true", help="Accept separately sampled live vehicle pose")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output is not empty: {args.output}")
    capture_bytes = (args.input / "meta.json").read_bytes()
    capture_meta = json.loads(capture_bytes)
    if args.masks is not None:
        seg = json.loads((args.masks / "meta.json").read_text())
        if seg["capture_meta_sha256"] != hashlib.sha256(capture_bytes).hexdigest():
            raise SystemExit("Masks belong to a different capture (metadata SHA256 mismatch)")
    elif args.labeled_only:
        parser.error("--labeled-only requires --masks")
    if capture_meta.get("source") == "airsim_rpc" and not capture_meta.get("pose_synchronized_by_pause") and not args.allow_unsynchronized:
        raise SystemExit("Vehicle pose is not synchronized: capture with --pause, or explicitly pass --allow-unsynchronized")
    config = load_config(args.config)
    settings = dict(config["cloud"])
    for key in ("stride", "min_depth", "max_depth", "voxel_size"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    settings["labeled_only"] = args.labeled_only
    data, meta, seg, views, counts = fuse(args.input, args.masks, settings, args.labeled_only)
    if not len(data["xyz"]) and args.masks is not None:
        raise SystemExit("No valid points: inspect depth limits, masks and --labeled-only")
    report = export(args.output, data, meta, seg, views, counts, settings)
    if not args.no_render:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        render(args.output, data, seg, settings["render_max_points"], meta["cloud_diagnostics"])
    print(f"Saved {len(data['xyz']):,} points in drone body FRD: {args.output}")
    print(f"Cloud status: {report['status']}")
    if not args.no_render:
        print("Open cloud.html in a browser for interactive rendering; cloud.png is the static preview.")


if __name__ == "__main__":
    main()
