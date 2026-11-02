"""Step 2: registered RGB-D frame -> per-view semantic IDs through official GSA."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config
from src.gsa import GroundedMobileSAM, segment_capture, validate_capture, validate_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 after CUDA_VISIBLE_DEVICES, or cpu")
    parser.add_argument("--classes", help="Comma-separated English category names, shared across all views")
    parser.add_argument("--max-detections", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output is not empty: {args.output}")
    if not 1 <= args.max_detections <= 65535:
        raise SystemExit("--max-detections must be in [1, 65535]")
    config = load_config(args.config)
    classes = validate_classes(args.classes.split(",") if args.classes else config["segmentation"]["classes"])
    for name, value in (("HF_HOME", ROOT / ".cache/huggingface"), ("TORCH_HOME", ROOT / ".cache/torch"),
                        ("MPLCONFIGDIR", ROOT / ".cache/matplotlib"), ("HF_HUB_OFFLINE", "1"),
                        ("TOKENIZERS_PARALLELISM", "false")):
        os.environ.setdefault(name, str(value))
    validate_capture(args.input)
    pipeline = GroundedMobileSAM(config, classes, args.device)
    report = segment_capture(args.input, args.output, pipeline, config, args.max_detections)
    for camera, count in report["detections"].items():
        print(f"{camera}: {count} detections", flush=True)
    print(f"Saved masks: {args.output}")


if __name__ == "__main__":
    main()
