#!/usr/bin/env python3
"""Selected OpenUAV RGB-D views -> direction semantics/depth; direct pixels, no point clouds."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import GSAActionPipeline
from src.actions import DEFAULT_WINDOW_DEG


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Dataset sample/trajectory directory or prepared capture")
    parser.add_argument('--stream', action='store_true', help='Persistent JSONL input/output worker; reuse loaded segmentation models')
    parser.add_argument('--cameras', nargs='+', help='Selected standard cameras; default FrontCamera DownCamera')
    parser.add_argument('--directions', type=Path, help='JSON list of name/yaw_deg/elevation_deg queries; overrides config actions.directions')
    parser.add_argument('--gravity-aligned', action=argparse.BooleanOptionalAction, default=True,
                        help='Default: up/down follow gravity and front follows yaw; --no-gravity-aligned uses body axes')
    parser.add_argument("--frame", help="Numeric frame ID; optional for a directory with one complete PNG frame")
    parser.add_argument("--output", type=Path, help="Opt in to intermediate files and action_values.json/csv; default memory only")
    parser.add_argument('--verbose', action='store_true', help='Enable model progress and failure tracebacks; default quiet')
    parser.add_argument('--ram-tags', action=argparse.BooleanOptionalAction, default=None,
                        help='Default: enable RAM when both front/down are selected; --no-ram-tags returns only direction rows')
    parser.add_argument("--config", help="Resource JSON; defaults to SEM3D_CONFIG or config/resources.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument('--init-lock', type=Path, help='Serialize first CUDA/model loading across workers using this lock file')
    parser.add_argument('--min-free-gib', type=float, default=0., help='Free CUDA memory required before model loading')
    parser.add_argument('--memory-wait', type=float, default=300., help='Maximum seconds waiting for startup lock/free memory')
    parser.add_argument("--classes", help="Comma-separated English categories; default from config")
    parser.add_argument("--camera-layout", type=Path, help="Override the standard five-camera calibration")
    parser.add_argument("--window-deg", type=float, default=DEFAULT_WINDOW_DEG,
                        help='Full cone opening angle, default 20 degrees (10-degree half-angle)')
    parser.add_argument("--kde-bandwidth", type=float, default=.4)
    parser.add_argument("--max-detections", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        # Official model loaders print progress; keep stdout valid JSON for callers.
        with redirect_stdout(sys.stderr):
            pipeline = GSAActionPipeline(args.config, device=args.device,
                                         classes=args.classes.split(",") if args.classes is not None else None,
                                         window_deg=args.window_deg, kde_bandwidth=args.kde_bandwidth,
                                         max_detections=args.max_detections, cameras=args.cameras,
                                         gravity_aligned=args.gravity_aligned,init_lock=args.init_lock,
                                         min_free_gib=args.min_free_gib,memory_wait=args.memory_wait,
                                         directions=json.loads(args.directions.read_text()) if args.directions else None)
        ram_tags = args.ram_tags if args.ram_tags is not None else (
            {'FrontCamera', 'DownCamera'} <= set(args.cameras or ('FrontCamera', 'DownCamera')))
        if args.stream:
            print(json.dumps(dict(ready=True, cameras=pipeline.cameras)), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                cold_start = pipeline._model is None
                capture_s = 0.
                with redirect_stdout(sys.stderr):
                    if request.get('live'):
                        from src.openuav.client import OpenUAVClient
                        from src.openuav.capture import OpenUAVFiveViewCapture, save_capture
                        started = time.perf_counter()
                        client = OpenUAVClient(ip=request['live']['ip'], port=request['live']['port'])
                        try:
                            capture = OpenUAVFiveViewCapture(client, cameras=pipeline.cameras,
                                raw_color_order=pipeline.config['capture']['raw_color_order'])
                            images = capture.capture(pause_sim=True)
                            save_capture(images, Path(request['input']), capture.frame_metadata)
                        finally:
                            client.client.close()
                        capture_s = time.perf_counter()-started
                    extra = {'ram_tags': True} if ram_tags else {}
                    rows = pipeline.run(request['input'], frame=request.get('frame'), output=request.get('output'),
                                        camera_layout=args.camera_layout, verbose=args.verbose, **extra)
                if ram_tags:
                    rows, tags = rows['actions'], rows['ram_tags']
                print(json.dumps(dict(id=request.get('id'), rows=rows, report=pipeline.last_report,
                                      capture_wall_s=capture_s, cold_start=cold_start,
                                      **({'ram_tags': tags} if ram_tags else {})),
                                 ensure_ascii=False, allow_nan=False), flush=True)
            return 0
        if args.input is None:
            parser.error('--input is required unless --stream is selected')
        with redirect_stdout(sys.stderr):
            extra = {'ram_tags': True} if ram_tags else {}
            rows = pipeline.run(args.input, frame=args.frame, output=args.output,
                                camera_layout=args.camera_layout, verbose=args.verbose, **extra)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        parser.exit(1, f"Pipeline failed: {error}\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
