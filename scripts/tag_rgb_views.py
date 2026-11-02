#!/usr/bin/env python3
"""Output official RAM tags for front/down RGB; no depth or GSA inference required."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import write_json
from src.ram_tags import RAMViewTagger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front-rgb', type=Path, required=True)
    parser.add_argument('--down-rgb', type=Path, required=True)
    parser.add_argument('--config', help='SEM3D_CONFIG or config/resources.json by default')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, help='Optional JSON file; stdout always contains the tags')
    parser.add_argument('--verbose', action='store_true', help='Enable official model progress on stderr')
    args = parser.parse_args(argv)
    try:
        tagger = RAMViewTagger(args.config, device=args.device)
        tags = tagger.tag(args.front_rgb, args.down_rgb, verbose=args.verbose)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, tags)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'RAM tagging failed: {error}\n')
    print(json.dumps(tags, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
