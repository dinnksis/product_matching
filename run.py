from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.io import run_inference
from src.scorer import HeuristicScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product matching inference")
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument(
        "--output-path", "--output_path", "-o", required=True, type=Path
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()
    scorer = HeuristicScorer.from_json(Path(__file__).parent / "model" / "config.json")
    run_inference(args.items_path, args.matches_path, args.output_path, scorer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

