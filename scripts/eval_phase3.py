#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tool_context.eval.phase3 import run_phase3_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/phase3.json")
    parser.add_argument("--model", choices=("pilot", "primary"), required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: config = json.load(handle)
    run_phase3_evaluation(config, args.model, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
