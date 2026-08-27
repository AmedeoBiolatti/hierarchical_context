#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tool_context.training.phase3 import build_teacher_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/phase3.json")
    parser.add_argument("--model", choices=("pilot", "primary"), required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: config = json.load(handle)
    build_teacher_cache(config, args.model, args.output, limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
