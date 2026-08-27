#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tool_context.training.phase3 import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/phase3.json")
    parser.add_argument("--model", choices=("pilot", "primary"), required=True)
    parser.add_argument("--teacher-cache", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int); parser.add_argument("--seed", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: config = json.load(handle)
    run_training(config, args.model, args.teacher_cache, args.output, limit=args.limit,
                 seed=args.seed, max_optimizer_steps=args.max_optimizer_steps, resume=args.resume)


if __name__ == "__main__":
    main()
