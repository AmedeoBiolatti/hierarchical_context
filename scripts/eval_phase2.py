#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tool_context.eval.quality import run_phase2_quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/phase2.json")
    parser.add_argument("--output", default="artifacts/phase2_quality.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    run_phase2_quality(config, args.output, limit=args.limit)


if __name__ == "__main__":
    main()
