#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tool_context.acceptance import assess_phase0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Phase 0 exit gates")
    parser.add_argument("--report", default="artifacts/phase0_metrics.json")
    args = parser.parse_args()
    with Path(args.report).open(encoding="utf-8") as handle:
        result = assess_phase0(json.load(handle))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

