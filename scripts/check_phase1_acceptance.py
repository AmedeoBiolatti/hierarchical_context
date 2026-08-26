#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tool_context.acceptance import assess_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Phase 1 exit gates")
    parser.add_argument("--report", default="artifacts/phase1_attention.json")
    args = parser.parse_args()
    result = assess_phase1(json.loads(Path(args.report).read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
