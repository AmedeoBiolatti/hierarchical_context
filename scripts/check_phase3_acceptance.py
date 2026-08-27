#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool_context.acceptance import assess_phase3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/phase3.json")
    parser.add_argument("--primary", default="artifacts/phase3_primary_eval.json")
    parser.add_argument("--confirmation", action="append", default=[])
    parser.add_argument("--output", default="artifacts/decisions/phase3.yaml")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); primary = json.loads(Path(args.primary).read_text())
    confirmations = [json.loads(Path(path).read_text()) for path in args.confirmation]
    decision = assess_phase3(primary, confirmations, config["acceptance"])
    lines = ["phase: 3", f"result: {'pass' if decision['passed'] else 'fail'}", "failures:"]
    lines += [f"  - {json.dumps(item)}" for item in decision["failures"]] or ["  []"]
    lines += ["observed:"]
    for key, value in decision["observed"].items():
        if isinstance(value, (int, float, str, bool)):
            lines.append(f"  {key}: {json.dumps(value)}")
    lines += ["decision: continue" if decision["passed"] else "decision: stop",
              "next_change: train predicted routing in Phase 4" if decision["passed"]
              else "next_change: revisit oracle adaptation, memory placement, and dense retention"]
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    if not decision["passed"]:
        raise SystemExit("Phase 3 acceptance failed: " + "; ".join(decision["failures"]))


if __name__ == "__main__":
    main()
