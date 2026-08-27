#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tool_context.prefill import run_phase2_prefill, run_prefill_worker
from tool_context.reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/phase2.json")
    parser.add_argument("--output", default="artifacts/phase2_prefill.json")
    parser.add_argument("--case"); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: config = json.load(handle)
    if args.worker:
        with open(args.case, encoding="utf-8") as handle: case = json.load(handle)
        try: result = run_prefill_worker(config, case)
        except torch.OutOfMemoryError as exc:
            result = {"status": "oom", "mode": case["mode"],
                      "layout": case["shape"]["name"],
                      "sequence_length": case["shape"]["total_tokens"], "error": str(exc)}
        write_json(args.output, result)
    else:
        run_phase2_prefill(config, args.output, __file__)


if __name__ == "__main__":
    import torch
    main()
