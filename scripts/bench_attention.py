#!/usr/bin/env python3
import argparse
import json

from tool_context.benchmarking import load_benchmark_config, run_phase1_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 1 sparse attention backends")
    parser.add_argument("--config", default="configs/experiment/phase1.json")
    parser.add_argument("--output", default="artifacts/phase1_attention.json")
    args = parser.parse_args()
    result = run_phase1_benchmarks(load_benchmark_config(args.config), args.output)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
