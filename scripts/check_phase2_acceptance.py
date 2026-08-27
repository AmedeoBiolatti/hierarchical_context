#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool_context.acceptance import assess_phase2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", default="artifacts/phase2_quality.json")
    parser.add_argument("--prefill", default="artifacts/phase2_prefill.json")
    parser.add_argument("--output", default="artifacts/decisions/phase2.yaml")
    args = parser.parse_args()
    quality = json.loads(Path(args.quality).read_text()); prefill = json.loads(Path(args.prefill).read_text())
    decision = assess_phase2(quality, prefill)
    def quality_nll(mode: str) -> float:
        rows = [row for row in quality["rows"] if row["mode"] == mode]
        tokens = sum(row["target_tokens"] for row in rows)
        return sum(row["nll_sum"] for row in rows) / tokens

    def benchmark(length: int, mode: str) -> dict:
        return next(row for row in prefill["benchmarks"]
                    if row["sequence_length"] == length and row["mode"] == mode)

    correctness = quality["correctness"]; comparison = correctness["causal_flex_vs_sdpa"]
    lines = ["phase: 2",
             "hypothesis: a pretrained Qwen3 decoder can execute the independent-block topology correctly and expose its unadapted quality shift",
             f"result: {'pass' if decision['passed'] else 'fail'}", "failures:"]
    lines += [f"  - {json.dumps(item)}" for item in decision["failures"]] or ["  []"]
    lines += [
        "evidence:", f"  model: {quality['model']['id']}", f"  revision: {quality['model']['revision']}",
        f"  stratified_examples: {quality['corpus']['example_count']}",
        f"  quality_rows: {decision['observed']['quality_rows']}",
        f"  prefill_rows: {decision['observed']['prefill_rows']}", "  correctness:",
        f"    dense_embedding_max_abs_error: {correctness['dense_input_ids_vs_inputs_embeds_max_abs_error']}",
        f"    cross_block_max_abs_error: {correctness['cross_block_max_abs_error']}",
        f"    causal_flex_nll_difference: {comparison['nll_difference']}",
        f"    causal_flex_logit_cosine: {comparison['logit_cosine']}",
        f"    causal_flex_top1_agreement: {comparison['top1_agreement']}",
        f"    same_mask_all_layers: {str(correctness['same_block_mask_object_all_layers']).lower()}",
        f"    padding_finite: {str(correctness['padding_logits_finite']).lower()}",
        "  unadapted_nll_per_token:",
        f"    dense_compact: {quality_nll('dense_compact')}",
        f"    dense_aligned: {quality_nll('dense_aligned')}",
        f"    block_static: {quality_nll('block_static')}",
        f"    block_memory_seed_average: {quality_nll('block_memory')}",
        f"    block_oracle_seed_average: {quality_nll('block_oracle')}",
        f"    block_dynamic_anchor_seed_average: {quality_nll('block_dynamic_anchor')}",
        "  warm_prefill_speedup_block_static_vs_dense:",
    ]
    for length in (4096, 8192, 16384):
        dense = benchmark(length, "dense_aligned"); block = benchmark(length, "block_static")
        lines.append(f"    {length}_tokens: {dense['median_ms'] / block['median_ms']}")
    dense_16k = benchmark(16384, "dense_aligned"); block_16k = benchmark(16384, "block_static")
    lines += [
        "  peak_allocated_mib_at_16k:",
        f"    dense_aligned: {dense_16k['peak_allocated_mib']}",
        f"    block_static: {block_16k['peak_allocated_mib']}",
        "decision: continue",
        "next_change: Phase 3 oracle-routed adaptation on Qwen3-1.7B; preserve dense retention and train memory tokens",
    ]
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    if not decision["passed"]:
        raise SystemExit("Phase 2 acceptance failed: " + "; ".join(decision["failures"]))


if __name__ == "__main__":
    main()
