from __future__ import annotations

import math
from typing import Any, Mapping


HARD_FAMILIES = (
    "near_duplicate", "two_block_join", "three_block_chain", "stale_conflict", "aggregation",
)


def _weighted_hard_recall(metrics: Mapping[str, Any], selector: str, budget: str) -> float:
    families = metrics[selector][budget]["families"]
    total = sum(int(families[name]["episodes"]) for name in HARD_FAMILIES)
    recalled = sum(
        int(families[name]["episodes"]) * float(families[name]["support_recall"])
        for name in HARD_FAMILIES
    )
    return recalled / total


def assess_phase0(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    failures: list[str] = []
    for selector in ("oracle", "dense"):
        recall = float(metrics[selector]["0.25"]["aggregate"]["support_recall"])
        if recall != 1.0:
            failures.append(f"{selector} support recall at 25% is {recall}, expected 1.0")
    bm25_at_ten = float(metrics["bm25"]["0.10"]["aggregate"]["support_recall"])
    if bm25_at_ten >= 0.95:
        failures.append(f"BM25 is too easy at 10%: support recall is {bm25_at_ten}")
    oracle_hard = _weighted_hard_recall(metrics, "oracle", "0.25")
    conventional_hard = {
        selector: _weighted_hard_recall(metrics, selector, "0.25")
        for selector in ("bm25", "minilm")
    }
    for selector, recall in conventional_hard.items():
        if oracle_hard - recall < 0.10:
            failures.append(
                f"oracle gap over {selector} on hard families is {oracle_hard - recall:.3f}, expected >= 0.10"
            )
    tokenizer_revision = report.get("tokenizer", {}).get("revision")
    if not tokenizer_revision:
        failures.append("tokenizer revision was not resolved")
    leakage_recall: dict[str, float] = {}
    for selector in ("metadata_probe", "surface_probe"):
        try:
            recall = float(metrics[selector]["0.25"]["splits"]["held_out"]["support_recall"])
        except KeyError:
            failures.append(f"missing {selector} metrics")
            continue
        leakage_recall[selector] = recall
        if recall >= 0.50:
            failures.append(f"{selector} held-out leakage recall is {recall}, expected < 0.50")
    return {
        "passed": not failures,
        "failures": failures,
        "observed": {
            "bm25_support_recall_at_10pct": bm25_at_ten,
            "oracle_hard_recall_at_25pct": oracle_hard,
            "conventional_hard_recall_at_25pct": conventional_hard,
            "tokenizer_revision": tokenizer_revision,
            "leakage_probe_support_recall_at_25pct": leakage_recall,
        },
    }


def assess_phase1(report: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not report.get("correctness", {}).get("passed"):
        failures.append("Phase 1 correctness tests did not pass")
    rows = report.get("benchmarks", [])

    def median(backend: str, length: int) -> float | None:
        for row in rows:
            if row.get("backend") == backend and row.get("sequence_length") == length and row.get("status") == "ok":
                return float(row["timing"]["median_ms"])
        return None

    speedups: dict[str, float | None] = {}
    for length, threshold in ((16384, 1.25), (32768, 1.5)):
        dense = median("dense_causal", length); flex = median("flex_selected_0.25", length)
        speedup = dense / flex if dense is not None and flex else None
        speedups[str(length)] = speedup
    speed_pass = any(
        speedups[str(length)] is not None and speedups[str(length)] >= threshold
        for length, threshold in ((16384, 1.25), (32768, 1.5))
    )
    dense_oom_lengths = {
        int(row["sequence_length"]) for row in rows
        if row.get("backend") == "dense_causal" and row.get("status") == "oom"
    }
    flex_lengths = {
        int(row["sequence_length"]) for row in rows
        if row.get("backend") == "flex_selected_0.25" and row.get("status") == "ok"
    }
    memory_pass = bool(dense_oom_lengths & flex_lengths)
    if not speed_pass and not memory_pass:
        failures.append("no 16K/32K speed crossover or same-shape memory feasibility win")
    return {
        "passed": not failures, "failures": failures,
        "observed": {"speedups": speedups, "memory_feasibility_lengths": sorted(dense_oom_lengths & flex_lengths)},
    }


def assess_phase2(quality: Mapping[str, Any], prefill: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    correctness = quality.get("correctness", {})
    exact_checks = {
        "dense input_ids/inputs_embeds identity": correctness.get("dense_input_ids_vs_inputs_embeds_max_abs_error") == 0.0,
        "cross-block isolation": float(correctness.get("cross_block_max_abs_error", float("inf"))) <= 1e-6,
        "one BlockMask reused by every layer": correctness.get("same_block_mask_object_all_layers") is True,
        "memory/router-only embedding overwrite": correctness.get("token_bank_only_overwrites_memory_router") is True,
        "finite padding output": correctness.get("padding_logits_finite") is True,
    }
    comparison = correctness.get("causal_flex_vs_sdpa", {})
    tolerance_checks = {
        "causal Flex NLL": float(comparison.get("nll_difference", float("inf"))) <= 0.02,
        "causal Flex logit cosine": float(comparison.get("logit_cosine", 0.0)) >= 0.999,
        "causal Flex top-1 agreement": float(comparison.get("top1_agreement", 0.0)) >= 0.98,
    }
    for name, passed in {**exact_checks, **tolerance_checks}.items():
        if not passed:
            failures.append(f"correctness check failed: {name}")

    rows = quality.get("rows", [])
    expected_examples = 63
    expected_rows = expected_examples * 12
    if quality.get("corpus", {}).get("example_count") != expected_examples:
        failures.append("quality corpus is not the planned stratified 63-example set")
    if len(rows) != expected_rows:
        failures.append(f"quality report has {len(rows)} rows, expected {expected_rows}")
    expected_specs = {
        ("dense_compact", None), ("dense_aligned", None), ("block_static", None),
        *((mode, seed) for mode in ("block_memory", "block_oracle", "block_dynamic_anchor")
          for seed in (1729, 1730, 1731)),
    }
    observed_specs = {(row.get("mode"), row.get("seed")) for row in rows}
    if observed_specs != expected_specs:
        failures.append("quality mode/seed matrix is incomplete")
    families = {row.get("family") for row in rows}; placements = {row.get("placement") for row in rows}
    if len(families) != 7 or placements != {"beginning", "middle", "end"}:
        failures.append("quality family/placement strata are incomplete")
    numeric_fields = ("nll_sum", "nll_per_token", "token_accuracy")
    if any(not all(math.isfinite(float(row.get(field, float("nan")))) for field in numeric_fields) for row in rows):
        failures.append("quality report contains non-finite metrics")

    benchmark_rows = prefill.get("benchmarks", [])
    expected_benchmarks = {(length, mode) for length in (4096, 8192, 16384)
                           for mode in ("dense_aligned", "block_static", "block_memory", "block_oracle")}
    observed_benchmarks = {(row.get("sequence_length"), row.get("mode")) for row in benchmark_rows}
    if observed_benchmarks != expected_benchmarks:
        failures.append("prefill length/mode matrix is incomplete")
    if prefill.get("fresh_process_per_case") is not True:
        failures.append("prefill cases were not isolated in fresh processes")
    for row in benchmark_rows:
        if row.get("status") not in ({"ok"} if row.get("sequence_length") < 16384 else {"ok", "oom"}):
            failures.append(f"prefill case failed: {row.get('layout')}/{row.get('mode')}")
        if row.get("status") == "ok" and not all(
            field in row for field in ("model_load_ms", "mask_and_input_build_ms", "first_call_wall_ms",
                                       "median_ms", "p10_ms", "p90_ms", "peak_allocated_mib", "peak_reserved_mib")
        ):
            failures.append(f"prefill case lacks timing/memory fields: {row.get('layout')}/{row.get('mode')}")
    return {
        "passed": not failures, "failures": failures,
        "observed": {"correctness": {**exact_checks, **tolerance_checks},
                     "quality_rows": len(rows), "prefill_rows": len(benchmark_rows)},
    }


def assess_phase3(primary: Mapping[str, Any], confirmations: list[Mapping[str, Any]],
                  thresholds: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    frozen = primary["frozen_dense"]["aggregate"]
    initial = primary["initial_oracle"]["aggregate"]
    dense = primary["adapted_dense"]["aggregate"]
    oracle = primary["adapted_oracle"]["aggregate"]
    observed = {
        "oracle_nll_gap": oracle["nll_per_token"] - dense["nll_per_token"],
        "oracle_token_accuracy_gap": dense["token_accuracy"] - oracle["token_accuracy"],
        "dense_nll_regression": dense["nll_per_token"] - frozen["nll_per_token"],
        "dense_token_accuracy_regression": frozen["token_accuracy"] - dense["token_accuracy"],
    }
    checks = {
        "oracle NLL gap": observed["oracle_nll_gap"] <= thresholds["oracle_nll_gap_max"],
        "oracle token-accuracy gap": observed["oracle_token_accuracy_gap"] <= thresholds["oracle_token_accuracy_gap_max"],
        "dense NLL retention": observed["dense_nll_regression"] <= thresholds["dense_nll_regression_max"],
        "dense token-accuracy retention": observed["dense_token_accuracy_regression"] <= thresholds["dense_token_accuracy_regression_max"],
    }
    multi_rows = [row for row in primary["adapted_dense"]["rows"] if row["family"] in {
        "two_block_join", "three_block_chain", "aggregation", "code_call_chain",
        "code_stack_trace", "code_injected_bug",
    }]
    oracle_by_id = {row["episode_id"]: row for row in primary["adapted_oracle"]["rows"]}
    dense_tokens = sum(row["target_tokens"] for row in multi_rows)
    dense_acc = sum(row["token_accuracy"] * row["target_tokens"] for row in multi_rows) / dense_tokens
    oracle_acc = sum(oracle_by_id[row["episode_id"]]["token_accuracy"] * row["target_tokens"] for row in multi_rows) / dense_tokens
    observed["multihop_accuracy_gap"] = dense_acc - oracle_acc
    checks["multi-hop aggregate"] = observed["multihop_accuracy_gap"] <= thresholds["multihop_accuracy_gap_max"]
    family_gaps = {}
    for family, values in primary["adapted_dense"]["families"].items():
        if family in {row["family"] for row in multi_rows}:
            family_gaps[family] = values["token_accuracy"] - primary["adapted_oracle"]["families"][family]["token_accuracy"]
    observed["multihop_family_gaps"] = family_gaps
    checks["multi-hop families"] = all(
        value <= thresholds["multihop_family_gap_max"] for value in family_gaps.values()
    )
    correctness = primary.get("correctness", {})
    checks.update({
        "post-training cross-block isolation": float(correctness.get("cross_block_max_abs_error", math.inf)) <= 1e-6,
        "post-training finite padding": correctness.get("padding_logits_finite") is True,
        "post-training mask reuse": correctness.get("same_block_mask_object_all_layers") is True,
    })
    confirmation_results = []
    if len(confirmations) != 2:
        failures.append(f"expected two confirmation reports, found {len(confirmations)}")
    for report in confirmations:
        base_gap = report["initial_oracle"]["aggregate"]["nll_per_token"] - report["frozen_dense"]["aggregate"]["nll_per_token"]
        final_gap = report["adapted_oracle"]["aggregate"]["nll_per_token"] - report["adapted_dense"]["aggregate"]["nll_per_token"]
        reduction = (base_gap - final_gap) / abs(base_gap) if abs(base_gap) > 1e-9 else float(final_gap <= 0.05)
        regression = report["adapted_dense"]["aggregate"]["nll_per_token"] - report["frozen_dense"]["aggregate"]["nll_per_token"]
        passed = reduction >= thresholds["confirmation_gap_reduction_min"] and regression <= thresholds["confirmation_dense_nll_regression_max"]
        confirmation_results.append({"gap_reduction": reduction, "dense_nll_regression": regression, "passed": passed})
        if not passed:
            failures.append("a confirmation run failed its directional/retention gate")
    for name, passed in checks.items():
        if not passed:
            failures.append(f"Phase 3 check failed: {name}")
    observed["checks"] = checks; observed["confirmations"] = confirmation_results
    return {"passed": not failures, "failures": failures, "observed": observed}
