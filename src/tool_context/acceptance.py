from __future__ import annotations

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
