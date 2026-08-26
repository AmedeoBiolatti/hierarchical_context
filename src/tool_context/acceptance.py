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
    return {
        "passed": not failures,
        "failures": failures,
        "observed": {
            "bm25_support_recall_at_10pct": bm25_at_ten,
            "oracle_hard_recall_at_25pct": oracle_hard,
            "conventional_hard_recall_at_25pct": conventional_hard,
            "tokenizer_revision": tokenizer_revision,
        },
    }

