from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from ..routing import BlockSelector, block_token_counts, select_under_budget
from ..schema import EpisodeGraph


def support_recalled(episode: EpisodeGraph, selected: Iterable[str]) -> bool:
    chosen = set(selected)
    return any(support.issubset(chosen) for support in episode.acceptable_support_sets)


def _empty_metrics() -> dict[str, float]:
    return {
        "episodes": 0.0,
        "support_recalled": 0.0,
        "opened_token_fraction_sum": 0.0,
        "selected_blocks_sum": 0.0,
        "no_tool_episodes": 0.0,
        "no_tool_false_positives": 0.0,
    }


def _finalize(raw: dict[str, float]) -> dict[str, float | int]:
    episodes = int(raw["episodes"])
    no_tool = int(raw["no_tool_episodes"])
    return {
        "episodes": episodes,
        "support_recall": raw["support_recalled"] / episodes if episodes else 0.0,
        "mean_opened_token_fraction": raw["opened_token_fraction_sum"] / episodes if episodes else 0.0,
        "mean_selected_blocks": raw["selected_blocks_sum"] / episodes if episodes else 0.0,
        "no_tool_false_positive_rate": raw["no_tool_false_positives"] / no_tool if no_tool else 0.0,
    }


def evaluate_selectors(
    episodes: Sequence[EpisodeGraph], selectors: Sequence[BlockSelector], tokenizer: Any,
    budgets: Sequence[float] = (0.10, 0.25, 0.35, 0.50),
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for selector in selectors:
        budget_results: dict[str, Any] = {}
        for budget in budgets:
            aggregate = _empty_metrics()
            families: dict[str, dict[str, float]] = defaultdict(_empty_metrics)
            for episode in episodes:
                ranking = selector.rank(episode)
                selected = select_under_budget(
                    episode, ranking, budget, tokenizer,
                    ignores_budget=selector.ignores_budget,
                )
                counts = block_token_counts(episode, tokenizer)
                opened = sum(counts[block_id] for block_id in selected)
                available = sum(counts.values())
                family = episode.template_family.split("/")[0]
                for metrics in (aggregate, families[family]):
                    metrics["episodes"] += 1
                    metrics["support_recalled"] += float(support_recalled(episode, selected))
                    metrics["opened_token_fraction_sum"] += opened / available if available else 0.0
                    metrics["selected_blocks_sum"] += len(selected)
                    if not episode.requires_tool:
                        metrics["no_tool_episodes"] += 1
                        metrics["no_tool_false_positives"] += float(bool(selected))
            budget_results[f"{budget:.2f}"] = {
                "aggregate": _finalize(aggregate),
                "families": {name: _finalize(raw) for name, raw in sorted(families.items())},
            }
        results[selector.name] = budget_results
    return results

