from tool_context.acceptance import assess_phase0


def family(episodes, recall):
    return {"episodes": episodes, "support_recall": recall}


def test_phase0_acceptance_gate_reports_decisive_metrics():
    hard = {
        "near_duplicate": family(100, 0.6), "two_block_join": family(100, 0.2),
        "three_block_chain": family(50, 0.0), "stale_conflict": family(75, 0.5),
        "aggregation": family(50, 0.0),
    }
    perfect = {name: family(values["episodes"], 1.0) for name, values in hard.items()}
    report = {
        "tokenizer": {"revision": "abc"},
        "metrics": {
            "oracle": {"0.25": {"aggregate": {"support_recall": 1.0}, "families": perfect}},
            "dense": {"0.25": {"aggregate": {"support_recall": 1.0}, "families": perfect}},
            "bm25": {
                "0.10": {"aggregate": {"support_recall": 0.2}},
                "0.25": {"families": hard},
            },
            "minilm": {"0.25": {"families": hard}},
        },
    }
    result = assess_phase0(report)
    assert result["passed"]
    assert result["observed"]["oracle_hard_recall_at_25pct"] == 1.0

