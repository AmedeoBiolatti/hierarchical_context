from tool_context.acceptance import assess_phase0, assess_phase1


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
            "metadata_probe": {"0.25": {"splits": {"held_out": {"support_recall": 0.2}}}},
            "surface_probe": {"0.25": {"splits": {"held_out": {"support_recall": 0.3}}}},
        },
    }
    result = assess_phase0(report)
    assert result["passed"]
    assert result["observed"]["oracle_hard_recall_at_25pct"] == 1.0


def test_phase1_acceptance_uses_25_percent_selected_speedup():
    report = {
        "correctness": {"passed": True},
        "benchmarks": [
            {"backend": "dense_causal", "sequence_length": 16384, "status": "ok", "timing": {"median_ms": 10.0}},
            {"backend": "flex_selected_0.25", "sequence_length": 16384, "status": "ok", "timing": {"median_ms": 7.5}},
        ],
    }
    result = assess_phase1(report)
    assert result["passed"]
    assert result["observed"]["speedups"]["16384"] == 10 / 7.5


def test_phase1_acceptance_rejects_missing_correctness_and_crossover():
    result = assess_phase1({"correctness": {"passed": False}, "benchmarks": []})
    assert not result["passed"]
    assert len(result["failures"]) == 2
