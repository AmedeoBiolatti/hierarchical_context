from tool_context.acceptance import assess_phase0, assess_phase1, assess_phase2, assess_phase3


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


def test_phase2_acceptance_requires_complete_measurements_not_quality_thresholds():
    families = ["single_lookup", "near_duplicate", "two_block_join", "three_block_chain",
                "stale_conflict", "aggregation", "no_tool"]
    placements = ["beginning", "middle", "end"]
    specs = [("dense_compact", None), ("dense_aligned", None), ("block_static", None)]
    specs += [(mode, seed) for mode in ("block_memory", "block_oracle", "block_dynamic_anchor")
              for seed in (1729, 1730, 1731)]
    rows = []
    for family in families:
        for placement in placements:
            for example in range(3):
                for mode, seed in specs:
                    rows.append({"family": family, "placement": placement, "mode": mode, "seed": seed,
                                 "nll_sum": 999.0, "nll_per_token": 999.0,
                                 "token_accuracy": 0.0, "exact_match": False})
    quality = {
        "corpus": {"example_count": 63}, "rows": rows,
        "correctness": {
            "dense_input_ids_vs_inputs_embeds_max_abs_error": 0.0,
            "cross_block_max_abs_error": 0.0, "same_block_mask_object_all_layers": True,
            "token_bank_only_overwrites_memory_router": True, "padding_logits_finite": True,
            "causal_flex_vs_sdpa": {"nll_difference": .01, "logit_cosine": .9999, "top1_agreement": 1.0},
        },
    }
    prefill = {"fresh_process_per_case": True, "benchmarks": []}
    for length in (4096, 8192, 16384):
        for mode in ("dense_aligned", "block_static", "block_memory", "block_oracle"):
            prefill["benchmarks"].append({
                "sequence_length": length, "layout": str(length), "mode": mode, "status": "ok",
                "model_load_ms": 1, "mask_and_input_build_ms": 1, "first_call_wall_ms": 1,
                "median_ms": 1, "p10_ms": 1, "p90_ms": 1,
                "peak_allocated_mib": 1, "peak_reserved_mib": 1,
            })
    assert assess_phase2(quality, prefill)["passed"]


def test_phase3_acceptance_uses_relative_quality_and_confirmation_gates():
    def section(nll, accuracy):
        return {"aggregate": {"nll_per_token": nll, "token_accuracy": accuracy},
                "families": {"two_block_join": {"token_accuracy": accuracy}},
                "rows": [{"episode_id": "x", "family": "two_block_join", "target_tokens": 2,
                          "token_accuracy": accuracy}]}
    report = {
        "frozen_dense": section(1.0, .9), "initial_oracle": section(1.5, .7),
        "adapted_dense": section(1.02, .9), "adapted_oracle": section(1.04, .89),
        "correctness": {"cross_block_max_abs_error": 0.0, "padding_logits_finite": True,
                        "same_block_mask_object_all_layers": True},
    }
    thresholds = {
        "oracle_nll_gap_max": .05, "oracle_token_accuracy_gap_max": .02,
        "dense_nll_regression_max": .05, "dense_token_accuracy_regression_max": .01,
        "multihop_accuracy_gap_max": .03, "multihop_family_gap_max": .05,
        "confirmation_gap_reduction_min": .3, "confirmation_dense_nll_regression_max": .1,
    }
    result = assess_phase3(report, [report, report], thresholds)
    assert result["passed"]
