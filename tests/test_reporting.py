from tool_context.reporting import result_document


def test_result_document_reserves_phase1_performance_fields():
    result = result_document(
        corpus_manifest={"corpus_sha256": "abc"}, tokenizer_identity={"kind": "test"},
        selector_identities={"random": {}}, metrics={}, configuration={"budgets": [0.25]},
    )
    assert result["result_schema_version"] == 1
    assert set(result["performance"]) == {
        "compile_time_ms", "mask_build_time_ms", "execution_time_ms",
        "peak_vram_mib", "active_token_fraction",
    }

