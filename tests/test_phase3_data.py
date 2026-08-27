from __future__ import annotations

import pytest

from tool_context.data.phase3 import ALL_FAMILIES, TRAIN_COUNTS, build_phase3_corpus, stratified_subset
from tool_context.packing import ByteTokenizer


@pytest.fixture(scope="module")
def corpus():
    return build_phase3_corpus(ByteTokenizer())


def test_phase3_corpus_has_frozen_counts_and_no_content_leakage(corpus) -> None:
    assert corpus.manifest["counts"] == {"train": 10_000, "development": 220, "test": 99}
    assert corpus.manifest["family_counts"]["train"] == dict(sorted(TRAIN_COUNTS.items()))
    assert corpus.manifest["family_counts"]["development"] == {family: 20 for family in sorted(ALL_FAMILIES)}
    assert set(corpus.manifest["content_hash_overlap"].values()) == {0}


def test_phase3_code_tasks_have_exact_support_and_stratified_subset(corpus) -> None:
    code = [item for item in corpus.train if item.episode.template_family.startswith("code_")]
    assert code and all(item.episode.acceptable_support_sets for item in code)
    subset = stratified_subset(corpus.train, 22, 1729)
    assert len(subset) == 22
    assert {item.episode.template_family.split("/")[0] for item in subset} == set(ALL_FAMILIES)
