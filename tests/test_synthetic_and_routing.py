from collections import Counter

from pathlib import Path

from tool_context.data.synthetic import DEFAULT_DISTRIBUTION, corpus_manifest, generate_corpus, read_corpus
from tool_context.eval.routing import evaluate_selectors, support_recalled
from tool_context.packing import ByteTokenizer
from tool_context.routing import BM25Selector, DenseSelector, EmbeddingSelector, OracleSelector, RandomSelector


class KeywordEncoder:
    identity = {"kind": "test"}

    def encode(self, texts):
        vocabulary = sorted(set(" ".join(texts).lower().split()))
        return [[float(text.lower().split().count(token)) for token in vocabulary] for text in texts]


def test_default_corpus_distribution_splits_and_determinism():
    first = generate_corpus()
    second = generate_corpus()
    assert len(first) == 500
    assert corpus_manifest(first) == corpus_manifest(second)
    assert [e.content_hash for e in first] == [e.content_hash for e in second]
    counts = Counter(e.template_family.split("/")[0] for e in first)
    assert counts == Counter(DEFAULT_DISTRIBUTION)
    assert Counter(e.metadata["split"] for e in first) == {"development": 400, "held_out": 100}


def test_randomized_identifiers_do_not_encode_relevance():
    episodes = generate_corpus(distribution={"single_lookup": 20})
    relevant_positions = []
    for episode in episodes:
        support = next(iter(episode.acceptable_support_sets))
        relevant_positions.append(next(i for i, block in enumerate(episode.tool_blocks) if block.block_id in support))
        assert all(not block.block_id.startswith("relevant") for block in episode.tool_blocks)
        assert all("answer" not in block.invariant_metadata["path"] for block in episode.tool_blocks)
    assert len(set(relevant_positions)) > 4


def test_selectors_share_evaluation_interface_and_controls_recall_perfectly():
    episodes = generate_corpus(distribution={"single_lookup": 8, "two_block_join": 8, "no_tool": 4})
    selectors = [RandomSelector(), BM25Selector(), EmbeddingSelector(KeywordEncoder()), OracleSelector(), DenseSelector()]
    metrics = evaluate_selectors(episodes, selectors, ByteTokenizer(), budgets=(0.25,))
    assert set(metrics) == {"random", "bm25", "minilm", "oracle", "dense"}
    assert metrics["oracle"]["0.25"]["aggregate"]["support_recall"] == 1.0
    assert metrics["dense"]["0.25"]["aggregate"]["support_recall"] == 1.0
    assert metrics["oracle"]["0.25"]["aggregate"]["no_tool_false_positive_rate"] == 0.0
    assert metrics["dense"]["0.25"]["aggregate"]["mean_opened_token_fraction"] == 1.0


def test_support_sets_allow_multiple_valid_evidence_sets():
    episode = generate_corpus(distribution={"single_lookup": 1})[0]
    support = next(iter(episode.acceptable_support_sets))
    assert support_recalled(episode, support)
    assert not support_recalled(episode, set())


def test_committed_fixture_corpus_covers_all_diagnostic_families():
    fixtures = read_corpus(Path(__file__).parent / "fixtures" / "episodes.jsonl")
    assert len(fixtures) == 12
    families = {episode.template_family.split("/")[0] for episode in fixtures}
    assert families == {
        "single_lookup", "near_duplicate", "two_block_join", "three_block_chain",
        "stale_conflict", "aggregation", "no_tool",
    }
