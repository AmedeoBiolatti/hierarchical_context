from collections import Counter

from tool_context.data.phase2 import build_phase2_corpus, phase2_manifest
from tool_context.data.synthetic import generate_corpus
from tool_context.packing import ByteTokenizer


def test_phase2_corpus_is_stratified_positioned_and_deterministic():
    tokenizer = ByteTokenizer()
    first = build_phase2_corpus(generate_corpus(), tokenizer)
    second = build_phase2_corpus(generate_corpus(), tokenizer)
    assert len(first) == 63
    assert phase2_manifest(first, tokenizer.identity) == phase2_manifest(second, tokenizer.identity)
    assert Counter(item.placement for item in first) == {"beginning": 21, "middle": 21, "end": 21}
    assert Counter(item.episode.template_family.split("/")[0] for item in first) == {
        "aggregation": 9, "near_duplicate": 9, "no_tool": 9, "single_lookup": 9,
        "stale_conflict": 9, "three_block_chain": 9, "two_block_join": 9,
    }
    grouped = {}
    for item in first:
        base = item.episode.metadata["phase2_base_episode_id"]
        grouped.setdefault(base, {})[item.placement] = dict(item.content_evidence_offsets)
    for placements in grouped.values():
        for block_id in placements["beginning"]:
            assert placements["beginning"][block_id] < placements["middle"][block_id]
            assert placements["middle"][block_id] < placements["end"][block_id]
