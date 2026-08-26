import pytest

from tool_context.masks.reference import MaskOptions, build_reference_mask, visible
from tool_context.packing import ByteTokenizer, PackedEpisode, TokenRole, pack_episode
from tool_context.schema import Anchor, CacheSemantics, EpisodeGraph, LayoutSpec, ToolBlock, ValidationError


class WordTokenizer:
    identity = {"kind": "word-test", "version": 1}

    def encode(self, text):
        return [sum(word.encode()) % 251 + 1 for word in text.split()]


def make_episode():
    return EpisodeGraph(
        episode_id="mask", anchor=Anchor("answer the task", "be exact", "shared anchor"),
        tool_blocks=(ToolBlock("left", "left evidence"), ToolBlock("right", "right evidence")),
        expected_answer="done", acceptable_support_sets=(frozenset({"left"}),),
        requires_tool=True, template_family="mask/development", seed=7,
    )


def make_layout(tile_size=1):
    return LayoutSpec("tiny", 16 if tile_size == 1 else 128, 2, 32 if tile_size == 1 else 128,
                      memory_tokens_per_block=2, router_capacity=2, answer_capacity=8,
                      tile_size=tile_size)


def indices(packed, role, block=None):
    return [i for i, item in enumerate(packed.token_role)
            if item == role and (block is None or packed.block_id[i] == block)]


def test_packing_is_byte_identical_and_selection_is_explicit():
    kwargs = dict(selected_blocks={"left"})
    first = pack_episode(make_episode(), make_layout(), WordTokenizer(), **kwargs)
    second = pack_episode(make_episode(), make_layout(), WordTokenizer(), **kwargs)
    assert first.canonical_json() == second.canonical_json()
    assert all(first.selected_block[i] for i in indices(first, TokenRole.T, "left"))
    assert not any(first.selected_block[i] for i in indices(first, TokenRole.T, "right"))


def test_mask_contract_and_no_cross_tool_visibility():
    packed = pack_episode(make_episode(), make_layout(), WordTokenizer(), selected_blocks={"left"})
    left = indices(packed, TokenRole.T, "left")
    right = indices(packed, TokenRole.T, "right")
    memories = indices(packed, TokenRole.M, "left")
    routers = indices(packed, TokenRole.R)
    answers = indices(packed, TokenRole.A)

    assert visible(packed, left[-1], left[0])
    assert not visible(packed, left[0], left[-1])
    assert not visible(packed, left[-1], right[0])
    assert visible(packed, memories[0], left[-1])
    assert not visible(packed, memories[0], right[0])
    assert visible(packed, routers[0], memories[-1])
    assert visible(packed, answers[0], left[-1])
    assert not visible(packed, answers[0], right[0])
    assert not visible(packed, answers[0], memories[0])
    assert visible(packed, answers[0], memories[0], MaskOptions(answer_sees_memory=True))


def test_cache_semantics_only_change_tool_anchor_visibility():
    packed = pack_episode(make_episode(), make_layout(), WordTokenizer())
    tool = indices(packed, TokenRole.T, "left")[0]
    anchor_key = [i for i, value in enumerate(packed.episode_anchor_key) if value][0]
    ordinary_global = indices(packed, TokenRole.G)[0]
    assert not visible(packed, tool, anchor_key)
    episode_options = MaskOptions(CacheSemantics.EPISODE_ANCHORED)
    assert visible(packed, tool, anchor_key, episode_options)
    assert not visible(packed, tool, ordinary_global, episode_options)


def test_padding_queries_have_exactly_one_safe_key():
    packed = pack_episode(make_episode(), make_layout(), WordTokenizer())
    pad = indices(packed, TokenRole.PAD)[0]
    visible_keys = [key for key in range(packed.layout.sequence_length) if visible(packed, pad, key)]
    assert visible_keys == [packed.safe_key_index]


def test_dense_reference_matches_predicate():
    packed = pack_episode(make_episode(), make_layout(), WordTokenizer(), selected_blocks={"left"})
    mask = build_reference_mask(packed)
    for query, row in enumerate(mask):
        assert list(row) == [int(visible(packed, query, key)) for key in range(len(row))]


def test_exact_handwritten_five_role_truth_table():
    layout = LayoutSpec("truth", 1, 1, 1, memory_tokens_per_block=1,
                        router_capacity=1, answer_capacity=1, tile_size=1)
    packed = PackedEpisode(
        episode_id="truth", layout=layout, tokenizer_identity={"kind": "manual"},
        input_ids=(1, 2, 0, 0, 3),
        token_role=(TokenRole.G, TokenRole.T, TokenRole.M, TokenRole.R, TokenRole.A),
        block_id=(None, "b", "b", None, None),
        local_position=(0, 0, 0, 0, 0), event_position=(0, 1, 1, 2, 3),
        selected_block=(False, True, True, False, False),
        valid_token=(True, True, True, True, True),
        episode_anchor_key=(False, False, False, False, False),
    )
    assert [list(row) for row in build_reference_mask(packed)] == [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
    ]


def test_selection_bitmap_only_changes_answer_query_rows():
    left = pack_episode(make_episode(), make_layout(), WordTokenizer(), selected_blocks={"left"})
    right = pack_episode(make_episode(), make_layout(), WordTokenizer(), selected_blocks={"right"})
    left_mask = build_reference_mask(left)
    right_mask = build_reference_mask(right)
    changed_queries = {
        query for query, (left_row, right_row) in enumerate(zip(left_mask, right_mask, strict=True))
        if left_row != right_row
    }
    assert changed_queries
    assert all(left.token_role[query] == TokenRole.A for query in changed_queries)


def test_tile_edges_and_production_dense_guard():
    packed = pack_episode(make_episode(), make_layout(128), WordTokenizer())
    assert packed.token_role[127] == TokenRole.PAD
    assert packed.token_role[128] == TokenRole.T
    assert packed.token_role[255] in (TokenRole.T, TokenRole.PAD)
    large = LayoutSpec.standard_layouts()[-1]
    large_packed = pack_episode(make_episode(), large, WordTokenizer())
    with pytest.raises(ValidationError, match="limited"):
        build_reference_mask(large_packed)


def test_overflow_and_unknown_selection_fail_instead_of_truncating():
    with pytest.raises(ValidationError, match="capacity"):
        pack_episode(make_episode(), LayoutSpec("small", 1, 2, 32, tile_size=1), WordTokenizer())
    with pytest.raises(ValidationError, match="unknown"):
        pack_episode(make_episode(), make_layout(), WordTokenizer(), selected_blocks={"missing"})


def test_byte_tokenizer_identity_is_versioned():
    tokenizer = ByteTokenizer()
    assert tokenizer.identity == {"kind": "byte", "version": 1}
    assert tokenizer.encode("A") == [68]
