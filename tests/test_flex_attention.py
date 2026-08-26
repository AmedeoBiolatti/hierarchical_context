import pytest

torch = pytest.importorskip("torch")

from tool_context.attention.torch_flex import (  # noqa: E402
    dense_causal_sdpa,
    dense_masked_sdpa,
    flex_attention_forward,
    segmented_sdpa,
)
from tool_context.masks.flex import (  # noqa: E402
    TensorMaskMetadata,
    build_dense_mask,
    build_flex_block_mask,
    mask_mod_from_metadata,
)
from tool_context.masks.reference import visible  # noqa: E402
from tool_context.packing import TokenRole, pack_episode  # noqa: E402
from tool_context.schema import Anchor, EpisodeGraph, LayoutSpec, ToolBlock, ValidationError  # noqa: E402


class WordTokenizer:
    identity = {"kind": "word-test"}

    def encode(self, text):
        return [sum(word.encode()) % 251 + 1 for word in text.split()]


def packed(selected=("left",), tile_size=128):
    episode = EpisodeGraph(
        episode_id="flex", anchor=Anchor("answer task", "be exact", "episode anchor"),
        tool_blocks=(ToolBlock("left", "left evidence"), ToolBlock("right", "right evidence")),
        expected_answer="done", acceptable_support_sets=(frozenset({"left"}),),
        requires_tool=True, template_family="test/development", seed=1,
    )
    layout = LayoutSpec(
        "flex-test", tile_size, 2, tile_size, memory_tokens_per_block=2,
        router_capacity=2, answer_capacity=4, tile_size=tile_size,
    )
    return pack_episode(episode, layout, WordTokenizer(), selected_blocks=set(selected))


def test_tensor_metadata_predicate_matches_frozen_reference():
    item = packed(tile_size=16)
    metadata = TensorMaskMetadata.from_packed([item])
    predicate = mask_mod_from_metadata(metadata)
    for query in range(item.layout.sequence_length):
        for key in range(item.layout.sequence_length):
            observed = bool(predicate(torch.tensor(0), torch.tensor(0), torch.tensor(query), torch.tensor(key)))
            assert observed == visible(item, query, key)


def test_tensor_metadata_rejects_mixed_layout_batch():
    with pytest.raises(ValidationError, match="same layout"):
        TensorMaskMetadata.from_packed([packed(tile_size=16), packed(tile_size=128)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_forward_backward_and_segmented_match_dense_reference():
    torch.manual_seed(1729)
    item = packed()
    metadata = TensorMaskMetadata.from_packed([item], device="cuda")
    dense_mask = build_dense_mask(metadata)
    block_mask = build_flex_block_mask(metadata, use_cache=False)
    shape_q = (1, 4, item.layout.sequence_length, 32)
    shape_kv = (1, 2, item.layout.sequence_length, 32)
    base_q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
    base_k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
    base_v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)

    dense_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (base_q, base_k, base_v)]
    flex_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (base_q, base_k, base_v)]
    dense = dense_masked_sdpa(*dense_inputs, dense_mask)
    flex = flex_attention_forward(*flex_inputs, block_mask)
    segmented = segmented_sdpa(base_q, base_k, base_v, metadata)
    torch.testing.assert_close(flex, dense, rtol=0.03, atol=0.03)
    torch.testing.assert_close(segmented, dense, rtol=0.03, atol=0.03)
    dense.float().square().mean().backward(); flex.float().square().mean().backward()
    for dense_grad, flex_grad in zip(
        (item.grad for item in dense_inputs), (item.grad for item in flex_inputs), strict=True,
    ):
        torch.testing.assert_close(flex_grad, dense_grad, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_dense_degenerate_mask_matches_causal_sdpa():
    from torch.nn.attention.flex_attention import create_block_mask

    length = 256
    causal = lambda b, h, q, k: q >= k
    block_mask = torch.compile(create_block_mask, fullgraph=True, dynamic=False)(
        causal, B=1, H=None, Q_LEN=length, KV_LEN=length, device="cuda", BLOCK_SIZE=128,
    )
    torch.manual_seed(7)
    query = torch.randn(1, 4, length, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, length, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    torch.testing.assert_close(
        flex_attention_forward(query, key, value, block_mask),
        dense_causal_sdpa(query, key, value), rtol=0.03, atol=0.03,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_isolation_selection_padding_boundaries_and_batching():
    left = packed(("left",)); right = packed(("right",))
    metadata_left = TensorMaskMetadata.from_packed([left], device="cuda")
    metadata_right = TensorMaskMetadata.from_packed([right], device="cuda")
    mask_left = build_dense_mask(metadata_left); mask_right = build_dense_mask(metadata_right)
    changed_rows = torch.nonzero((mask_left != mask_right).any(dim=-1)[0, 0]).flatten().tolist()
    assert changed_rows and all(left.token_role[index] == TokenRole.A for index in changed_rows)
    for boundary in (127, 128, 129, 255, 256, 257):
        assert 0 <= boundary < left.layout.sequence_length
        assert bool(mask_left[0, 0, boundary].any())
    padding_rows = torch.tensor([not valid for valid in left.valid_token], device="cuda")
    assert torch.all(mask_left[0, 0, padding_rows].sum(dim=-1) == 1)

    batch_metadata = TensorMaskMetadata.from_packed([left, right], device="cuda")
    batch_mask = build_flex_block_mask(batch_metadata, use_cache=False)
    length = left.layout.sequence_length
    query = torch.randn(2, 4, length, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, length, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    assert flex_attention_forward(query, key, value, batch_mask).shape == query.shape

    left_tool = [i for i, role in enumerate(left.token_role) if role == TokenRole.T and left.block_id[i] == "left"]
    right_tool = [i for i, role in enumerate(left.token_role) if role == TokenRole.T and left.block_id[i] == "right"]
    base = flex_attention_forward(query[:1], key[:1], value[:1], build_flex_block_mask(metadata_left))
    changed_key = key[:1].clone(); changed_value = value[:1].clone()
    changed_key[:, :, right_tool] += 10; changed_value[:, :, right_tool] += 10
    perturbed = flex_attention_forward(query[:1], changed_key, changed_value, build_flex_block_mask(metadata_left))
    torch.testing.assert_close(base[:, :, left_tool], perturbed[:, :, left_tool], rtol=0, atol=0)
