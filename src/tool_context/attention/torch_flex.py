from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from ..masks.flex import TensorMaskMetadata
from ..masks.reference import MaskOptions
from ..packing import TokenRole
from ..schema import CacheSemantics, ValidationError


_COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, fullgraph=True, dynamic=False)


def flex_attention_forward(query: Tensor, key: Tensor, value: Tensor, block_mask: BlockMask) -> Tensor:
    return _COMPILED_FLEX_ATTENTION(
        query, key, value, block_mask=block_mask,
        enable_gqa=query.shape[1] != key.shape[1],
        # PyTorch 2.13's safe-row fast path produces NaNs when safety is provided
        # by different KV tiles within one query tile (valid and padding rows mix).
        kernel_options={"BACKEND": "TRITON", "ROWS_GUARANTEED_SAFE": False},
    )


def dense_causal_sdpa(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=query.shape[1] != key.shape[1],
    )


def dense_masked_sdpa(query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, enable_gqa=query.shape[1] != key.shape[1],
    )


def _sdpa(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None, *, causal: bool = False) -> Tensor:
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=causal,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def segmented_sdpa(
    query: Tensor, key: Tensor, value: Tensor, metadata: TensorMaskMetadata,
    options: MaskOptions = MaskOptions(),
) -> Tensor:
    """Execute the frozen topology as compact region-wise SDPA calls."""
    if metadata.batch_size != 1 or query.shape[0] != 1:
        raise ValidationError("segmented SDPA currently supports batch size 1")
    roles = metadata.token_role[0]; blocks = metadata.block_index[0]
    valid = metadata.valid_token[0]; selected = metadata.selected_block[0]
    output = torch.empty_like(query)

    def indices(condition: Tensor) -> Tensor:
        return torch.nonzero(condition, as_tuple=False).flatten()

    def assign(q_indices: Tensor, k_indices: Tensor, mask: Tensor | None = None, *, causal: bool = False) -> None:
        if q_indices.numel() == 0:
            return
        result = _sdpa(
            query.index_select(2, q_indices), key.index_select(2, k_indices),
            value.index_select(2, k_indices), mask, causal=causal,
        )
        output[:, :, q_indices, :] = result

    g = indices(valid & (roles == int(TokenRole.G)))
    assign(g, g, causal=True)
    tool_slots = torch.unique(blocks[valid & (roles == int(TokenRole.T))]).tolist()
    tool_indices = [indices(valid & (roles == int(TokenRole.T)) & (blocks == slot)) for slot in tool_slots]
    if (
        options.cache_semantics == CacheSemantics.PERSISTENT_ARTIFACT
        and tool_indices and len({item.numel() for item in tool_indices}) == 1
    ):
        q_batch = torch.cat([query.index_select(2, item) for item in tool_indices], dim=0)
        k_batch = torch.cat([key.index_select(2, item) for item in tool_indices], dim=0)
        v_batch = torch.cat([value.index_select(2, item) for item in tool_indices], dim=0)
        tool_outputs = _sdpa(q_batch, k_batch, v_batch, causal=True)
        for batch_index, tool in enumerate(tool_indices):
            output[:, :, tool, :] = tool_outputs[batch_index:batch_index + 1]
    else:
        for slot, tool in zip(tool_slots, tool_indices, strict=True):
            anchor = indices(valid & metadata.episode_anchor_key[0]); keys = torch.cat((anchor, tool))
            mask = torch.ones((tool.numel(), keys.numel()), dtype=torch.bool, device=query.device)
            mask[:, anchor.numel():] = torch.ones(
                (tool.numel(), tool.numel()), dtype=torch.bool, device=query.device,
            ).tril()
            assign(tool, keys, mask[None, None])
    memory_indices = [indices(valid & (roles == int(TokenRole.M)) & (blocks == slot)) for slot in tool_slots]
    nonempty_pairs = [(tool, memory) for tool, memory in zip(tool_indices, memory_indices, strict=True) if memory.numel()]
    if nonempty_pairs and len({(tool.numel(), memory.numel()) for tool, memory in nonempty_pairs}) == 1:
        tool_length = nonempty_pairs[0][0].numel(); memory_length = nonempty_pairs[0][1].numel()
        q_batch = torch.cat([query.index_select(2, memory) for _, memory in nonempty_pairs], dim=0)
        k_batch = torch.cat([key.index_select(2, torch.cat((tool, memory))) for tool, memory in nonempty_pairs], dim=0)
        v_batch = torch.cat([value.index_select(2, torch.cat((tool, memory))) for tool, memory in nonempty_pairs], dim=0)
        mask = torch.ones((memory_length, tool_length + memory_length), dtype=torch.bool, device=query.device)
        mask[:, tool_length:] = torch.ones((memory_length, memory_length), dtype=torch.bool, device=query.device).tril()
        memory_outputs = _sdpa(q_batch, k_batch, v_batch, mask[None, None])
        for batch_index, (_, memory) in enumerate(nonempty_pairs):
            output[:, :, memory, :] = memory_outputs[batch_index:batch_index + 1]
    else:
        for tool, memory in nonempty_pairs:
            keys = torch.cat((tool, memory))
            mask = torch.ones((memory.numel(), keys.numel()), dtype=torch.bool, device=query.device)
            mask[:, tool.numel():] = torch.ones(
                (memory.numel(), memory.numel()), dtype=torch.bool, device=query.device,
            ).tril()
            assign(memory, keys, mask[None, None])
    memory_all = indices(valid & (roles == int(TokenRole.M)))
    router = indices(valid & (roles == int(TokenRole.R)))
    if router.numel():
        keys = torch.cat((g, memory_all, router)); prefix_length = g.numel() + memory_all.numel()
        mask = torch.ones((router.numel(), keys.numel()), dtype=torch.bool, device=query.device)
        mask[:, prefix_length:] = torch.ones(
            (router.numel(), router.numel()), dtype=torch.bool, device=query.device,
        ).tril()
        assign(router, keys, mask[None, None])
    answer = indices(valid & (roles == int(TokenRole.A)))
    if answer.numel():
        prefixes = [g, router, indices(valid & (roles == int(TokenRole.T)) & selected)]
        if options.answer_sees_memory:
            prefixes.append(memory_all)
        prefix = torch.cat(prefixes); keys = torch.cat((prefix, answer))
        mask = torch.ones((answer.numel(), keys.numel()), dtype=torch.bool, device=query.device)
        mask[:, prefix.numel():] = torch.ones(
            (answer.numel(), answer.numel()), dtype=torch.bool, device=query.device,
        ).tril()
        assign(answer, keys, mask[None, None])
    padding = indices(~valid)
    if padding.numel():
        safe = int(metadata.safe_key_index[0].item()); safe_value = value[:, :, safe:safe + 1, :]
        if query.shape[1] != key.shape[1]:
            safe_value = safe_value.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        output[:, :, padding, :] = safe_value.expand(-1, -1, padding.numel(), -1)
    return output
