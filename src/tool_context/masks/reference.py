from __future__ import annotations

from dataclasses import dataclass

from ..packing import PackedEpisode, TokenRole
from ..schema import CacheSemantics, ValidationError


@dataclass(frozen=True, slots=True)
class MaskOptions:
    cache_semantics: CacheSemantics = CacheSemantics.PERSISTENT_ARTIFACT
    answer_sees_memory: bool = False


def visible(packed: PackedEpisode, query: int, key: int, options: MaskOptions = MaskOptions()) -> bool:
    q_role = packed.token_role[query]
    k_role = packed.token_role[key]
    if not packed.valid_token[key]:
        return False
    if not packed.valid_token[query]:
        return key == packed.safe_key_index

    q_local = packed.local_position[query]
    k_local = packed.local_position[key]

    if q_role == TokenRole.G:
        return k_role == TokenRole.G and k_local <= q_local

    if q_role == TokenRole.T:
        same_tool = (
            k_role == TokenRole.T
            and packed.block_id[key] == packed.block_id[query]
            and k_local <= q_local
        )
        episode_anchor = (
            options.cache_semantics == CacheSemantics.EPISODE_ANCHORED
            and k_role == TokenRole.G
            and packed.episode_anchor_key[key]
        )
        return same_tool or episode_anchor

    if q_role == TokenRole.M:
        same_block = packed.block_id[key] == packed.block_id[query]
        return same_block and (
            k_role == TokenRole.T or (k_role == TokenRole.M and k_local <= q_local)
        )

    if q_role == TokenRole.R:
        return (
            k_role in (TokenRole.G, TokenRole.M)
            or (k_role == TokenRole.R and k_local <= q_local)
        )

    if q_role == TokenRole.A:
        if k_role in (TokenRole.G, TokenRole.R):
            return True
        if options.answer_sees_memory and k_role == TokenRole.M:
            return True
        if k_role == TokenRole.T:
            return packed.selected_block[key]
        return k_role == TokenRole.A and k_local <= q_local

    return key == packed.safe_key_index


def build_reference_mask(
    packed: PackedEpisode,
    cache_semantics: CacheSemantics = CacheSemantics.PERSISTENT_ARTIFACT,
    *,
    answer_sees_memory: bool = False,
    max_tokens: int = 4096,
) -> tuple[bytes, ...]:
    """Build the dense golden mask for small correctness cases.

    Production-sized layouts intentionally exceed this backend's limit; Phase 1
    consumes ``visible`` directly when constructing a sparse BlockMask.
    """
    length = packed.layout.sequence_length
    if length > max_tokens:
        raise ValidationError(
            f"dense reference mask is limited to {max_tokens} tokens, got {length}; "
            "use visible() or a tiny test layout"
        )
    options = MaskOptions(cache_semantics, answer_sees_memory)
    return tuple(
        bytes(1 if visible(packed, query, key, options) else 0 for key in range(length))
        for query in range(length)
    )

