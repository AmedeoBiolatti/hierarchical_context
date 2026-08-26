from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask

from ..packing import PackedEpisode, TokenRole
from ..schema import CacheSemantics, ValidationError
from .reference import MaskOptions


@dataclass(frozen=True, slots=True)
class TensorMaskMetadata:
    token_role: Tensor
    block_index: Tensor
    local_position: Tensor
    valid_token: Tensor
    episode_anchor_key: Tensor
    selected_block: Tensor
    safe_key_index: Tensor
    signature: str

    @property
    def batch_size(self) -> int:
        return int(self.token_role.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.token_role.shape[1])

    @property
    def device(self) -> torch.device:
        return self.token_role.device

    def to(self, device: torch.device | str) -> "TensorMaskMetadata":
        return TensorMaskMetadata(
            token_role=self.token_role.to(device), block_index=self.block_index.to(device),
            local_position=self.local_position.to(device), valid_token=self.valid_token.to(device),
            episode_anchor_key=self.episode_anchor_key.to(device),
            selected_block=self.selected_block.to(device), safe_key_index=self.safe_key_index.to(device),
            signature=self.signature,
        )

    @classmethod
    def from_packed(
        cls, packed_episodes: Sequence[PackedEpisode], *, device: torch.device | str = "cpu",
    ) -> "TensorMaskMetadata":
        if not packed_episodes:
            raise ValidationError("at least one packed episode is required")
        first = packed_episodes[0]
        if any(item.layout != first.layout for item in packed_episodes[1:]):
            raise ValidationError("all packed episodes in a mask batch must use the same layout")

        roles: list[list[int]] = []; blocks: list[list[int]] = []; locals_: list[list[int]] = []
        valid: list[list[bool]] = []; anchors: list[list[bool]] = []; selected: list[list[bool]] = []
        safe: list[int] = []; signature_rows: list[object] = []
        for item in packed_episodes:
            ordered_ids: dict[str, int] = {}
            for block_id in item.block_id:
                if block_id is not None and block_id not in ordered_ids:
                    ordered_ids[block_id] = len(ordered_ids)
            numeric_blocks = [ordered_ids.get(block_id, -1) for block_id in item.block_id]
            row_roles = [int(role) for role in item.token_role]
            roles.append(row_roles); blocks.append(numeric_blocks); locals_.append(list(item.local_position))
            valid.append(list(item.valid_token)); anchors.append(list(item.episode_anchor_key))
            selected.append(list(item.selected_block)); safe.append(item.safe_key_index)
            signature_rows.append([
                row_roles, numeric_blocks, list(item.local_position), list(item.valid_token),
                list(item.episode_anchor_key), list(item.selected_block), item.safe_key_index,
            ])
        signature = hashlib.sha256(json.dumps(signature_rows, separators=(",", ":")).encode()).hexdigest()
        return cls(
            token_role=torch.tensor(roles, dtype=torch.int8, device=device),
            block_index=torch.tensor(blocks, dtype=torch.int16, device=device),
            local_position=torch.tensor(locals_, dtype=torch.int32, device=device),
            valid_token=torch.tensor(valid, dtype=torch.bool, device=device),
            episode_anchor_key=torch.tensor(anchors, dtype=torch.bool, device=device),
            selected_block=torch.tensor(selected, dtype=torch.bool, device=device),
            safe_key_index=torch.tensor(safe, dtype=torch.int32, device=device), signature=signature,
        )


def mask_mod_from_metadata(metadata: TensorMaskMetadata, options: MaskOptions = MaskOptions()):
    role = metadata.token_role; block = metadata.block_index; local = metadata.local_position
    valid = metadata.valid_token; anchor = metadata.episode_anchor_key
    selected = metadata.selected_block; safe = metadata.safe_key_index
    episode_anchored = options.cache_semantics == CacheSemantics.EPISODE_ANCHORED
    answer_sees_memory = options.answer_sees_memory

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del head
        q_role = role[batch, query]; k_role = role[batch, key]
        q_valid = valid[batch, query]; k_valid = valid[batch, key]
        q_local = local[batch, query]; k_local = local[batch, key]
        same_block = block[batch, query] == block[batch, key]
        g_visible = (q_role == int(TokenRole.G)) & (k_role == int(TokenRole.G)) & (k_local <= q_local)
        t_visible = (q_role == int(TokenRole.T)) & (
            ((k_role == int(TokenRole.T)) & same_block & (k_local <= q_local))
            | (episode_anchored & (k_role == int(TokenRole.G)) & anchor[batch, key])
        )
        m_visible = (q_role == int(TokenRole.M)) & same_block & (
            (k_role == int(TokenRole.T)) | ((k_role == int(TokenRole.M)) & (k_local <= q_local))
        )
        r_visible = (q_role == int(TokenRole.R)) & (
            (k_role == int(TokenRole.G)) | (k_role == int(TokenRole.M))
            | ((k_role == int(TokenRole.R)) & (k_local <= q_local))
        )
        a_visible = (q_role == int(TokenRole.A)) & (
            (k_role == int(TokenRole.G)) | (k_role == int(TokenRole.R))
            | (answer_sees_memory & (k_role == int(TokenRole.M)))
            | ((k_role == int(TokenRole.T)) & selected[batch, key])
            | ((k_role == int(TokenRole.A)) & (k_local <= q_local))
        )
        ordinary = k_valid & (g_visible | t_visible | m_visible | r_visible | a_visible)
        padding = (~q_valid) & (key == safe[batch])
        return torch.where(q_valid, ordinary, padding)

    return mask_mod


_BLOCK_MASK_CACHE: dict[tuple[object, ...], BlockMask] = {}
_COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask, fullgraph=True, dynamic=False)


def clear_block_mask_cache() -> None:
    _BLOCK_MASK_CACHE.clear()


def build_flex_block_mask(
    metadata: TensorMaskMetadata, options: MaskOptions = MaskOptions(), *,
    block_size: int = 128, use_cache: bool = True,
) -> BlockMask:
    if metadata.device.type != "cuda":
        raise ValidationError("FlexAttention BlockMask metadata must be on CUDA")
    cache_key = (
        metadata.signature, str(metadata.device), block_size,
        options.cache_semantics.value, options.answer_sees_memory,
    )
    if use_cache and cache_key in _BLOCK_MASK_CACHE:
        return _BLOCK_MASK_CACHE[cache_key]
    mask = _COMPILED_CREATE_BLOCK_MASK(
        mask_mod_from_metadata(metadata, options), B=metadata.batch_size, H=None,
        Q_LEN=metadata.sequence_length, KV_LEN=metadata.sequence_length,
        device=metadata.device, BLOCK_SIZE=block_size, separate_full_blocks=True,
    )
    if use_cache:
        _BLOCK_MASK_CACHE[cache_key] = mask
    return mask


def build_dense_mask(metadata: TensorMaskMetadata, options: MaskOptions = MaskOptions()) -> Tensor:
    return create_mask(
        mask_mod_from_metadata(metadata, options), B=metadata.batch_size, H=None,
        Q_LEN=metadata.sequence_length, KV_LEN=metadata.sequence_length, device=metadata.device,
    )
