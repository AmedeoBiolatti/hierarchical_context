from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .schema import EpisodeGraph, LayoutSpec, ValidationError, canonical_json


PACKING_VERSION = 1
DEFAULT_QWEN_TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class TokenRole(IntEnum):
    PAD = 0
    G = 1
    T = 2
    M = 3
    R = 4
    A = 5


class Tokenizer(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def encode(self, text: str) -> Sequence[int]: ...


class ByteTokenizer:
    """Small deterministic tokenizer used by fixtures and offline tests."""

    @property
    def identity(self) -> dict[str, Any]:
        return {"kind": "byte", "version": 1}

    def encode(self, text: str) -> list[int]:
        return [byte + 3 for byte in text.encode("utf-8")]


class HuggingFaceTokenizer:
    """Adapter which records the resolved Hub revision when available."""

    def __init__(
        self, model_id: str = "Qwen/Qwen3-0.6B",
        revision: str | None = DEFAULT_QWEN_TOKENIZER_REVISION,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'tokenizers' extra to use HuggingFaceTokenizer") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model_id = model_id
        init_kwargs = getattr(self._tokenizer, "init_kwargs", {})
        self._revision = revision or init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
        if self._revision is None:
            for key in ("vocab_file", "merges_file", "tokenizer_file"):
                candidate = init_kwargs.get(key)
                if not candidate:
                    continue
                parts = Path(candidate).parts
                if "snapshots" in parts:
                    position = parts.index("snapshots")
                    if position + 1 < len(parts):
                        self._revision = parts[position + 1]
                        break

    @property
    def identity(self) -> dict[str, Any]:
        return {"kind": "huggingface", "model_id": self._model_id, "revision": self._revision}

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True, slots=True)
class PackedEpisode:
    episode_id: str
    layout: LayoutSpec
    tokenizer_identity: dict[str, Any]
    input_ids: tuple[int, ...]
    token_role: tuple[TokenRole, ...]
    block_id: tuple[str | None, ...]
    local_position: tuple[int, ...]
    event_position: tuple[int, ...]
    selected_block: tuple[bool, ...]
    valid_token: tuple[bool, ...]
    episode_anchor_key: tuple[bool, ...]

    def __post_init__(self) -> None:
        expected = self.layout.sequence_length
        arrays: Iterable[tuple[str, Sequence[Any]]] = (
            ("input_ids", self.input_ids), ("token_role", self.token_role),
            ("block_id", self.block_id), ("local_position", self.local_position),
            ("event_position", self.event_position), ("selected_block", self.selected_block),
            ("valid_token", self.valid_token), ("episode_anchor_key", self.episode_anchor_key),
        )
        for name, values in arrays:
            if len(values) != expected:
                raise ValidationError(f"packed {name} has length {len(values)}, expected {expected}")

    @property
    def safe_key_index(self) -> int:
        for index, (role, valid) in enumerate(zip(self.token_role, self.valid_token, strict=True)):
            if valid and role == TokenRole.G:
                return index
        raise ValidationError("packed episode has no valid global token for safe padding attention")

    def to_dict(self) -> dict[str, Any]:
        return {
            "packing_version": PACKING_VERSION,
            "episode_id": self.episode_id,
            "layout": {
                key: getattr(self.layout, key)
                for key in (
                    "name", "global_capacity", "tool_count", "tool_capacity",
                    "memory_tokens_per_block", "router_capacity", "answer_capacity", "tile_size",
                )
            },
            "tokenizer_identity": self.tokenizer_identity,
            "input_ids": list(self.input_ids),
            "token_role": [int(role) for role in self.token_role],
            "block_id": list(self.block_id),
            "local_position": list(self.local_position),
            "event_position": list(self.event_position),
            "selected_block": list(self.selected_block),
            "valid_token": list(self.valid_token),
            "episode_anchor_key": list(self.episode_anchor_key),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _tool_text(block: Any) -> str:
    header = canonical_json({
        "tool_type": block.tool_type,
        "version": block.version,
        "invariant_metadata": dict(block.invariant_metadata),
        "content_hash": block.content_hash,
    })
    return f"[TOOL]\n{header}\n[CONTENT]\n{block.content}"


def pack_episode(
    episode: EpisodeGraph,
    layout: LayoutSpec,
    tokenizer: Tokenizer,
    *,
    selected_blocks: Iterable[str] | None = None,
    placeholder_token_id: int = 0,
) -> PackedEpisode:
    if len(episode.tool_blocks) > layout.tool_count:
        raise ValidationError(
            f"episode has {len(episode.tool_blocks)} tools but layout allows {layout.tool_count}"
        )
    known = {block.block_id for block in episode.tool_blocks}
    selected = known if selected_blocks is None else set(selected_blocks)
    unknown = selected.difference(known)
    if unknown:
        raise ValidationError(f"selected_blocks references unknown blocks: {sorted(unknown)}")

    ids: list[int] = []
    roles: list[TokenRole] = []
    block_ids: list[str | None] = []
    local_positions: list[int] = []
    event_positions: list[int] = []
    selections: list[bool] = []
    valid: list[bool] = []
    anchor_keys: list[bool] = []

    def append_region(
        token_ids: Sequence[int], capacity: int, role: TokenRole, *,
        block_id: str | None = None, event_position: int = 0,
        selected_block: bool = False, anchor_start: int | None = None,
    ) -> None:
        if len(token_ids) > capacity:
            raise ValidationError(
                f"{role.name} region for {block_id or 'global'} needs {len(token_ids)} tokens; capacity is {capacity}"
            )
        for local, token_id in enumerate(token_ids):
            ids.append(int(token_id)); roles.append(role); block_ids.append(block_id)
            local_positions.append(local); event_positions.append(event_position)
            selections.append(selected_block); valid.append(True)
            anchor_keys.append(anchor_start is not None and local >= anchor_start)
        padding = capacity - len(token_ids)
        ids.extend([placeholder_token_id] * padding); roles.extend([TokenRole.PAD] * padding)
        block_ids.extend([None] * padding); local_positions.extend([-1] * padding)
        event_positions.extend([-1] * padding); selections.extend([False] * padding)
        valid.extend([False] * padding); anchor_keys.extend([False] * padding)

    prefix = "\n".join(part for part in (episode.anchor.system_instruction, episode.anchor.task) if part)
    prefix_ids = list(tokenizer.encode(prefix)) or [placeholder_token_id]
    anchor_ids = list(tokenizer.encode(episode.anchor.episode_tool_anchor))
    global_ids = prefix_ids + anchor_ids
    append_region(
        global_ids, layout.global_capacity, TokenRole.G,
        anchor_start=len(prefix_ids) if anchor_ids else None,
    )

    for tool_index in range(layout.tool_count):
        if tool_index < len(episode.tool_blocks):
            block = episode.tool_blocks[tool_index]
            append_region(
                tokenizer.encode(_tool_text(block)), layout.tool_capacity, TokenRole.T,
                block_id=block.block_id, event_position=tool_index + 1,
                selected_block=block.block_id in selected,
            )
        else:
            append_region((), layout.tool_capacity, TokenRole.T, event_position=tool_index + 1)

    memory_ids: list[int] = []
    memory_blocks: list[str] = []
    for block in episode.tool_blocks:
        memory_ids.extend([placeholder_token_id] * layout.memory_tokens_per_block)
        memory_blocks.extend([block.block_id] * layout.memory_tokens_per_block)
    for local, (token_id, memory_block) in enumerate(zip(memory_ids, memory_blocks, strict=True)):
        block_local = local % layout.memory_tokens_per_block
        ids.append(token_id); roles.append(TokenRole.M); block_ids.append(memory_block)
        local_positions.append(block_local); event_positions.append(1 + known_index(episode, memory_block))
        selections.append(memory_block in selected); valid.append(True); anchor_keys.append(False)
    memory_padding = layout.summary_capacity - len(memory_ids)
    ids.extend([placeholder_token_id] * memory_padding); roles.extend([TokenRole.PAD] * memory_padding)
    block_ids.extend([None] * memory_padding); local_positions.extend([-1] * memory_padding)
    event_positions.extend([-1] * memory_padding); selections.extend([False] * memory_padding)
    valid.extend([False] * memory_padding); anchor_keys.extend([False] * memory_padding)

    append_region(
        [placeholder_token_id] * layout.router_capacity,
        layout.router_strip_capacity, TokenRole.R, event_position=layout.tool_count + 1,
    )
    append_region(
        tokenizer.encode(episode.expected_answer), layout.answer_strip_capacity,
        TokenRole.A, event_position=layout.tool_count + 2,
    )

    return PackedEpisode(
        episode_id=episode.episode_id, layout=layout,
        tokenizer_identity=dict(tokenizer.identity), input_ids=tuple(ids), token_role=tuple(roles),
        block_id=tuple(block_ids), local_position=tuple(local_positions),
        event_position=tuple(event_positions), selected_block=tuple(selections),
        valid_token=tuple(valid), episode_anchor_key=tuple(anchor_keys),
    )


def known_index(episode: EpisodeGraph, block_id: str) -> int:
    for index, block in enumerate(episode.tool_blocks):
        if block.block_id == block_id:
            return index
    raise ValidationError(f"unknown block {block_id!r}")
