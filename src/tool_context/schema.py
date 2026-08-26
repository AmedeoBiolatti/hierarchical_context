from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1


class ValidationError(ValueError):
    """Raised when an experiment object violates the frozen contract."""


class CacheSemantics(StrEnum):
    PERSISTENT_ARTIFACT = "persistent_artifact"
    EPISODE_ANCHORED = "episode_anchored"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_digest(content: str, metadata: Mapping[str, Any]) -> str:
    payload = {"content": content, "invariant_metadata": dict(metadata)}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Anchor:
    task: str
    system_instruction: str = ""
    episode_tool_anchor: str = ""

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValidationError("anchor.task must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "system_instruction": self.system_instruction,
            "episode_tool_anchor": self.episode_tool_anchor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Anchor":
        return cls(
            task=str(data["task"]),
            system_instruction=str(data.get("system_instruction", "")),
            episode_tool_anchor=str(data.get("episode_tool_anchor", "")),
        )


@dataclass(frozen=True, slots=True)
class ToolBlock:
    block_id: str
    content: str
    tool_type: str = "text"
    invariant_metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValidationError("tool block_id must not be empty")
        if not self.content:
            raise ValidationError(f"tool block {self.block_id!r} has empty content")
        metadata = _frozen_mapping(self.invariant_metadata)
        object.__setattr__(self, "invariant_metadata", metadata)
        expected = content_digest(self.content, metadata)
        if self.content_hash and self.content_hash != expected:
            raise ValidationError(f"tool block {self.block_id!r} has an inconsistent content_hash")
        object.__setattr__(self, "content_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "content": self.content,
            "tool_type": self.tool_type,
            "invariant_metadata": dict(self.invariant_metadata),
            "version": self.version,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolBlock":
        return cls(
            block_id=str(data["block_id"]),
            content=str(data["content"]),
            tool_type=str(data.get("tool_type", "text")),
            invariant_metadata=data.get("invariant_metadata", {}),
            version=str(data.get("version", "")),
            content_hash=str(data.get("content_hash", "")),
        )


def _support_sets(value: Iterable[Iterable[str]]) -> tuple[frozenset[str], ...]:
    normalized = tuple(frozenset(map(str, support)) for support in value)
    if not normalized:
        raise ValidationError("acceptable_support_sets must contain at least one set")
    if len(set(normalized)) != len(normalized):
        raise ValidationError("acceptable_support_sets contains duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class EpisodeGraph:
    episode_id: str
    anchor: Anchor
    tool_blocks: tuple[ToolBlock, ...]
    expected_answer: str
    acceptable_support_sets: tuple[frozenset[str], ...]
    requires_tool: bool
    template_family: str
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValidationError("episode_id must not be empty")
        blocks = tuple(self.tool_blocks)
        object.__setattr__(self, "tool_blocks", blocks)
        ids = [block.block_id for block in blocks]
        if len(ids) != len(set(ids)):
            raise ValidationError(f"episode {self.episode_id!r} has duplicate block IDs")
        support_sets = _support_sets(self.acceptable_support_sets)
        object.__setattr__(self, "acceptable_support_sets", support_sets)
        unknown = set().union(*support_sets).difference(ids)
        if unknown:
            raise ValidationError(f"support sets reference unknown blocks: {sorted(unknown)}")
        has_empty = any(not support for support in support_sets)
        if self.requires_tool and has_empty:
            raise ValidationError("tool-required episodes cannot have an empty support set")
        if not self.requires_tool and not has_empty:
            raise ValidationError("no-tool episodes must include an empty support set")
        if not self.template_family.strip():
            raise ValidationError("template_family must not be empty")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "anchor": self.anchor.to_dict(),
            "tool_blocks": [block.to_dict() for block in self.tool_blocks],
            "expected_answer": self.expected_answer,
            "acceptable_support_sets": [sorted(s) for s in self.acceptable_support_sets],
            "requires_tool": self.requires_tool,
            "template_family": self.template_family,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeGraph":
        version = int(data.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version {version}")
        return cls(
            episode_id=str(data["episode_id"]),
            anchor=Anchor.from_dict(data["anchor"]),
            tool_blocks=tuple(ToolBlock.from_dict(item) for item in data["tool_blocks"]),
            expected_answer=str(data["expected_answer"]),
            acceptable_support_sets=tuple(frozenset(s) for s in data["acceptable_support_sets"]),
            requires_tool=bool(data["requires_tool"]),
            template_family=str(data["template_family"]),
            seed=int(data["seed"]),
            metadata=data.get("metadata", {}),
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    name: str
    global_capacity: int
    tool_count: int
    tool_capacity: int
    memory_tokens_per_block: int = 8
    router_capacity: int = 8
    answer_capacity: int = 256
    tile_size: int = 128

    def __post_init__(self) -> None:
        for field_name in (
            "global_capacity", "tool_count", "tool_capacity",
            "memory_tokens_per_block", "router_capacity", "answer_capacity", "tile_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValidationError(f"layout {field_name} must be positive")
        for field_name in ("global_capacity", "tool_capacity"):
            if getattr(self, field_name) % self.tile_size:
                raise ValidationError(f"layout {field_name} must be divisible by tile_size")
        if self.tool_count * self.memory_tokens_per_block > self.summary_capacity:
            raise ValidationError("summary strip capacity is too small")

    @staticmethod
    def _align(value: int, tile_size: int) -> int:
        return ((value + tile_size - 1) // tile_size) * tile_size

    @property
    def summary_capacity(self) -> int:
        return self._align(self.tool_count * self.memory_tokens_per_block, self.tile_size)

    @property
    def router_strip_capacity(self) -> int:
        return self._align(self.router_capacity, self.tile_size)

    @property
    def answer_strip_capacity(self) -> int:
        return self._align(self.answer_capacity, self.tile_size)

    @property
    def sequence_length(self) -> int:
        return (
            self.global_capacity + self.tool_count * self.tool_capacity
            + self.summary_capacity + self.router_strip_capacity + self.answer_strip_capacity
        )

    @classmethod
    def standard_layouts(cls) -> tuple["LayoutSpec", ...]:
        return (
            cls("g512_8x512", 512, 8, 512),
            cls("g512_8x1024", 512, 8, 1024),
            cls("g1024_16x2048", 1024, 16, 2048),
        )

