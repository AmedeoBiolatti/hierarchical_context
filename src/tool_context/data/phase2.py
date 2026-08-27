from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

from ..packing import Tokenizer
from ..schema import EpisodeGraph, ToolBlock, canonical_json


PHASE2_CORPUS_VERSION = 1
PLACEMENTS = ("beginning", "middle", "end")
_FILLER = "Neutral archival padding contains no task evidence."
_FILLER_REPETITIONS = 24


@dataclass(frozen=True, slots=True)
class Phase2Example:
    episode: EpisodeGraph
    placement: str
    content_evidence_offsets: tuple[tuple[str, int], ...]


def _position_content(content: str, placement: str) -> tuple[str, int]:
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown evidence placement {placement!r}")
    if placement == "beginning":
        before, after = 0, _FILLER_REPETITIONS
    elif placement == "middle":
        before = after = _FILLER_REPETITIONS // 2
    else:
        before, after = _FILLER_REPETITIONS, 0
    prefix = " ".join([_FILLER] * before)
    suffix = " ".join([_FILLER] * after)
    pieces = [piece for piece in (prefix, content, suffix) if piece]
    return " ".join(pieces), len(prefix)


def _transform(base: EpisodeGraph, placement: str, tokenizer: Tokenizer) -> Phase2Example:
    blocks: list[ToolBlock] = []
    offsets: list[tuple[str, int]] = []
    for block in base.tool_blocks:
        content, character_offset = _position_content(block.content, placement)
        token_offset = len(tokenizer.encode(content[:character_offset]))
        blocks.append(ToolBlock(
            block_id=block.block_id, content=content, tool_type=block.tool_type,
            invariant_metadata=block.invariant_metadata, version=block.version,
        ))
        offsets.append((block.block_id, token_offset))
    metadata = dict(base.metadata)
    metadata.update({
        "phase2_corpus_version": PHASE2_CORPUS_VERSION,
        "phase2_base_episode_id": base.episode_id,
        "evidence_placement": placement,
    })
    episode = EpisodeGraph(
        episode_id=f"phase2-{base.episode_id}-{placement}", anchor=base.anchor,
        tool_blocks=tuple(blocks), expected_answer=base.expected_answer,
        acceptable_support_sets=base.acceptable_support_sets, requires_tool=base.requires_tool,
        template_family=base.template_family, seed=base.seed, metadata=metadata,
    )
    return Phase2Example(episode, placement, tuple(offsets))


def build_phase2_corpus(
    episodes: Sequence[EpisodeGraph], tokenizer: Tokenizer, *, examples_per_family: int = 3,
) -> tuple[Phase2Example, ...]:
    held_out: dict[str, list[EpisodeGraph]] = defaultdict(list)
    for episode in episodes:
        if episode.metadata.get("split") == "held_out":
            held_out[episode.template_family.split("/")[0]].append(episode)
    if len(held_out) != 7:
        raise ValueError(f"expected seven held-out families, found {sorted(held_out)}")
    result: list[Phase2Example] = []
    for family in sorted(held_out):
        selected = sorted(held_out[family], key=lambda item: item.episode_id)[:examples_per_family]
        if len(selected) != examples_per_family:
            raise ValueError(f"family {family} has only {len(selected)} held-out examples")
        for episode in selected:
            for placement in PLACEMENTS:
                result.append(_transform(episode, placement, tokenizer))
    return tuple(result)


def phase2_manifest(examples: Sequence[Phase2Example], tokenizer_identity: dict[str, Any]) -> dict[str, Any]:
    records = [
        {
            "episode": item.episode.to_dict(), "placement": item.placement,
            "content_evidence_offsets": list(item.content_evidence_offsets),
        }
        for item in examples
    ]
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    family_counts: dict[str, int] = defaultdict(int)
    placement_counts: dict[str, int] = defaultdict(int)
    for item in examples:
        family_counts[item.episode.template_family.split("/")[0]] += 1
        placement_counts[item.placement] += 1
    return {
        "phase2_corpus_version": PHASE2_CORPUS_VERSION,
        "example_count": len(examples), "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "family_counts": dict(sorted(family_counts.items())),
        "placement_counts": dict(sorted(placement_counts.items())),
        "tokenizer": tokenizer_identity,
    }
