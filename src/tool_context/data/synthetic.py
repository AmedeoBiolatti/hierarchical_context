from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Callable, Iterable

from ..schema import Anchor, EpisodeGraph, ToolBlock, canonical_json


CORPUS_VERSION = 1
DEFAULT_DISTRIBUTION = {
    "single_lookup": 100,
    "near_duplicate": 100,
    "two_block_join": 100,
    "three_block_chain": 50,
    "stale_conflict": 75,
    "aggregation": 50,
    "no_tool": 25,
}

_NAMES = ("amber", "cedar", "cobalt", "falcon", "indigo", "juniper", "lotus", "quartz")
_VALUES = ("K17", "M42", "P09", "R31", "T88", "V24", "X63", "Z55")


def _block(rng: random.Random, index: int, content: str, *, kind: str = "record") -> ToolBlock:
    opaque = hashlib.sha256(f"{rng.random()}:{index}".encode()).hexdigest()[:10]
    return ToolBlock(
        block_id=f"blk_{opaque}", content=content, tool_type=kind,
        invariant_metadata={"path": f"artifact_{rng.randrange(1000, 9999)}.txt", "format": "text"},
        version=f"v{rng.randrange(1, 20)}",
    )


def _finish(
    family: str, index: int, seed: int, rng: random.Random, task: str,
    answer: str, blocks: list[ToolBlock], supports: Iterable[Iterable[str]],
    *, requires_tool: bool = True, episode_anchor: str = "",
) -> EpisodeGraph:
    rng.shuffle(blocks)
    split = "held_out" if index % 5 == 0 else "development"
    return EpisodeGraph(
        episode_id=f"{family}-{index:04d}",
        anchor=Anchor(task=task, system_instruction="Answer exactly from the applicable records.", episode_tool_anchor=episode_anchor),
        tool_blocks=tuple(blocks), expected_answer=answer,
        acceptable_support_sets=tuple(frozenset(group) for group in supports),
        requires_tool=requires_tool, template_family=f"{family}/{split}", seed=seed,
        metadata={"split": split, "generator_version": CORPUS_VERSION},
    )


def _single(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    target = rng.choice(_NAMES); value = rng.choice(_VALUES)
    blocks = [_block(rng, 0, f"Project {target} has access code {value}. Routine status is green.")]
    relevant = blocks[0].block_id
    for i, name in enumerate(rng.sample([n for n in _NAMES if n != target], 7), 1):
        blocks.append(_block(rng, i, f"Project {name} has access code {rng.choice(_VALUES)}. Routine status is green."))
    return _finish("single_lookup", index, seed, rng, f"What is the access code for project {target}?", value, blocks, [[relevant]])


def _near_duplicate(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    service = rng.choice(_NAMES); environments = list(_NAMES); rng.shuffle(environments)
    target_env = environments[0]; value = rng.choice(_VALUES); blocks = []; relevant = ""
    for i, environment in enumerate(environments):
        block_value = value if environment == target_env else rng.choice([v for v in _VALUES if v != value])
        block = _block(rng, i, f"Service {service}; deployment environment {environment}; release key {block_value}.")
        blocks.append(block)
        if environment == target_env:
            relevant = block.block_id
    return _finish(
        "near_duplicate", index, seed, rng,
        f"Return the release key for service {service} in environment {target_env}.",
        value, blocks, [[relevant]],
    )


def _two_join(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    project = rng.choice(_NAMES); alias = f"node-{rng.randrange(100, 999)}"; value = rng.choice(_VALUES)
    first = _block(rng, 0, f"Project {project} is assigned internal node {alias}.")
    second = _block(rng, 1, f"Internal node {alias} currently publishes result {value}.")
    blocks = [first, second]
    for i in range(2, 8):
        other = f"node-{rng.randrange(100, 999)}"
        blocks.append(_block(rng, i, f"Internal node {other} currently publishes result {rng.choice(_VALUES)}."))
    return _finish("two_block_join", index, seed, rng, f"What result is published by the node assigned to project {project}?", value, blocks, [[first.block_id, second.block_id]])


def _three_chain(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    project = rng.choice(_NAMES); node = f"node-{rng.randrange(100, 999)}"
    slot = f"slot-{rng.randrange(20, 99)}"; value = rng.choice(_VALUES)
    first = _block(rng, 0, f"Project {project} delegates to internal node {node}.")
    second = _block(rng, 1, f"Internal node {node} resolves to storage slot {slot}.")
    third = _block(rng, 2, f"Storage slot {slot} contains final result {value}.")
    blocks = [first, second, third]
    for i in range(3, 8):
        blocks.append(_block(rng, i, f"Storage slot slot-{rng.randrange(20, 99)} contains final result {rng.choice(_VALUES)}."))
    return _finish("three_block_chain", index, seed, rng, f"Follow the delegation for project {project} and return its final result.", value, blocks, [[first.block_id, second.block_id, third.block_id]])


def _conflict(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    subject = rng.choice(_NAMES); revisions = list(range(1, 9)); rng.shuffle(revisions)
    authority = revisions[0]; value = rng.choice(_VALUES); blocks = []; relevant = ""
    for i, revision in enumerate(revisions):
        block_value = value if revision == authority else rng.choice([v for v in _VALUES if v != value])
        block = _block(rng, i, f"Record for {subject}; revision {revision}; configured value {block_value}.")
        blocks.append(block)
        if revision == authority:
            relevant = block.block_id
    episode_anchor = f"For this episode, the authoritative revision is {authority}."
    return _finish("stale_conflict", index, seed, rng, f"Give the authoritative configured value for {subject}.", value, blocks, [[relevant]], episode_anchor=episode_anchor)


def _aggregation(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    project = rng.choice(_NAMES); values = [rng.randrange(2, 30) for _ in range(3)]
    blocks = []; relevant = []
    for i, value in enumerate(values):
        block = _block(rng, i, f"Project {project}; shard {i + 1}; active units {value}.")
        blocks.append(block); relevant.append(block.block_id)
    for i in range(3, 8):
        blocks.append(_block(rng, i, f"Project {rng.choice([n for n in _NAMES if n != project])}; shard {i}; active units {rng.randrange(2, 30)}."))
    answer = str(sum(values))
    return _finish("aggregation", index, seed, rng, f"Sum active units across all three shards of project {project}.", answer, blocks, [relevant])


def _no_tool(index: int, seed: int, rng: random.Random) -> EpisodeGraph:
    token = f"ACK-{rng.randrange(1000, 9999)}"
    blocks = [_block(rng, i, f"Unrelated archived status record {rng.choice(_VALUES)} for {rng.choice(_NAMES)}.") for i in range(8)]
    return _finish(
        "no_tool", index, seed, rng,
        f"No tool is required. Reply with the literal token {token}.", token, blocks, [[]],
        requires_tool=False,
    )


_BUILDERS: dict[str, Callable[[int, int, random.Random], EpisodeGraph]] = {
    "single_lookup": _single,
    "near_duplicate": _near_duplicate,
    "two_block_join": _two_join,
    "three_block_chain": _three_chain,
    "stale_conflict": _conflict,
    "aggregation": _aggregation,
    "no_tool": _no_tool,
}


def generate_corpus(seed: int = 1729, distribution: dict[str, int] | None = None) -> tuple[EpisodeGraph, ...]:
    distribution = dict(DEFAULT_DISTRIBUTION if distribution is None else distribution)
    unknown = set(distribution).difference(_BUILDERS)
    if unknown:
        raise ValueError(f"unknown diagnostic families: {sorted(unknown)}")
    episodes: list[EpisodeGraph] = []
    for family, count in distribution.items():
        if count < 0:
            raise ValueError(f"negative count for {family}")
        for index in range(count):
            episode_seed = int.from_bytes(hashlib.sha256(f"{seed}:{family}:{index}".encode()).digest()[:8], "big")
            episodes.append(_BUILDERS[family](index, episode_seed, random.Random(episode_seed)))
    return tuple(episodes)


def corpus_manifest(
    episodes: Iterable[EpisodeGraph], tokenizer_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    items = tuple(episodes)
    lines = [canonical_json(episode.to_dict()) for episode in items]
    return {
        "schema_version": 1,
        "generator_version": CORPUS_VERSION,
        "episode_count": len(items),
        "family_counts": dict(sorted(Counter(e.template_family.split("/")[0] for e in items).items())),
        "corpus_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        "tokenizer": tokenizer_identity,
    }


def write_corpus(
    path: str | Path, episodes: Iterable[EpisodeGraph],
    tokenizer_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    items = tuple(episodes)
    with destination.open("w", encoding="utf-8") as handle:
        for episode in items:
            handle.write(canonical_json(episode.to_dict()) + "\n")
    manifest = corpus_manifest(items, tokenizer_identity)
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_corpus(path: str | Path) -> tuple[EpisodeGraph, ...]:
    with Path(path).open(encoding="utf-8") as handle:
        return tuple(EpisodeGraph.from_dict(json.loads(line)) for line in handle if line.strip())
