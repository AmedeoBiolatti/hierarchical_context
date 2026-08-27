from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import random
from typing import Any, Iterable, Mapping, Sequence

from .phase2 import PLACEMENTS, Phase2Example, _transform, build_phase2_corpus
from .synthetic import generate_corpus
from ..packing import Tokenizer
from ..schema import Anchor, EpisodeGraph, ToolBlock, canonical_json


PHASE3_CORPUS_VERSION = 1
TRAIN_COUNTS: dict[str, int] = {
    "single_lookup": 1500, "near_duplicate": 1500, "two_block_join": 1500,
    "three_block_chain": 1250, "stale_conflict": 1250, "aggregation": 1000,
    "no_tool": 500, "code_definition": 500, "code_call_chain": 400,
    "code_stack_trace": 300, "code_injected_bug": 300,
}
CODE_FAMILIES = ("code_definition", "code_call_chain", "code_stack_trace", "code_injected_bug")
ALL_FAMILIES = tuple(TRAIN_COUNTS)


@dataclass(frozen=True, slots=True)
class Phase3Corpus:
    train: tuple[Phase2Example, ...]
    development: tuple[Phase2Example, ...]
    test: tuple[Phase2Example, ...]
    manifest: Mapping[str, Any]


def _code_block(rng: random.Random, index: int, content: str) -> ToolBlock:
    opaque = hashlib.sha256(f"phase3:{rng.random()}:{index}".encode()).hexdigest()[:12]
    return ToolBlock(
        block_id=f"src_{opaque}", content=f"{content}\n# artifact {opaque}", tool_type="code",
        invariant_metadata={"path": f"module_{rng.randrange(1000, 9999)}.py", "language": "python"},
        version=f"r{rng.randrange(1, 30)}",
    )


def _code_episode(family: str, index: int, seed: int, split: str) -> EpisodeGraph:
    rng = random.Random(seed); value = f"K{rng.randrange(100, 999)}"
    name = f"fn_{rng.randrange(1000, 9999)}"; helper = f"helper_{rng.randrange(1000, 9999)}"
    blocks: list[ToolBlock] = []; supports: list[str] = []
    if family == "code_definition":
        relevant = _code_block(rng, 0, f'def {name}():\n    return "{value}"')
        blocks.append(relevant); supports = [relevant.block_id]
        task = (f"What literal does {name} return?" if split != "test"
                else f"Emit RETURN_LITERAL for SYMBOL={name}.")
        answer = value
    elif family == "code_call_chain":
        entry = _code_block(rng, 0, f"def {name}():\n    return {helper}()")
        leaf = _code_block(rng, 1, f'def {helper}():\n    return "{value}"')
        blocks.extend((entry, leaf)); supports = [entry.block_id, leaf.block_id]
        task = (f"Follow calls from {name} and return the final literal." if split != "test"
                else f"Resolve CALL_TARGET transitively from ENTRY={name}; emit the terminal literal.")
        answer = value
    elif family == "code_stack_trace":
        line = rng.randrange(20, 80); exception = f"Fault{rng.randrange(10, 99)}"
        trace = _code_block(rng, 0, f'Traceback: file worker.py line {line}, in {name}\n{exception}: failed')
        source = _code_block(rng, 1, f"def {name}():\n    # line {line}\n    raise {exception}('failed')")
        blocks.extend((trace, source)); supports = [trace.block_id, source.block_id]
        task = ("Name the exception raised at the traced source line." if split != "test"
                else "Join TRACE frame to SOURCE line and emit EXCEPTION_CLASS.")
        answer = exception
    elif family == "code_injected_bug":
        expected = rng.randrange(2, 9); wrong = expected + 1
        spec = _code_block(rng, 0, f"Contract: {name}(x) returns x + {expected}.")
        implementation = _code_block(rng, 1, f"def {name}(x):\n    return x + {wrong}")
        blocks.extend((spec, implementation)); supports = [spec.block_id, implementation.block_id]
        task = (f"What integer should replace {wrong} to repair {name}?" if split != "test"
                else f"Compare CONTRACT and IMPLEMENTATION for SYMBOL={name}; emit REPLACEMENT_INTEGER.")
        answer = str(expected)
    else:
        raise ValueError(f"unknown code family {family}")
    for distractor in range(len(blocks), 8):
        other = f"fn_{rng.randrange(1000, 9999)}"
        blocks.append(_code_block(rng, distractor, f"def {other}():\n    return {rng.randrange(10, 99)}"))
    rng.shuffle(blocks)
    return EpisodeGraph(
        episode_id=f"phase3-{split}-{family}-{index:05d}",
        anchor=Anchor(task=task, system_instruction="Return only the exact requested value."),
        tool_blocks=tuple(blocks), expected_answer=answer,
        acceptable_support_sets=(frozenset(supports),), requires_tool=True,
        template_family=f"{family}/{split}", seed=seed,
        metadata={"split": split, "template_variant": f"{family}/{split}_v1",
                  "generator_version": PHASE3_CORPUS_VERSION},
    )


def _record_split(counts: Mapping[str, int], root_seed: int, split: str) -> list[EpisodeGraph]:
    result: list[EpisodeGraph] = []
    for family, wanted in counts.items():
        generated = generate_corpus(root_seed, {family: max(wanted * 2, 10)})
        candidates = [item for item in generated if item.metadata["split"] == "development"]
        if len(candidates) < wanted:
            raise ValueError(f"not enough generated {family} examples")
        for ordinal, base in enumerate(candidates[:wanted]):
            metadata = dict(base.metadata); metadata.update({"split": split, "phase3_source_id": base.episode_id})
            blocks = tuple(ToolBlock(
                block_id=block.block_id,
                content=f"{block.content}\nRecord nonce: {hashlib.sha256(f'{root_seed}:{family}:{ordinal}:{block.block_id}'.encode()).hexdigest()[:12]}",
                tool_type=block.tool_type, invariant_metadata=block.invariant_metadata, version=block.version,
            ) for block in base.tool_blocks)
            result.append(EpisodeGraph(
                episode_id=f"phase3-{split}-{family}-{ordinal:05d}", anchor=base.anchor,
                tool_blocks=blocks, expected_answer=base.expected_answer,
                acceptable_support_sets=base.acceptable_support_sets, requires_tool=base.requires_tool,
                template_family=f"{family}/{split}", seed=base.seed, metadata=metadata,
            ))
    return result


def _place(episodes: Sequence[EpisodeGraph], tokenizer: Tokenizer) -> tuple[Phase2Example, ...]:
    return tuple(_transform(episode, PLACEMENTS[index % len(PLACEMENTS)], tokenizer)
                 for index, episode in enumerate(episodes))


def _code_split(count_per_family: int | Mapping[str, int], root_seed: int, split: str) -> list[EpisodeGraph]:
    result = []
    for family_index, family in enumerate(CODE_FAMILIES):
        count = count_per_family if isinstance(count_per_family, int) else count_per_family[family]
        for index in range(count):
            seed = int.from_bytes(hashlib.sha256(
                f"{root_seed}:{family}:{index}".encode()).digest()[:8], "big")
            result.append(_code_episode(family, index, seed, split))
    return result


def _manifest(train: Sequence[Phase2Example], development: Sequence[Phase2Example],
              test: Sequence[Phase2Example], tokenizer: Tokenizer) -> dict[str, Any]:
    splits = {"train": train, "development": development, "test": test}
    records = {name: [item.episode.to_dict() for item in values] for name, values in splits.items()}
    hashes = {name: hashlib.sha256(
        ("\n".join(canonical_json(item) for item in values) + "\n").encode()).hexdigest()
        for name, values in records.items()}
    content_sets = {
        name: {block.content_hash for item in values for block in item.episode.tool_blocks}
        for name, values in splits.items()
    }
    overlap = {
        f"{left}_{right}": len(content_sets[left] & content_sets[right])
        for left, right in (("train", "development"), ("train", "test"), ("development", "test"))
    }
    if any(overlap.values()):
        raise ValueError(f"Phase 3 corpus content leakage detected: {overlap}")
    return {
        "phase3_corpus_version": PHASE3_CORPUS_VERSION, "tokenizer": dict(tokenizer.identity),
        "counts": {name: len(values) for name, values in splits.items()},
        "family_counts": {name: dict(sorted(Counter(
            item.episode.template_family.split("/")[0] for item in values).items()))
            for name, values in splits.items()},
        "sha256": hashes, "content_hash_overlap": overlap,
    }


def build_phase3_corpus(tokenizer: Tokenizer) -> Phase3Corpus:
    record_counts = {key: value for key, value in TRAIN_COUNTS.items() if key not in CODE_FAMILIES}
    train_episodes = _record_split(record_counts, 2718, "train")
    train_episodes += _code_split({key: TRAIN_COUNTS[key] for key in CODE_FAMILIES}, 2718, "train")
    development_episodes = _record_split({key: 20 for key in record_counts}, 31415, "development")
    development_episodes += _code_split(20, 31415, "development")
    train = _place(train_episodes, tokenizer); development = _place(development_episodes, tokenizer)

    frozen = list(build_phase2_corpus(generate_corpus(seed=1729), tokenizer, examples_per_family=3))
    code_test: list[Phase2Example] = []
    for episode in _code_split(3, 16180, "test"):
        code_test.extend(_transform(episode, placement, tokenizer) for placement in PLACEMENTS)
    test = tuple(frozen + code_test)
    return Phase3Corpus(train, development, test, _manifest(train, development, test, tokenizer))


def stratified_subset(examples: Sequence[Phase2Example], size: int, seed: int) -> tuple[Phase2Example, ...]:
    if size > len(examples):
        raise ValueError("subset is larger than its source")
    groups: dict[str, list[Phase2Example]] = {}
    for item in examples:
        groups.setdefault(item.episode.template_family.split("/")[0], []).append(item)
    rng = random.Random(seed); result: list[Phase2Example] = []
    while len(result) < size:
        progressed = False
        for family in sorted(groups):
            if not groups[family] or len(result) == size:
                continue
            index = rng.randrange(len(groups[family])); result.append(groups[family].pop(index)); progressed = True
        if not progressed:
            break
    return tuple(result)
