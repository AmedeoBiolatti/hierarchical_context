from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Any, Protocol, Sequence

from .schema import EpisodeGraph, ToolBlock, canonical_json


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True, slots=True)
class RankedBlock:
    block_id: str
    score: float


class BlockSelector(Protocol):
    name: str
    ignores_budget: bool

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]: ...


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class RandomSelector:
    name = "random"
    ignores_budget = False

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        seed = int.from_bytes(hashlib.sha256(f"{episode.seed}:random".encode()).digest()[:8], "big")
        rng = random.Random(seed); ids = [block.block_id for block in episode.tool_blocks]; rng.shuffle(ids)
        return tuple(RankedBlock(block_id, float(len(ids) - i)) for i, block_id in enumerate(ids))


class MetadataProbeSelector:
    """Leakage probe restricted to opaque IDs and invariant block metadata."""

    name = "metadata_probe"
    ignores_budget = False

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        ranked = []
        for block in episode.tool_blocks:
            payload = canonical_json({
                "block_id": block.block_id,
                "tool_type": block.tool_type,
                "metadata": dict(block.invariant_metadata),
                "version": block.version,
            })
            score = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
            ranked.append(RankedBlock(block.block_id, float(score)))
        return tuple(sorted(ranked, key=lambda item: (-item.score, item.block_id)))


class SurfaceProbeSelector:
    """Leakage probe using only content length and punctuation counts."""

    name = "surface_probe"
    ignores_budget = False

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        ranked = []
        for block in episode.tool_blocks:
            text = block.content
            features = (len(text), text.count("|"), text.count("="), text.count(";"), text.count("."))
            payload = canonical_json(features)
            tie_break = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") / 2**32
            score = float(sum((index + 1) * value for index, value in enumerate(features))) + tie_break
            ranked.append(RankedBlock(block.block_id, score))
        return tuple(sorted(ranked, key=lambda item: (-item.score, item.block_id)))


class BM25Selector:
    name = "bm25"
    ignores_budget = False

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1; self.b = b

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        documents = [_tokens(block.content) for block in episode.tool_blocks]
        query = _tokens(f"{episode.anchor.task} {episode.anchor.episode_tool_anchor}")
        average_length = sum(map(len, documents)) / max(len(documents), 1)
        document_frequency = {term: sum(term in doc for doc in documents) for term in set(query)}
        scores = []
        for block, document in zip(episode.tool_blocks, documents, strict=True):
            score = 0.0
            for term in query:
                frequency = document.count(term)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * len(document) / average_length)
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append(RankedBlock(block.block_id, score))
        return tuple(sorted(scores, key=lambda item: (-item.score, item.block_id)))


class SentenceEncoder(Protocol):
    identity: dict[str, Any]

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class MiniLMEncoder:
    def __init__(
        self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: str | None = DEFAULT_MINILM_REVISION,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'retrieval' extra to use MiniLM retrieval") from exc
        self._model = SentenceTransformer(model_id, revision=revision)
        resolved = revision
        if resolved is None:
            modules = getattr(self._model, "_modules", {})
            first = next(iter(modules.values()), None)
            auto_model = getattr(first, "auto_model", None)
            resolved = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
        self.identity = {"kind": "sentence_transformers", "model_id": model_id, "revision": resolved}

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


class EmbeddingSelector:
    name = "minilm"
    ignores_budget = False

    def __init__(self, encoder: SentenceEncoder) -> None:
        self.encoder = encoder

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        query = f"{episode.anchor.task} {episode.anchor.episode_tool_anchor}".strip()
        vectors = self.encoder.encode([query] + [block.content for block in episode.tool_blocks])
        query_vector = vectors[0]
        scores = []
        for block, vector in zip(episode.tool_blocks, vectors[1:], strict=True):
            score = sum(float(a) * float(b) for a, b in zip(query_vector, vector, strict=True))
            scores.append(RankedBlock(block.block_id, score))
        return tuple(sorted(scores, key=lambda item: (-item.score, item.block_id)))


class OracleSelector:
    name = "oracle"
    ignores_budget = True

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        support = min(episode.acceptable_support_sets, key=lambda item: (len(item), sorted(item)))
        return tuple(RankedBlock(block_id, 1.0) for block_id in sorted(support))


class DenseSelector:
    name = "dense"
    ignores_budget = True

    def rank(self, episode: EpisodeGraph) -> tuple[RankedBlock, ...]:
        return tuple(RankedBlock(block.block_id, 1.0) for block in episode.tool_blocks)


def block_token_counts(episode: EpisodeGraph, tokenizer: Any) -> dict[str, int]:
    return {block.block_id: max(1, len(tokenizer.encode(block.content))) for block in episode.tool_blocks}


def select_under_budget(
    episode: EpisodeGraph, ranking: Sequence[RankedBlock], budget_fraction: float,
    tokenizer: Any, *, ignores_budget: bool = False,
) -> tuple[str, ...]:
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if isinstance(ranking, tuple):
        ranked = ranking
    else:
        ranked = tuple(ranking)
    if ignores_budget:
        return tuple(item.block_id for item in ranked)
    counts = block_token_counts(episode, tokenizer)
    limit = math.floor(sum(counts.values()) * budget_fraction)
    selected: list[str] = []; used = 0
    for item in ranked:
        size = counts[item.block_id]
        if used + size <= limit:
            selected.append(item.block_id); used += size
    return tuple(selected)
