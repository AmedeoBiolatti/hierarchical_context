from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.synthetic import corpus_manifest, generate_corpus, read_corpus, write_corpus
from .eval.routing import evaluate_selectors
from .packing import ByteTokenizer, HuggingFaceTokenizer
from .reporting import environment_metadata, result_document, write_json
from .routing import (
    DEFAULT_MINILM_REVISION, BM25Selector, DenseSelector, EmbeddingSelector,
    MiniLMEncoder, OracleSelector, RandomSelector,
)


def generate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the deterministic Phase 0 diagnostic corpus")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tokenizer", choices=("byte", "qwen"), default="qwen")
    args = parser.parse_args(argv)
    episodes = generate_corpus(args.seed)
    tokenizer = ByteTokenizer() if args.tokenizer == "byte" else HuggingFaceTokenizer()
    manifest = write_corpus(args.output, episodes, tokenizer.identity)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 0 block-selection baselines")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", choices=("byte", "qwen"), default="byte")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision", default=DEFAULT_MINILM_REVISION)
    parser.add_argument("--budget", type=float, nargs="+", default=(0.10, 0.25, 0.35, 0.50))
    args = parser.parse_args(argv)

    episodes = read_corpus(args.corpus)
    tokenizer = ByteTokenizer() if args.tokenizer == "byte" else HuggingFaceTokenizer()
    selectors = [RandomSelector(), BM25Selector(), OracleSelector(), DenseSelector()]
    identities: dict[str, object] = {
        "random": {"kind": "seeded_random"}, "bm25": {"kind": "local_bm25", "k1": 1.5, "b": 0.75},
        "oracle": {"kind": "gold_support"}, "dense": {"kind": "all_blocks"},
    }
    if not args.skip_embeddings:
        encoder = MiniLMEncoder(args.embedding_model, args.embedding_revision)
        selectors.insert(2, EmbeddingSelector(encoder))
        identities["minilm"] = encoder.identity
    metrics = evaluate_selectors(episodes, selectors, tokenizer, args.budget)
    document = result_document(
        corpus_manifest=corpus_manifest(episodes, tokenizer.identity), tokenizer_identity=tokenizer.identity,
        selector_identities=identities, metrics=metrics,
        configuration={"budgets": args.budget, "skip_embeddings": args.skip_embeddings},
    )
    write_json(args.output, document)


def environment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record the local experiment environment")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_json(args.output, {"result_schema_version": 1, "environment": environment_metadata()})
