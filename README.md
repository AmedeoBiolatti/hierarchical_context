# Tool Context Attention

Hierarchical tool-context attention experiment. Phase 0 defines backend-neutral
episode schemas, deterministic fixed-shape packing, a dense reference mask,
synthetic diagnostics, and retrieval baselines. Phase 1 adds a compiled
FlexAttention `BlockMask`, dense and segmented comparison paths, CUDA correctness
tests, and machine-readable kernel benchmarks.

## Development

```bash
uv sync --extra dev --extra tokenizers --extra retrieval --extra kernel
pytest
```

The committed lock records the CUDA environment proven by the Phase 1 smoke
test. Qwen model weights are not required for Phases 0–1. Corpus manifests and
the canonical evaluation use the Qwen3-0.6B tokenizer; deterministic offline
tests use the built-in byte tokenizer.

```bash
tca-generate --output artifacts/corpora/phase0.jsonl
tca-evaluate --corpus artifacts/corpora/phase0.jsonl --output artifacts/phase0_metrics.json --tokenizer qwen
tca-environment --output artifacts/environment.json
```

## Phase 1 kernel benchmark

Run on the configured RTX 4080 Laptop GPU in performance mode:

```bash
tca-bench-attention --config configs/experiment/phase1.json --output artifacts/phase1_attention.json
python scripts/check_phase1_acceptance.py --report artifacts/phase1_attention.json
```

Compilation, mask construction, execution percentiles, active tile fraction,
peak VRAM, thermal state, OOM boundaries, and the go/no-go decision are stored
in the result JSON. The Triton wrapper deliberately leaves
`ROWS_GUARANTEED_SAFE` disabled because PyTorch 2.13 produces NaNs for mixed
valid/padding query tiles when that optimization is enabled.
