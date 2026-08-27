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

## Phase 2 pretrained-model integration

Phase 2 pins Qwen3-0.6B, evaluates a stratified 63-example teacher-forced set,
and compares compact/aligned dense attention with static, random-memory, oracle,
and episode-anchored block topologies. The 4K quality layout and three memory
initialization seeds are fixed in `configs/experiment/phase2.json`.

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_phase2.py
PYTHONPATH=src .venv/bin/python scripts/bench_prefill.py
PYTHONPATH=src .venv/bin/python scripts/check_phase2_acceptance.py
```

The prefill driver starts a fresh process for every mode/length pair and records
model loading, mask/input preparation, first-call latency, warm percentiles,
peak VRAM, and thermal metadata. Phase 2 intentionally stops at teacher forcing:
generation, custom KV caches, training, and Qwen3-1.7B are Phase 3 work.

## Phase 3 oracle-path adaptation

Phase 3 adds a deterministic 10,000-example training corpus (including four
code-reasoning families), frozen BF16 Qwen weights with rank-16 LoRA adapters,
learned memory/router tokens, an oracle-supervised router head, and a sharded
top-256 teacher cache. Training uses three oracle examples followed by one dense
replay example, gradient accumulation of eight, and retention-aware development
checkpointing. The smoke stage deliberately uses the cached 0.6B model before
the decisive 1.7B run.

```bash
# 0.6B smoke/pilot
uv run python scripts/cache_phase3_teacher.py --model pilot \
  --limit 512 --output artifacts/teacher_cache/phase3-pilot
uv run python scripts/train_phase3.py --model pilot \
  --teacher-cache artifacts/teacher_cache/phase3-pilot \
  --limit 32 --max-optimizer-steps 4 --output artifacts/phase3-smoke

# Decisive 1.7B run (10K primary, then 2K confirmation seeds 1730/1731)
uv run python scripts/cache_phase3_teacher.py --model primary \
  --output artifacts/teacher_cache/phase3-primary
uv run python scripts/train_phase3.py --model primary \
  --teacher-cache artifacts/teacher_cache/phase3-primary \
  --output artifacts/phase3-primary
uv run python scripts/eval_phase3.py --model primary \
  --checkpoint artifacts/phase3-primary/checkpoint-01250 \
  --output artifacts/phase3_primary_eval.json
```

The primary acceptance decision also requires two 2,000-example confirmation
runs with seeds 1730 and 1731. Phase 3 remains teacher-forced and uses oracle
selection for the answer path; predicted routing is intentionally deferred to
Phase 4.
