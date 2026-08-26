# Tool Context Attention

Phase 0 of the hierarchical tool-context attention experiment. This repository
defines backend-neutral episode schemas, deterministic fixed-shape packing, a
dense reference attention mask, synthetic diagnostics, retrieval baselines, and
machine-readable evaluation reports.

## Development

```bash
uv venv --python 3.11
uv pip install -e '.[dev,tokenizers,retrieval]'
pytest
```

The dependency lock is intentionally deferred until the Phase 1 CUDA smoke
test. Qwen model weights are not required for Phase 0. Corpus manifests and the
canonical evaluation use the Qwen3-0.6B tokenizer; deterministic offline tests
use the built-in byte tokenizer.

```bash
tca-generate --output artifacts/corpora/phase0.jsonl
tca-evaluate --corpus artifacts/corpora/phase0.jsonl --output artifacts/phase0_metrics.json --tokenizer qwen
tca-environment --output artifacts/environment.json
```
