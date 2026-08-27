from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file

from ..data.phase2 import Phase2Example
from ..data.phase3 import build_phase3_corpus
from .quality import model_correctness_probe
from ..modeling.phase3 import OracleAdaptationModel, OracleRouterHead, build_oracle_adaptation_model
from ..modeling.qwen3_block import MemoryRouterTokenBank, Phase2Mode, prepare_model_input, set_model_attention
from ..packing import HuggingFaceTokenizer
from ..reporting import environment_metadata, write_json
from ..schema import LayoutSpec
from ..training.losses import router_labels
from ..training.phase3 import load_base_model, pack_training_example


MULTIHOP_FAMILIES = {
    "two_block_join", "three_block_chain", "aggregation", "code_call_chain",
    "code_stack_trace", "code_injected_bug",
}


def load_adaptation_checkpoint(config: Mapping[str, Any], model_name: str,
                               checkpoint: str | Path) -> OracleAdaptationModel:
    from peft import PeftModel

    checkpoint = Path(checkpoint); manifest = json.loads((checkpoint / "manifest.json").read_text())
    base = load_base_model(config["models"][model_name], training=False)
    language_model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
    hidden = int(base.config.hidden_size)
    bank = MemoryRouterTokenBank(hidden, memory_tokens=8, router_tokens=8, seed=int(manifest["seed"]))
    router = OracleRouterHead(hidden); tensors = load_file(checkpoint / "auxiliary.safetensors")
    bank.load_state_dict({key.removeprefix("token_bank."): value for key, value in tensors.items()
                          if key.startswith("token_bank.")})
    router.load_state_dict({key.removeprefix("router_head."): value for key, value in tensors.items()
                            if key.startswith("router_head.")})
    return OracleAdaptationModel(language_model, bank, router).to("cuda").eval().requires_grad_(False)


def _metric_row(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    logits = logits.float(); loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum",
    )
    correct = logits.argmax(-1).eq(targets)
    return {"target_tokens": int(targets.numel()), "nll_sum": float(loss),
            "token_accuracy": float(correct.float().mean()), "exact_match": bool(correct.all())}


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(row["target_tokens"] for row in rows)
    return {"episodes": len(rows), "target_tokens": tokens,
            "nll_per_token": sum(row["nll_sum"] for row in rows) / tokens,
            "token_accuracy": sum(row["token_accuracy"] * row["target_tokens"] for row in rows) / tokens,
            "exact_match": sum(float(row["exact_match"]) for row in rows) / len(rows)}


@torch.inference_mode()
def evaluate_language_model(model: torch.nn.Module, examples: Sequence[Phase2Example], layout: LayoutSpec,
                            tokenizer: HuggingFaceTokenizer, mode: Phase2Mode,
                            adaptation: OracleAdaptationModel | None = None) -> dict[str, Any]:
    rows = []; router_rows = []
    if adaptation is None:
        set_model_attention(model, mode)
    for index, example in enumerate(examples):
        teacher = pack_training_example(example, layout, tokenizer, mode)
        if adaptation is None:
            prepared = prepare_model_input(model, teacher, mode)
            logits = model(**prepared.model_kwargs).logits; targets = prepared.target_token_ids
            output = None
        else:
            output = adaptation(teacher, mode); logits = output.logits; targets = output.prepared.target_token_ids
        family = example.episode.template_family.split("/")[0]
        row = {"episode_id": example.episode.episode_id, "family": family,
               "placement": example.placement, **_metric_row(logits, targets)}
        rows.append(row)
        if output is not None and output.block_logits is not None:
            labels, no_tool = router_labels(teacher, output.block_logits.device)
            ranking = output.block_logits[0].argsort(descending=True)
            positives = set(torch.nonzero(labels[0], as_tuple=False).flatten().tolist())
            router_rows.append({
                "episode_id": example.episode.episode_id, "family": family,
                "support_recall_at_25pct": float(positives.issubset(set(ranking[:2].tolist()))),
                "support_recall_at_37_5pct": float(positives.issubset(set(ranking[:3].tolist()))),
                "no_tool_correct": float((output.no_tool_logit.sigmoid() >= 0.5).eq(no_tool.bool()).item()),
            })
        if (index + 1) % 10 == 0 or index + 1 == len(examples):
            print(f"phase3 eval {mode.value}: {index + 1}/{len(examples)}", flush=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    result = {"mode": mode.value, "aggregate": _aggregate(rows), "families": {
        family: _aggregate(values) for family, values in sorted(groups.items())}, "rows": rows}
    if router_rows:
        result["router"] = {
            "support_recall_at_25pct": sum(row["support_recall_at_25pct"] for row in router_rows) / len(router_rows),
            "support_recall_at_37_5pct": sum(row["support_recall_at_37_5pct"] for row in router_rows) / len(router_rows),
            "no_tool_accuracy": sum(row["no_tool_correct"] for row in router_rows) / len(router_rows),
            "rows": router_rows,
        }
    return result


def run_phase3_evaluation(config: Mapping[str, Any], model_name: str, checkpoint: str | Path,
                          output: str | Path) -> dict[str, Any]:
    model_config = config["models"][model_name]
    tokenizer = HuggingFaceTokenizer(model_config["model_id"], model_config["revision"])
    corpus = build_phase3_corpus(tokenizer); examples = corpus.test; layout = LayoutSpec(**config["layout"])
    base = load_base_model(model_config, training=False)
    frozen_dense = evaluate_language_model(base, examples, layout, tokenizer, Phase2Mode.DENSE_ALIGNED)
    del base; torch.cuda.empty_cache()

    initial_base = load_base_model(model_config, training=False)
    manifest = json.loads((Path(checkpoint) / "manifest.json").read_text())
    initial = build_oracle_adaptation_model(initial_base, seed=int(manifest["seed"])).to("cuda").eval().requires_grad_(False)
    initial_oracle = evaluate_language_model(initial.language_model, examples, layout, tokenizer,
                                             Phase2Mode.BLOCK_ORACLE, initial)
    del initial; torch.cuda.empty_cache()

    adapted = load_adaptation_checkpoint(config, model_name, checkpoint)
    adapted_dense = evaluate_language_model(adapted.language_model, examples, layout, tokenizer,
                                            Phase2Mode.DENSE_ALIGNED, adapted)
    adapted_oracle = evaluate_language_model(adapted.language_model, examples, layout, tokenizer,
                                             Phase2Mode.BLOCK_ORACLE, adapted)
    probe_example = examples[0]
    correctness = model_correctness_probe(
        adapted.language_model,
        pack_training_example(probe_example, layout, tokenizer, Phase2Mode.DENSE_ALIGNED),
        pack_training_example(probe_example, layout, tokenizer, Phase2Mode.BLOCK_STATIC),
        pack_training_example(probe_example, layout, tokenizer, Phase2Mode.BLOCK_ORACLE),
    )
    report = {
        "result_schema_version": 1, "phase": 3, "kind": "evaluation", "environment": environment_metadata(),
        "model": dict(model_config), "checkpoint": str(checkpoint), "corpus": dict(corpus.manifest),
        "frozen_dense": frozen_dense, "initial_oracle": initial_oracle,
        "adapted_dense": adapted_dense, "adapted_oracle": adapted_oracle,
        "correctness": correctness,
    }
    write_json(output, report); return report
