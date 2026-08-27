from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as functional

from ..data.phase2 import Phase2Example, build_phase2_corpus, phase2_manifest
from ..data.synthetic import generate_corpus
from ..modeling.qwen3_block import (
    MemoryRouterTokenBank, Phase2Mode, assert_phase2_model, prepare_model_input,
    set_model_attention,
)
from ..packing import HuggingFaceTokenizer, TeacherForcedEpisode, TokenRole, pack_teacher_forcing_episode
from ..reporting import environment_metadata, write_json
from ..schema import LayoutSpec


FIXED_MODES = (Phase2Mode.DENSE_COMPACT, Phase2Mode.DENSE_ALIGNED, Phase2Mode.BLOCK_STATIC)
SEEDED_MODES = (Phase2Mode.BLOCK_MEMORY, Phase2Mode.BLOCK_ORACLE, Phase2Mode.BLOCK_DYNAMIC_ANCHOR)


def load_phase2_model(config: Mapping[str, Any]) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=None,
    )
    model = model.to("cuda").eval().requires_grad_(False)
    assert_phase2_model(model)
    return model


def phase2_layout(config: Mapping[str, Any]) -> LayoutSpec:
    return LayoutSpec(**config["layout"])


def _support(example: Phase2Example) -> set[str]:
    groups = example.episode.acceptable_support_sets
    return set(min(groups, key=lambda group: (len(group), sorted(group)))) if groups else set()


def _teacher(
    example: Phase2Example, layout: LayoutSpec, tokenizer: HuggingFaceTokenizer, mode: Phase2Mode,
) -> TeacherForcedEpisode:
    memory = mode.uses_memory
    selected: Iterable[str] | None = _support(example) if mode == Phase2Mode.BLOCK_ORACLE else None
    return pack_teacher_forcing_episode(
        example.episode, layout, tokenizer, selected_blocks=selected,
        placeholder_token_id=int(tokenizer._tokenizer.pad_token_id or 0),
        enable_memory_tokens=memory, enable_router_tokens=memory,
    )


def _metrics(logits: torch.Tensor, prepared: Any) -> dict[str, Any]:
    logits = logits.float()
    targets = prepared.target_token_ids
    loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
    predicted = logits.argmax(dim=-1)
    correct = predicted.eq(targets)
    return {
        "target_tokens": int(targets.numel()), "nll_sum": float(loss.item()),
        "nll_per_token": float(loss.item() / targets.numel()),
        "token_accuracy": float(correct.float().mean().item()),
        "exact_match": bool(correct.all().item()),
    }


@torch.inference_mode()
def evaluate_one(model: torch.nn.Module, teacher: TeacherForcedEpisode, mode: Phase2Mode,
                 bank: MemoryRouterTokenBank | None) -> dict[str, Any]:
    set_model_attention(model, mode)
    prepared = prepare_model_input(model, teacher, mode, token_bank=bank)
    result = model(**prepared.model_kwargs)
    return _metrics(result.logits, prepared)


@torch.inference_mode()
def model_correctness_probe(
    model: torch.nn.Module, dense_teacher: TeacherForcedEpisode,
    static_teacher: TeacherForcedEpisode, memory_teacher: TeacherForcedEpisode,
) -> dict[str, Any]:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Identical token and embedding inputs must preserve the untouched dense path.
    set_model_attention(model, Phase2Mode.DENSE_ALIGNED)
    dense = prepare_model_input(model, dense_teacher, Phase2Mode.DENSE_ALIGNED)
    ids_kwargs = dict(dense.model_kwargs)
    embeds = ids_kwargs.pop("inputs_embeds")
    ids_kwargs["input_ids"] = torch.tensor(
        dense_teacher.packed.input_ids, dtype=torch.long, device=embeds.device,
    ).unsqueeze(0)
    by_ids = model(**ids_kwargs).logits.float()
    by_embeds = model(**dense.model_kwargs).logits.float()
    embedding_error = float((by_ids - by_embeds).abs().max().item())

    # The native Transformers flex implementation should reproduce causal SDPA.
    compact = prepare_model_input(model, dense_teacher, Phase2Mode.DENSE_COMPACT)
    set_model_attention(model, Phase2Mode.DENSE_COMPACT)
    sdpa_logits = model(**compact.model_kwargs).logits.float()
    set_model_attention(model, Phase2Mode.BLOCK_STATIC)
    flex_kwargs = dict(compact.model_kwargs)
    flex_logits = model(**flex_kwargs).logits.float()
    sdpa_nll = _metrics(sdpa_logits, compact)["nll_per_token"]
    flex_nll = _metrics(flex_logits, compact)["nll_per_token"]
    cosine = float(functional.cosine_similarity(sdpa_logits, flex_logits, dim=-1).mean().item())
    top1 = float(sdpa_logits.argmax(-1).eq(flex_logits.argmax(-1)).float().mean().item())

    # Perturb one tool and inspect every hidden-state boundary of another tool.
    block = prepare_model_input(model, static_teacher, Phase2Mode.BLOCK_STATIC)
    kwargs = dict(block.model_kwargs); kwargs["output_hidden_states"] = True
    mask_ids: list[int] = []
    hooks = []
    for layer in base_model.model.layers:
        def record_mask(_module: Any, _args: Any, call_kwargs: dict[str, Any]) -> None:
            mask_ids.append(id(call_kwargs.get("attention_mask")))
        hooks.append(layer.self_attn.register_forward_pre_hook(record_mask, with_kwargs=True))
    base = model(**kwargs)
    for hook in hooks:
        hook.remove()
    roles = static_teacher.packed.token_role; blocks = static_teacher.packed.block_id
    tool_ids = list(dict.fromkeys(item for role, item in zip(roles, blocks, strict=True)
                                  if role == TokenRole.T and item is not None))
    source = tool_ids[0]; changed = tool_ids[1]
    source_positions = torch.tensor(
        [i for i, (role, item) in enumerate(zip(roles, blocks, strict=True))
         if role == TokenRole.T and item == source], device=embeds.device,
    )
    changed_positions = torch.tensor(
        [i for i, (role, item) in enumerate(zip(roles, blocks, strict=True))
         if role == TokenRole.T and item == changed], device=embeds.device,
    )
    perturbed_kwargs = dict(kwargs)
    perturbed = kwargs["inputs_embeds"].clone()
    generator = torch.Generator(device="cuda").manual_seed(9917)
    noise = torch.randn(perturbed[:, changed_positions].shape, generator=generator,
                        device=perturbed.device, dtype=perturbed.dtype)
    perturbed[:, changed_positions] = noise
    perturbed_kwargs["inputs_embeds"] = perturbed
    changed_result = model(**perturbed_kwargs)
    layer_errors = [
        float((left.index_select(1, source_positions) - right.index_select(1, source_positions)).abs().max().item())
        for left, right in zip(base.hidden_states, changed_result.hidden_states, strict=True)
    ]

    bank = MemoryRouterTokenBank(
        model.config.hidden_size, memory_tokens=memory_teacher.packed.layout.memory_tokens_per_block,
        router_tokens=memory_teacher.packed.layout.router_capacity, seed=1729,
    ).to(embeds.device)
    memory_ids = torch.tensor(memory_teacher.packed.input_ids, dtype=torch.long, device=embeds.device).unsqueeze(0)
    memory_base = model.get_input_embeddings()(memory_ids)
    banked = bank.apply(memory_base, memory_teacher)
    memory_roles = memory_teacher.packed.token_role
    mr = torch.tensor([role in (TokenRole.M, TokenRole.R) for role in memory_roles], device=embeds.device)
    unchanged = bool(torch.equal(banked[:, ~mr], memory_base[:, ~mr]))
    overwritten = bool(mr.any() and not torch.equal(banked[:, mr], memory_base[:, mr]))
    padding_positions = torch.tensor(
        [i for i, valid in enumerate(static_teacher.packed.valid_token) if not valid][:8], device=embeds.device,
    )
    padding_kwargs = dict(kwargs); padding_kwargs.pop("output_hidden_states")
    padding_kwargs["logits_to_keep"] = padding_positions
    padding_finite = bool(torch.isfinite(model(**padding_kwargs).logits).all().item())
    return {
        "dense_input_ids_vs_inputs_embeds_max_abs_error": embedding_error,
        "causal_flex_vs_sdpa": {
            "nll_difference": abs(float(flex_nll - sdpa_nll)), "logit_cosine": cosine,
            "top1_agreement": top1,
        },
        "cross_block_layer_max_abs_errors": layer_errors,
        "cross_block_max_abs_error": max(layer_errors),
        "same_block_mask_object_all_layers": len(mask_ids) == len(base_model.model.layers) and len(set(mask_ids)) == 1,
        "token_bank_only_overwrites_memory_router": unchanged and overwritten,
        "padding_logits_finite": padding_finite,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["seed"], row["family"], row["placement"])].append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        tokens = sum(row["target_tokens"] for row in members)
        output.append({
            "mode": key[0], "seed": key[1], "family": key[2], "placement": key[3],
            "episodes": len(members), "target_tokens": tokens,
            "nll_per_token": sum(row["nll_sum"] for row in members) / tokens,
            "token_accuracy": sum(row["token_accuracy"] * row["target_tokens"] for row in members) / tokens,
            "exact_match": sum(float(row["exact_match"]) for row in members) / len(members),
        })
    return output


def run_phase2_quality(config: Mapping[str, Any], output: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 evaluation requires CUDA")
    tokenizer = HuggingFaceTokenizer(config["model_id"], config["revision"])
    examples = list(build_phase2_corpus(
        generate_corpus(seed=int(config["corpus_seed"])), tokenizer,
        examples_per_family=int(config["examples_per_family"]),
    ))
    if limit is not None:
        examples = examples[:limit]
    layout = phase2_layout(config); model = load_phase2_model(config)
    banks = {
        seed: MemoryRouterTokenBank(
            model.config.hidden_size, memory_tokens=layout.memory_tokens_per_block,
            router_tokens=layout.router_capacity, initializer_range=float(model.config.initializer_range), seed=seed,
        ).to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        for seed in config["memory_seeds"]
    }
    dense_probe = _teacher(examples[0], layout, tokenizer, Phase2Mode.DENSE_ALIGNED)
    static_probe = _teacher(examples[0], layout, tokenizer, Phase2Mode.BLOCK_STATIC)
    memory_probe = _teacher(examples[0], layout, tokenizer, Phase2Mode.BLOCK_MEMORY)
    report: dict[str, Any] = {
        "result_schema_version": 1, "phase": 2, "kind": "quality",
        "environment": environment_metadata(), "configuration": dict(config),
        "model": {"id": config["model_id"], "revision": config["revision"]},
        "corpus": phase2_manifest(examples, tokenizer.identity),
        "memory_token_checksums": {str(seed): bank.checksums() for seed, bank in banks.items()},
        "correctness": model_correctness_probe(model, dense_probe, static_probe, memory_probe),
        "rows": [], "aggregates": [],
    }
    write_json(output, report)
    dense_baselines: dict[tuple[str, str], dict[str, float]] = {}
    for index, example in enumerate(examples):
        family = example.episode.template_family.split("/")[0]
        specs = [(mode, None) for mode in FIXED_MODES]
        specs += [(mode, seed) for mode in SEEDED_MODES for seed in config["memory_seeds"]]
        episode_rows = []
        for mode, seed in specs:
            teacher = _teacher(example, layout, tokenizer, mode)
            values = evaluate_one(model, teacher, mode, banks.get(seed))
            row = {"episode_id": example.episode.episode_id, "family": family,
                   "placement": example.placement, "mode": mode.value, "seed": seed, **values}
            episode_rows.append(row)
            if mode in (Phase2Mode.DENSE_COMPACT, Phase2Mode.DENSE_ALIGNED):
                dense_baselines[(example.episode.episode_id, mode.value)] = values
        for row in episode_rows:
            identity = row["episode_id"]
            for baseline in (Phase2Mode.DENSE_COMPACT, Phase2Mode.DENSE_ALIGNED):
                reference = dense_baselines[(identity, baseline.value)]
                row[f"nll_delta_vs_{baseline.value}"] = row["nll_per_token"] - reference["nll_per_token"]
                row[f"token_accuracy_delta_vs_{baseline.value}"] = row["token_accuracy"] - reference["token_accuracy"]
                row[f"exact_match_delta_vs_{baseline.value}"] = float(row["exact_match"]) - float(reference["exact_match"])
        report["rows"].extend(episode_rows)
        report["progress"] = {"completed_examples": index + 1, "total_examples": len(examples)}
        write_json(output, report)
        print(f"phase2 quality: {index + 1}/{len(examples)} examples", flush=True)
    report["aggregates"] = _aggregate(report["rows"])
    report.pop("progress", None); write_json(output, report)
    return report
