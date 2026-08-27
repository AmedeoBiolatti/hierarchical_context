from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from .distillation import TeacherCache, TeacherCacheWriter, compress_teacher_logits
from .losses import phase3_loss
from ..data.phase2 import Phase2Example
from ..data.phase3 import Phase3Corpus, build_phase3_corpus, stratified_subset
from ..eval.quality import phase2_layout
from ..modeling.phase3 import OracleAdaptationModel, build_oracle_adaptation_model, trainable_parameter_audit
from ..modeling.phase3 import OracleRouterHead
from ..modeling.qwen3_block import MemoryRouterTokenBank
from ..modeling.qwen3_block import Phase2Mode, prepare_model_input, set_model_attention
from ..packing import HuggingFaceTokenizer, pack_teacher_forcing_episode
from ..reporting import environment_metadata, write_json
from ..schema import LayoutSpec


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_base_model(model_config: Mapping[str, Any], *, training: bool) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"], revision=model_config["revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=None,
    ).to("cuda")
    model.config.use_cache = False
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval().requires_grad_(False)
    return model


def support_for(example: Phase2Example) -> set[str]:
    groups = example.episode.acceptable_support_sets
    return set(min(groups, key=lambda group: (len(group), sorted(group))))


def pack_training_example(example: Phase2Example, layout: LayoutSpec, tokenizer: HuggingFaceTokenizer,
                          mode: Phase2Mode):
    memory = mode.uses_memory
    return pack_teacher_forcing_episode(
        example.episode, layout, tokenizer,
        selected_blocks=support_for(example) if mode == Phase2Mode.BLOCK_ORACLE else None,
        placeholder_token_id=int(tokenizer._tokenizer.pad_token_id or 0),
        enable_memory_tokens=memory, enable_router_tokens=memory,
    )


def cache_metadata(config: Mapping[str, Any], model_config: Mapping[str, Any], corpus: Phase3Corpus) -> dict[str, Any]:
    return {
        "model_id": model_config["model_id"], "revision": model_config["revision"],
        "tokenizer": corpus.manifest["tokenizer"], "layout": dict(config["layout"]),
        "train_corpus_sha256": corpus.manifest["sha256"]["train"],
        "mode": Phase2Mode.DENSE_ALIGNED.value,
        "top_k": int(config["training"]["teacher_top_k"]),
    }


@torch.inference_mode()
def build_teacher_cache(config: Mapping[str, Any], model_name: str, destination: str | Path,
                        *, limit: int | None = None, seed: int | None = None) -> dict[str, Any]:
    model_config = config["models"][model_name]
    tokenizer = HuggingFaceTokenizer(model_config["model_id"], model_config["revision"])
    corpus = build_phase3_corpus(tokenizer); examples: Sequence[Phase2Example] = corpus.train
    if limit is not None:
        examples = stratified_subset(examples, limit, int(config["training"]["primary_seed"] if seed is None else seed))
    model = load_base_model(model_config, training=False); layout = LayoutSpec(**config["layout"])
    writer = TeacherCacheWriter(destination, cache_metadata(config, model_config, corpus))
    set_model_attention(model, Phase2Mode.DENSE_ALIGNED)
    for index, example in enumerate(examples):
        teacher = pack_training_example(example, layout, tokenizer, Phase2Mode.DENSE_ALIGNED)
        prepared = prepare_model_input(model, teacher, Phase2Mode.DENSE_ALIGNED)
        logits = model(**prepared.model_kwargs).logits
        writer.add(example.episode.episode_id, compress_teacher_logits(
            logits, int(config["training"]["teacher_top_k"]),
        ))
        if (index + 1) % 25 == 0 or index + 1 == len(examples):
            print(f"teacher cache: {index + 1}/{len(examples)}", flush=True)
    manifest = writer.finish(); del model; torch.cuda.empty_cache()
    return manifest


def _optimizer(model: OracleAdaptationModel, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    lora = []; auxiliary = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if "lora_" in name else auxiliary).append(parameter)
    return torch.optim.AdamW([
        {"params": lora, "lr": training["learning_rate_lora"], "weight_decay": training["weight_decay_lora"]},
        {"params": auxiliary, "lr": training["learning_rate_tokens_router"], "weight_decay": 0.0},
    ], betas=(training["adam_beta1"], training["adam_beta2"]), eps=training["adam_epsilon"])


def _scheduler(optimizer: torch.optim.Optimizer, steps: int, warmup_fraction: float):
    warmup = max(1, round(steps * warmup_fraction))
    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.inference_mode()
def _development_metrics(model: OracleAdaptationModel, examples: Sequence[Phase2Example],
                         layout: LayoutSpec, tokenizer: HuggingFaceTokenizer) -> dict[str, Any]:
    """Measure the two paths used for retention-aware checkpoint selection."""
    was_training = model.training
    model.eval()
    metrics: dict[str, Any] = {}
    try:
        for mode in (Phase2Mode.DENSE_ALIGNED, Phase2Mode.BLOCK_ORACLE):
            nll_sum = 0.0; correct = 0; target_tokens = 0
            for example in examples:
                teacher = pack_training_example(example, layout, tokenizer, mode)
                output = model(teacher, mode)
                logits = output.logits.float(); targets = output.prepared.target_token_ids
                nll_sum += float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum",
                ))
                correct += int(logits.argmax(-1).eq(targets).sum())
                target_tokens += int(targets.numel())
            metrics[mode.value] = {
                "episodes": len(examples), "target_tokens": target_tokens,
                "nll_per_token": nll_sum / target_tokens,
                "token_accuracy": correct / target_tokens,
            }
    finally:
        model.train(was_training)
    return metrics


def _prune_checkpoints(output: Path, evaluations: Sequence[Mapping[str, Any]], current: str) -> list[str]:
    qualifying = sorted(
        (row for row in evaluations if row["qualifies_dense_retention"]),
        key=lambda row: (row["block_oracle"]["nll_per_token"], row["step"]),
    )[:3]
    keep = {str(row["checkpoint"]) for row in qualifying}; keep.add(current)
    for path in output.glob("checkpoint-*"):
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path)
    return sorted(keep)


def _save_checkpoint(destination: Path, model: OracleAdaptationModel,
                     optimizer: torch.optim.Optimizer, scheduler: Any,
                     state: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        model.language_model.save_pretrained(temporary / "adapter", safe_serialization=True,
                                             save_embedding_layers=False)
        tensors = {f"token_bank.{key}": value.detach().cpu() for key, value in model.token_bank.state_dict().items()}
        tensors.update({f"router_head.{key}": value.detach().cpu() for key, value in model.router_head.state_dict().items()})
        save_file(tensors, temporary / "auxiliary.safetensors")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate()}, temporary / "trainer_state.pt")
        write_json(temporary / "manifest.json", state)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _restore_training_model(base: torch.nn.Module, checkpoint: Path) -> tuple[OracleAdaptationModel, dict[str, Any]]:
    from peft import PeftModel

    manifest = json.loads((checkpoint / "manifest.json").read_text())
    language_model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=True)
    language_model.enable_input_require_grads()
    hidden = int(base.config.hidden_size); seed = int(manifest["seed"])
    bank = MemoryRouterTokenBank(hidden, memory_tokens=8, router_tokens=8, seed=seed)
    router = OracleRouterHead(hidden); tensors = load_file(checkpoint / "auxiliary.safetensors")
    bank.load_state_dict({key.removeprefix("token_bank."): value for key, value in tensors.items()
                          if key.startswith("token_bank.")})
    router.load_state_dict({key.removeprefix("router_head."): value for key, value in tensors.items()
                            if key.startswith("router_head.")})
    return OracleAdaptationModel(language_model, bank, router).to("cuda").train(), manifest


@dataclass(slots=True)
class TrainingResult:
    model: OracleAdaptationModel
    report: dict[str, Any]


def run_training(config: Mapping[str, Any], model_name: str, teacher_cache_path: str | Path,
                 output_dir: str | Path, *, limit: int | None = None, seed: int | None = None,
                 max_optimizer_steps: int | None = None, resume: str | Path | None = None) -> TrainingResult:
    model_config = config["models"][model_name]; training = config["training"]
    seed = int(training["primary_seed"] if seed is None else seed)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tokenizer = HuggingFaceTokenizer(model_config["model_id"], model_config["revision"])
    corpus = build_phase3_corpus(tokenizer); examples: Sequence[Phase2Example] = corpus.train
    if limit is not None:
        examples = stratified_subset(examples, limit, seed)
    else:
        examples = list(examples)
        random.Random(seed).shuffle(examples)
    teacher_cache = TeacherCache(teacher_cache_path, cache_metadata(config, model_config, corpus))
    missing = [item.episode.episode_id for item in examples if item.episode.episode_id not in teacher_cache.manifest["entries"]]
    if missing:
        raise ValueError(f"teacher cache misses {len(missing)} selected examples")
    base = load_base_model(model_config, training=True)
    if resume is None:
        model = build_oracle_adaptation_model(
            base, seed=seed, rank=int(config["lora"]["rank"]), alpha=int(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        ).to("cuda")
        resume_manifest = None
    else:
        model, resume_manifest = _restore_training_model(base, Path(resume))
        if int(resume_manifest["seed"]) != seed:
            raise ValueError("resume checkpoint seed differs from requested seed")
    audit = trainable_parameter_audit(model); optimizer = _optimizer(model, config)
    accumulation = int(training["gradient_accumulation"])
    total_steps = math.ceil(len(examples) / accumulation)
    if max_optimizer_steps is not None:
        total_steps = min(total_steps, max_optimizer_steps)
    scheduler = _scheduler(optimizer, total_steps, float(training["warmup_fraction"]))
    optimizer_step = 0; starting_microstep = 0
    if resume is not None:
        saved = torch.load(Path(resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                for key, value in optimizer.state[parameter].items():
                    if torch.is_tensor(value):
                        optimizer.state[parameter][key] = value.to(parameter.device)
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        optimizer_step = int(resume_manifest["step"]); starting_microstep = optimizer_step * accumulation
    layout = LayoutSpec(**config["layout"]); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "result_schema_version": 1, "phase": 3, "kind": "training", "environment": environment_metadata(),
        "model": dict(model_config), "seed": seed, "configuration": dict(config),
        "corpus": dict(corpus.manifest), "trainable": audit, "steps": [], "development": [],
        "config_sha256": _sha256_json(config),
        "teacher_cache": {"path": str(teacher_cache_path), "metadata": teacher_cache.manifest["metadata"],
                          "shard_sha256": teacher_cache.manifest["shard_sha256"]},
        "resumed_from": str(resume) if resume is not None else None,
    }
    formal_run = max_optimizer_steps is None
    development_baseline = None
    if formal_run:
        development_baseline = _development_metrics(model, corpus.development, layout, tokenizer)
        report["development_baseline"] = development_baseline
    optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for microstep, example in enumerate(examples[starting_microstep:], start=starting_microstep):
        if optimizer_step >= total_steps:
            break
        mode = (Phase2Mode.DENSE_ALIGNED if (microstep + 1) % int(training["dense_replay_interval"]) == 0
                else Phase2Mode.BLOCK_ORACLE)
        teacher = pack_training_example(example, layout, tokenizer, mode)
        output_value = model(teacher, mode)
        losses = phase3_loss(
            output_value, teacher, teacher_cache.get(example.episode.episode_id),
            kl_weight=float(training["kl_weight"]), router_weight=float(training["router_weight"]),
            positive_weight=float(training["router_positive_weight"]),
            no_tool_weight=float(training["no_tool_weight"]),
        )
        (losses.total / accumulation).backward()
        completes_step = (microstep + 1) % accumulation == 0 or microstep + 1 == len(examples)
        if not completes_step:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {optimizer_step + 1}")
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); optimizer_step += 1
        row = {
            "step": optimizer_step, "microstep": microstep + 1, "mode": mode.value,
            "loss": float(losses.total.detach()), "answer_loss": float(losses.answer.detach()),
            "kl_loss": float(losses.distillation.detach()), "router_loss": float(losses.router.detach()),
            "gradient_norm": float(gradient_norm), "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        report["steps"].append(row); write_json(output / "training.json", report)
        print(f"phase3 train: {optimizer_step}/{total_steps} loss={row['loss']:.4f}", flush=True)
        is_final = optimizer_step == total_steps
        development = None
        if formal_run and (optimizer_step % int(training["eval_every_steps"]) == 0 or is_final):
            development = _development_metrics(model, corpus.development, layout, tokenizer)
            regression = (development[Phase2Mode.DENSE_ALIGNED.value]["nll_per_token"]
                          - development_baseline[Phase2Mode.DENSE_ALIGNED.value]["nll_per_token"])
            development.update({
                "step": optimizer_step, "dense_nll_regression": regression,
                "qualifies_dense_retention": regression <= float(config["acceptance"]["dense_nll_regression_max"]),
                "checkpoint": f"checkpoint-{optimizer_step:05d}",
            })
            report["development"].append(development)
        if optimizer_step % int(training["save_every_steps"]) == 0 or is_final:
            checkpoint_name = f"checkpoint-{optimizer_step:05d}"
            _save_checkpoint(output / checkpoint_name, model, optimizer, scheduler,
                             {"step": optimizer_step, "seed": seed, "model": model_config,
                              "config_sha256": report["config_sha256"],
                              "corpus_sha256": corpus.manifest["sha256"]["train"],
                              "teacher_cache_metadata": teacher_cache.manifest["metadata"],
                              "development": development})
            if formal_run and report["development"]:
                report["retained_checkpoints"] = _prune_checkpoints(
                    output, report["development"], checkpoint_name,
                )
    report["completed_steps"] = optimizer_step
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["elapsed_seconds"] = time.perf_counter() - started; write_json(output / "training.json", report)
    return TrainingResult(model, report)
