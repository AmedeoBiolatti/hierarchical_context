from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable

import torch

from .attention.torch_flex import dense_causal_sdpa, dense_masked_sdpa, flex_attention_forward, segmented_sdpa
from .masks.flex import TensorMaskMetadata, build_dense_mask, build_flex_block_mask, clear_block_mask_cache
from .packing import PackedEpisode, TokenRole
from .reporting import environment_metadata, write_json
from .schema import LayoutSpec


@dataclass(frozen=True, slots=True)
class BenchmarkShape:
    name: str
    total_tokens: int
    global_capacity: int
    tool_count: int
    tool_capacity: int
    memory_tokens_per_block: int = 8
    router_capacity: int = 8
    answer_capacity: int = 256
    tile_size: int = 128

    def layout(self) -> LayoutSpec:
        layout = LayoutSpec(
            self.name, self.global_capacity, self.tool_count, self.tool_capacity,
            self.memory_tokens_per_block, self.router_capacity, self.answer_capacity, self.tile_size,
        )
        if layout.sequence_length != self.total_tokens:
            raise ValueError(f"{self.name} produces {layout.sequence_length} tokens, expected {self.total_tokens}")
        return layout


def benchmark_packed(layout: LayoutSpec, selected_slots: Iterable[int]) -> PackedEpisode:
    selected_slots = set(selected_slots)
    roles: list[TokenRole] = []; blocks: list[str | None] = []; local: list[int] = []
    events: list[int] = []; selected: list[bool] = []; valid: list[bool] = []

    def region(role: TokenRole, capacity: int, *, slot: int | None = None, count: int | None = None) -> None:
        active = capacity if count is None else count
        for position in range(capacity):
            is_valid = position < active
            roles.append(role if is_valid else TokenRole.PAD)
            blocks.append(f"tool-{slot}" if is_valid and slot is not None else None)
            local.append(position if is_valid else -1); events.append((slot + 1) if slot is not None else 0)
            selected.append(bool(is_valid and slot in selected_slots)); valid.append(is_valid)

    region(TokenRole.G, layout.global_capacity)
    for slot in range(layout.tool_count):
        region(TokenRole.T, layout.tool_capacity, slot=slot)
    memory_count = layout.tool_count * layout.memory_tokens_per_block
    for slot in range(layout.tool_count):
        region(TokenRole.M, layout.memory_tokens_per_block, slot=slot)
    memory_padding = layout.summary_capacity - memory_count
    if memory_padding:
        region(TokenRole.PAD, memory_padding, count=0)
    region(TokenRole.R, layout.router_strip_capacity, count=layout.router_capacity)
    region(TokenRole.A, layout.answer_strip_capacity, count=layout.answer_capacity)
    length = layout.sequence_length
    return PackedEpisode(
        episode_id=f"benchmark-{layout.name}", layout=layout, tokenizer_identity={"kind": "benchmark"},
        input_ids=tuple(0 for _ in range(length)), token_role=tuple(roles), block_id=tuple(blocks),
        local_position=tuple(local), event_position=tuple(events), selected_block=tuple(selected),
        valid_token=tuple(valid), episode_anchor_key=tuple(False for _ in range(length)),
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _time_cuda(function: Callable[[], torch.Tensor], warmups: int, repetitions: int) -> dict[str, Any]:
    torch.cuda.synchronize(); start_wall = time.perf_counter()
    output = function(); torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start_wall) * 1000
    for _ in range(warmups):
        output = function()
    torch.cuda.synchronize(); del output
    samples: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); output = function(); end.record(); end.synchronize()
        samples.append(float(start.elapsed_time(end))); del output
    return {
        "first_call_wall_ms": first_ms,
        "samples_ms": samples,
        "median_ms": statistics.median(samples), "p10_ms": _percentile(samples, 0.10),
        "p90_ms": _percentile(samples, 0.90),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def _mask_build(metadata: TensorMaskMetadata) -> tuple[Any, dict[str, float]]:
    clear_block_mask_cache(); torch.cuda.synchronize(); start = time.perf_counter()
    mask = build_flex_block_mask(metadata); torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter(); cached = build_flex_block_mask(metadata); torch.cuda.synchronize()
    cache_ms = (time.perf_counter() - start) * 1000
    assert cached is mask
    return mask, {"cold_ms": cold_ms, "cache_lookup_ms": cache_ms}


def _case(
    backend: str, layout: LayoutSpec, raw_fraction: float, function: Callable[[], torch.Tensor],
    warmups: int, repetitions: int, *, mask_build: dict[str, float] | None = None,
    active_attention_tile_fraction: float | None = None,
) -> dict[str, Any]:
    base = {
        "backend": backend, "layout": layout.name, "sequence_length": layout.sequence_length,
        "raw_tool_selection_fraction": raw_fraction, "mask_build": mask_build,
        "active_attention_tile_fraction": active_attention_tile_fraction,
    }
    try:
        base.update({"status": "ok", "timing": _time_cuda(function, warmups, repetitions)})
    except torch.OutOfMemoryError as exc:
        base.update({"status": "oom", "error": str(exc)})
        torch.cuda.empty_cache()
    except (RuntimeError, NotImplementedError) as exc:
        base.update({"status": "unsupported", "error": str(exc)})
        torch.cuda.empty_cache()
    return base


def cuda_correctness_probe() -> dict[str, Any]:
    layout = LayoutSpec(
        "correctness_probe", 128, 2, 128, memory_tokens_per_block=8,
        router_capacity=8, answer_capacity=128, tile_size=128,
    )
    metadata = TensorMaskMetadata.from_packed([benchmark_packed(layout, {0})], device="cuda")
    dense_mask = build_dense_mask(metadata); block_mask = build_flex_block_mask(metadata, use_cache=False)
    torch.manual_seed(1729); length = layout.sequence_length
    base = (
        torch.randn(1, 4, length, 32, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 2, length, 32, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 2, length, 32, device="cuda", dtype=torch.bfloat16),
    )
    dense_inputs = [item.clone().requires_grad_(True) for item in base]
    flex_inputs = [item.clone().requires_grad_(True) for item in base]
    dense = dense_masked_sdpa(*dense_inputs, dense_mask); flex = flex_attention_forward(*flex_inputs, block_mask)
    torch.testing.assert_close(flex, dense, rtol=0.03, atol=0.03)
    dense.float().square().mean().backward(); flex.float().square().mean().backward()
    max_gradient_error = 0.0
    for dense_item, flex_item in zip(dense_inputs, flex_inputs, strict=True):
        torch.testing.assert_close(flex_item.grad, dense_item.grad, rtol=0.05, atol=0.05)
        max_gradient_error = max(
            max_gradient_error, float((flex_item.grad - dense_item.grad).abs().max().item()),
        )
    return {
        "passed": True, "source": "CUDA probe plus tests/test_flex_attention.py",
        "max_output_absolute_error": float((flex - dense).abs().max().item()),
        "max_gradient_absolute_error": max_gradient_error,
    }


def run_phase1_benchmarks(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 benchmarks require CUDA")
    seed = int(config.get("seed", 1729)); torch.manual_seed(seed); random.seed(seed)
    warmups = int(config.get("warmups", 10)); repetitions = int(config.get("repetitions", 30))
    q_heads = int(config.get("query_heads", 16)); kv_heads = int(config.get("kv_heads", 8))
    head_dim = int(config.get("head_dim", 128)); device = torch.device("cuda")
    shapes = [BenchmarkShape(**item) for item in config["layouts"]]
    correctness = cuda_correctness_probe()
    report: dict[str, Any] = {
        "result_schema_version": 2, "phase": 1, "environment": environment_metadata(),
        "configuration": config, "correctness": correctness,
        "benchmarks": [], "decision": None,
    }
    prior_oom: set[str] = set(); slower_at: dict[str, set[int]] = {}
    for shape in shapes:
        layout = shape.layout(); length = layout.sequence_length
        query = torch.randn(1, q_heads, length, head_dim, device=device, dtype=torch.bfloat16)
        key = torch.randn(1, kv_heads, length, head_dim, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(key)

        cases: list[dict[str, Any]] = []
        if "dense_causal" not in prior_oom:
            cases.append(_case("dense_causal", layout, 1.0, lambda: dense_causal_sdpa(query, key, value), warmups, repetitions))
        else:
            cases.append({"backend": "dense_causal", "layout": layout.name, "sequence_length": length,
                          "raw_tool_selection_fraction": 1.0, "status": "skipped_after_gate",
                          "reason": "smaller layout exhausted memory"})
        all_packed = benchmark_packed(layout, range(layout.tool_count))
        all_metadata = TensorMaskMetadata.from_packed([all_packed], device=device)
        all_mask = None; build_times = None
        try:
            all_mask, build_times = _mask_build(all_metadata)
            density = 1.0 - all_mask.sparsity() / 100.0
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            cases.append({"backend": "flex_all", "layout": layout.name, "sequence_length": length,
                          "raw_tool_selection_fraction": 1.0, "status": "oom", "error": str(exc)})
            density = None; torch.cuda.empty_cache()
        if all_mask is not None:
            cases.append(_case(
                "flex_all", layout, 1.0, lambda: flex_attention_forward(query, key, value, all_mask),
                warmups, repetitions, mask_build=build_times, active_attention_tile_fraction=density,
            ))
        dense_masked_gated = length == 32768 and {8192, 16384}.issubset(slower_at.get("dense_masked", set()))
        if "dense_masked" not in prior_oom and not dense_masked_gated:
            try:
                torch.cuda.synchronize(); start = time.perf_counter(); dense_mask = build_dense_mask(all_metadata)
                torch.cuda.synchronize(); dense_build = {"cold_ms": (time.perf_counter() - start) * 1000}
                cases.append(_case(
                    "dense_masked", layout, 1.0,
                    lambda: dense_masked_sdpa(query, key, value, dense_mask), warmups, repetitions,
                    mask_build=dense_build, active_attention_tile_fraction=density,
                ))
                del dense_mask
            except torch.OutOfMemoryError as exc:
                cases.append({"backend": "dense_masked", "layout": layout.name, "sequence_length": length,
                              "raw_tool_selection_fraction": 1.0, "status": "oom", "error": str(exc)})
                torch.cuda.empty_cache()
        elif dense_masked_gated or "dense_masked" in prior_oom:
            cases.append({"backend": "dense_masked", "layout": layout.name, "sequence_length": length,
                          "raw_tool_selection_fraction": 1.0, "status": "skipped_after_gate",
                          "reason": "2x slower at 8K and 16K" if dense_masked_gated else "smaller layout exhausted memory"})
        segmented_gated = length == 32768 and {8192, 16384}.issubset(slower_at.get("segmented", set()))
        if "segmented" not in prior_oom and not segmented_gated:
            cases.append(_case(
                "segmented", layout, 1.0,
                lambda: segmented_sdpa(query, key, value, all_metadata), warmups, repetitions,
                active_attention_tile_fraction=density,
            ))
        else:
            cases.append({"backend": "segmented", "layout": layout.name, "sequence_length": length,
                          "raw_tool_selection_fraction": 1.0, "status": "skipped_after_gate",
                          "reason": "2x slower at 8K and 16K" if segmented_gated else "smaller layout exhausted memory"})
        for fraction in (0.125, 0.25, 0.5):
            name = f"flex_selected_{fraction:g}"
            if name in prior_oom:
                cases.append({"backend": name, "layout": layout.name, "sequence_length": length,
                              "raw_tool_selection_fraction": fraction, "status": "skipped_after_gate",
                              "reason": "smaller layout exhausted memory"})
                continue
            count = max(1, round(layout.tool_count * fraction))
            selected_slots = sorted(random.Random(f"{seed}:{layout.name}:{fraction}").sample(range(layout.tool_count), count))
            metadata = TensorMaskMetadata.from_packed([benchmark_packed(layout, selected_slots)], device=device)
            try:
                mask, selected_build = _mask_build(metadata); selected_density = 1.0 - mask.sparsity() / 100.0
                cases.append(_case(
                    name, layout, fraction, lambda mask=mask: flex_attention_forward(query, key, value, mask),
                    warmups, repetitions, mask_build=selected_build,
                    active_attention_tile_fraction=selected_density,
                ))
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                cases.append({"backend": name, "layout": layout.name, "sequence_length": length,
                              "raw_tool_selection_fraction": fraction, "status": "oom", "error": str(exc)})
                torch.cuda.empty_cache()

        causal = next((item for item in cases if item["backend"] == "dense_causal" and item["status"] == "ok"), None)
        for item in cases:
            if item["status"] == "oom":
                prior_oom.add(item["backend"])
            if causal and item["status"] == "ok" and item["backend"] in {"dense_masked", "segmented"}:
                if item["timing"]["median_ms"] >= 2 * causal["timing"]["median_ms"]:
                    slower_at.setdefault(item["backend"], set()).add(length)
        report["benchmarks"].extend(cases); write_json(output, report)
        del query, key, value, all_metadata, all_mask; torch.cuda.empty_cache()

    from .acceptance import assess_phase1
    report["decision"] = assess_phase1(report)
    write_json(output, report)
    return report


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
