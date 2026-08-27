from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import torch

from .benchmarking import BenchmarkShape, benchmark_packed
from .modeling.qwen3_block import MemoryRouterTokenBank, Phase2Mode, prepare_model_input, set_model_attention
from .packing import TeacherForcedEpisode, TokenRole
from .reporting import environment_metadata, write_json


PREFILL_MODES = ("dense_aligned", "block_static", "block_memory", "block_oracle")


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values); position = fraction * (len(values) - 1)
    low = math.floor(position); high = math.ceil(position)
    return values[low] if low == high else values[low] * (high - position) + values[high] * (position - low)


def _teacher(shape: BenchmarkShape, mode: Phase2Mode) -> TeacherForcedEpisode:
    layout = shape.layout()
    packed = benchmark_packed(layout, {0, 1} if mode == Phase2Mode.BLOCK_ORACLE else range(layout.tool_count))
    if mode in (Phase2Mode.DENSE_ALIGNED, Phase2Mode.BLOCK_STATIC):
        roles = tuple(TokenRole.PAD if role in (TokenRole.M, TokenRole.R) else role for role in packed.token_role)
        valid = tuple(False if role in (TokenRole.M, TokenRole.R) else okay
                      for role, okay in zip(packed.token_role, packed.valid_token, strict=True))
        packed = replace(packed, token_role=roles, valid_token=valid)
    target = layout.sequence_length - 1
    return TeacherForcedEpisode(packed, (target,), (target - 1,), (0,))


def run_prefill_worker(config: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    from .eval.quality import load_phase2_model

    started = time.perf_counter(); model = load_phase2_model(config)
    load_ms = (time.perf_counter() - started) * 1000
    shape = BenchmarkShape(**case["shape"]); mode = Phase2Mode(case["mode"])
    teacher = _teacher(shape, mode)
    bank = None
    if mode.uses_memory:
        bank = MemoryRouterTokenBank(
            model.config.hidden_size, memory_tokens=shape.memory_tokens_per_block,
            router_tokens=shape.router_capacity, initializer_range=float(model.config.initializer_range), seed=1730,
        ).to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    set_model_attention(model, mode)
    torch.cuda.synchronize(); started = time.perf_counter()
    prepared = prepare_model_input(model, teacher, mode, token_bank=bank)
    torch.cuda.synchronize(); mask_ms = (time.perf_counter() - started) * 1000
    kwargs = dict(prepared.model_kwargs)
    kwargs["logits_to_keep"] = torch.tensor([shape.total_tokens - 2], device="cuda")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        output = model(**kwargs)
    torch.cuda.synchronize(); first_ms = (time.perf_counter() - started) * 1000; del output
    for _ in range(int(config["prefill"]["warmups"])):
        with torch.inference_mode(): output = model(**kwargs)
    torch.cuda.synchronize(); del output
    samples = []
    for _ in range(int(config["prefill"]["repetitions"])):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode(): output = model(**kwargs)
        end.record(); end.synchronize(); samples.append(float(start.elapsed_time(end))); del output
    return {
        "status": "ok", "mode": mode.value, "layout": shape.name,
        "sequence_length": shape.total_tokens, "model_load_ms": load_ms, "mask_and_input_build_ms": mask_ms,
        "first_call_wall_ms": first_ms, "median_ms": statistics.median(samples),
        "p10_ms": _percentile(samples, .1), "p90_ms": _percentile(samples, .9), "samples_ms": samples,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "thermal": environment_metadata()["thermal"],
    }


def run_phase2_prefill(config: Mapping[str, Any], output: str | Path, script: str | Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "result_schema_version": 1, "phase": 2, "kind": "prefill", "environment": environment_metadata(),
        "configuration": dict(config), "fresh_process_per_case": True, "benchmarks": [],
    }
    write_json(output, report)
    with tempfile.TemporaryDirectory(prefix="phase2-prefill-") as temporary:
        config_path = Path(temporary) / "config.json"; write_json(config_path, config)
        for shape in config["prefill"]["layouts"]:
            for mode in PREFILL_MODES:
                case = {"shape": shape, "mode": mode}; case_path = Path(temporary) / "case.json"
                result_path = Path(temporary) / "result.json"; write_json(case_path, case)
                command = [sys.executable, str(script), "--worker", "--config", str(config_path),
                           "--case", str(case_path), "--output", str(result_path)]
                completed = subprocess.run(command, text=True, capture_output=True)
                if completed.returncode == 0:
                    row = json.loads(result_path.read_text())
                else:
                    error = (completed.stderr or completed.stdout).strip()
                    status = "oom" if "out of memory" in error.lower() else "error"
                    row = {"status": status, "mode": mode, "layout": shape["name"],
                           "sequence_length": shape["total_tokens"], "error": error[-4000:]}
                report["benchmarks"].append(row); write_json(output, report)
                print(f"phase2 prefill: {shape['name']} {mode} -> {row['status']}", flush=True)
    return report
