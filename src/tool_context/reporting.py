from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


RESULT_SCHEMA_VERSION = 1


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_metadata() -> dict[str, Any]:
    gpu_line = _command_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpu: dict[str, Any] | None = None
    if gpu_line:
        fields = [part.strip() for part in gpu_line.splitlines()[0].split(",")]
        if len(fields) == 3:
            gpu = {"name": fields[0], "memory_total_mib": int(fields[1]), "driver_version": fields[2]}
    packages: dict[str, Any] = {}
    try:
        import torch
        packages.update({
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        })
    except ImportError:
        pass
    try:
        import triton
        packages["triton"] = triton.__version__
    except ImportError:
        pass
    try:
        import transformers
        packages["transformers"] = transformers.__version__
    except ImportError:
        pass
    thermal = {
        "power_profile": _command_output(["powerprofilesctl", "get"]),
        "platform_profile": None,
        "gpu_state": _command_output([
            "nvidia-smi", "--query-gpu=pstate,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]),
    }
    try:
        thermal["platform_profile"] = Path("/sys/firmware/acpi/platform_profile").read_text().strip()
    except OSError:
        pass
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu": gpu,
        "cuda_runtime": _command_output(["nvcc", "--version"]),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "packages": packages,
        "thermal": thermal,
    }


def result_document(
    *, corpus_manifest: Mapping[str, Any], tokenizer_identity: Mapping[str, Any],
    selector_identities: Mapping[str, Any], metrics: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "environment": environment_metadata(),
        "command": sys.argv,
        "configuration": dict(configuration),
        "corpus": dict(corpus_manifest),
        "tokenizer": dict(tokenizer_identity),
        "selectors": dict(selector_identities),
        "metrics": dict(metrics),
        "performance": {
            "compile_time_ms": None,
            "mask_build_time_ms": None,
            "execution_time_ms": None,
            "peak_vram_mib": None,
            "active_token_fraction": None,
        },
    }


def write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
