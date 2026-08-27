from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from safetensors.torch import load_file, save_file

from ..reporting import write_json
from ..schema import ValidationError


@dataclass(frozen=True, slots=True)
class TeacherDistribution:
    token_ids: Tensor
    log_probabilities: Tensor
    residual_log_probability: Tensor

    def to(self, device: torch.device | str) -> "TeacherDistribution":
        return TeacherDistribution(
            self.token_ids.to(device), self.log_probabilities.to(device),
            self.residual_log_probability.to(device),
        )


def compress_teacher_logits(logits: Tensor, top_k: int = 256) -> TeacherDistribution:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValidationError("teacher logits must have shape [1, targets, vocabulary]")
    log_probs = torch.log_softmax(logits.float(), dim=-1).squeeze(0)
    values, indices = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]), dim=-1)
    top_mass = values.exp().sum(-1).clamp(max=1.0 - 1e-7)
    residual = torch.log1p(-top_mass)
    return TeacherDistribution(indices.to(torch.int32).cpu(), values.to(torch.float16).cpu(), residual.cpu())


def residual_kl(student_logits: Tensor, teacher: TeacherDistribution) -> Tensor:
    student = torch.log_softmax(student_logits.float().squeeze(0), dim=-1)
    teacher = teacher.to(student.device)
    indices = teacher.token_ids.long()
    student_top = student.gather(-1, indices)
    teacher_top = teacher.log_probabilities.float()
    teacher_mass = teacher_top.exp()
    student_top_mass = student_top.exp().sum(-1).clamp(max=1.0 - 1e-7)
    student_residual = torch.log1p(-student_top_mass)
    teacher_residual = teacher.residual_log_probability.float()
    value = (teacher_mass * (teacher_top - student_top)).sum(-1)
    value += teacher_residual.exp() * (teacher_residual - student_residual)
    # Cached top-k log probabilities use fp16, so roundoff can put the
    # analytically non-negative KL a few ulps below zero.
    return value.mean().clamp_min(0.0)


class TeacherCacheWriter:
    def __init__(self, destination: str | Path, metadata: Mapping[str, Any], *, shard_size: int = 512) -> None:
        self.destination = Path(destination); self.destination.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata); self.shard_size = shard_size
        self.pending: list[tuple[str, TeacherDistribution]] = []; self.entries: dict[str, Any] = {}
        self.shard_hashes: dict[str, str] = {}; self.shards = 0

    def add(self, episode_id: str, distribution: TeacherDistribution) -> None:
        if episode_id in self.entries or any(name == episode_id for name, _ in self.pending):
            raise ValidationError(f"duplicate teacher-cache episode {episode_id}")
        self.pending.append((episode_id, distribution))
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        shard_name = f"teacher-{self.shards:05d}.safetensors"; offsets = [0]
        for _, item in self.pending:
            offsets.append(offsets[-1] + item.token_ids.shape[0])
        shard_path = self.destination / shard_name
        save_file({
            "token_ids": torch.cat([item.token_ids for _, item in self.pending]),
            "log_probabilities": torch.cat([item.log_probabilities for _, item in self.pending]),
            "residual_log_probability": torch.cat([item.residual_log_probability for _, item in self.pending]),
        }, shard_path)
        self.shard_hashes[shard_name] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        for index, (episode_id, _) in enumerate(self.pending):
            self.entries[episode_id] = {"shard": shard_name, "start": offsets[index], "end": offsets[index + 1]}
        self.pending.clear(); self.shards += 1

    def finish(self) -> dict[str, Any]:
        self.flush()
        manifest = {"cache_schema_version": 1, "metadata": self.metadata,
                    "episode_count": len(self.entries), "entries": self.entries,
                    "shard_sha256": self.shard_hashes}
        write_json(self.destination / "manifest.json", manifest)
        return manifest


class TeacherCache:
    def __init__(self, source: str | Path, expected_metadata: Mapping[str, Any] | None = None) -> None:
        self.source = Path(source); self.manifest = json.loads((self.source / "manifest.json").read_text())
        if self.manifest.get("cache_schema_version") != 1:
            raise ValidationError("unsupported teacher cache schema")
        if expected_metadata is not None and self.manifest["metadata"] != dict(expected_metadata):
            raise ValidationError("teacher cache metadata does not match the training run")
        self._loaded_name: str | None = None; self._loaded: dict[str, Tensor] | None = None

    def __len__(self) -> int:
        return int(self.manifest["episode_count"])

    def get(self, episode_id: str) -> TeacherDistribution:
        try:
            entry = self.manifest["entries"][episode_id]
        except KeyError as exc:
            raise ValidationError(f"teacher cache misses episode {episode_id}") from exc
        if entry["shard"] != self._loaded_name:
            shard_path = self.source / entry["shard"]
            expected = self.manifest["shard_sha256"][entry["shard"]]
            if hashlib.sha256(shard_path.read_bytes()).hexdigest() != expected:
                raise ValidationError(f"teacher cache shard checksum failed: {entry['shard']}")
            self._loaded = load_file(shard_path); self._loaded_name = entry["shard"]
        assert self._loaded is not None
        region = slice(int(entry["start"]), int(entry["end"]))
        return TeacherDistribution(
            self._loaded["token_ids"][region], self._loaded["log_probabilities"][region],
            self._loaded["residual_log_probability"][region],
        )
