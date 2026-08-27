from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor, nn

from ..masks.flex import TensorMaskMetadata, build_flex_block_mask
from ..masks.reference import MaskOptions
from ..packing import TeacherForcedEpisode, TokenRole
from ..schema import CacheSemantics, ValidationError


FLEX_KERNEL_OPTIONS = {"BACKEND": "TRITON", "ROWS_GUARANTEED_SAFE": False}


class Phase2Mode(StrEnum):
    DENSE_COMPACT = "dense_compact"
    DENSE_ALIGNED = "dense_aligned"
    BLOCK_STATIC = "block_static"
    BLOCK_MEMORY = "block_memory"
    BLOCK_ORACLE = "block_oracle"
    BLOCK_DYNAMIC_ANCHOR = "block_dynamic_anchor"

    @property
    def uses_memory(self) -> bool:
        return self in {
            Phase2Mode.BLOCK_MEMORY, Phase2Mode.BLOCK_ORACLE, Phase2Mode.BLOCK_DYNAMIC_ANCHOR,
        }

    @property
    def uses_flex(self) -> bool:
        return self not in {Phase2Mode.DENSE_COMPACT, Phase2Mode.DENSE_ALIGNED}


class MemoryRouterTokenBank(nn.Module):
    def __init__(
        self, hidden_size: int, *, memory_tokens: int = 8, router_tokens: int = 8,
        initializer_range: float = 0.02, seed: int = 1729,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        memory = torch.empty(memory_tokens, hidden_size, dtype=torch.float32)
        router = torch.empty(router_tokens, hidden_size, dtype=torch.float32)
        memory.normal_(mean=0.0, std=initializer_range, generator=generator)
        router.normal_(mean=0.0, std=initializer_range, generator=generator)
        self.memory = nn.Parameter(memory)
        self.router = nn.Parameter(router)
        self.seed = seed

    def checksums(self) -> dict[str, str]:
        import hashlib

        return {
            "memory": hashlib.sha256(self.memory.detach().float().cpu().numpy().tobytes()).hexdigest(),
            "router": hashlib.sha256(self.router.detach().float().cpu().numpy().tobytes()).hexdigest(),
        }

    def apply(self, base_embeddings: Tensor, teacher: TeacherForcedEpisode) -> Tensor:
        packed = teacher.packed
        if base_embeddings.shape[:2] != (1, packed.layout.sequence_length):
            raise ValidationError("embedding shape does not match the packed episode")
        result = base_embeddings.clone()
        roles = torch.tensor([int(item) for item in packed.token_role], device=result.device)
        local = torch.tensor(packed.local_position, device=result.device)
        memory_positions = torch.nonzero(roles == int(TokenRole.M), as_tuple=False).flatten()
        router_positions = torch.nonzero(roles == int(TokenRole.R), as_tuple=False).flatten()
        if memory_positions.numel():
            memory_local = local.index_select(0, memory_positions)
            if int(memory_local.max()) >= self.memory.shape[0]:
                raise ValidationError("memory-token bank is smaller than the packed memory strip")
            result[0, memory_positions] = self.memory.to(result.dtype).index_select(0, memory_local)
        if router_positions.numel():
            router_local = local.index_select(0, router_positions)
            if int(router_local.max()) >= self.router.shape[0]:
                raise ValidationError("router-token bank is smaller than the packed router strip")
            result[0, router_positions] = self.router.to(result.dtype).index_select(0, router_local)
        return result


@dataclass(slots=True)
class PreparedModelInput:
    model_kwargs: dict[str, Any]
    target_token_ids: Tensor
    prediction_positions: Tensor
    target_positions: Tensor
    block_mask: Any | None = None


def _tensor_ids(teacher: TeacherForcedEpisode, device: torch.device) -> Tensor:
    return torch.tensor(teacher.packed.input_ids, dtype=torch.long, device=device).unsqueeze(0)


def _target_tensors(teacher: TeacherForcedEpisode, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.tensor(teacher.target_token_ids, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(teacher.prediction_positions, dtype=torch.long, device=device),
        torch.tensor(teacher.target_positions, dtype=torch.long, device=device),
    )


def prepare_model_input(
    model: nn.Module,
    teacher: TeacherForcedEpisode,
    mode: Phase2Mode,
    *,
    token_bank: MemoryRouterTokenBank | None = None,
) -> PreparedModelInput:
    device = next(model.parameters()).device
    input_ids = _tensor_ids(teacher, device)
    targets, prediction_positions, target_positions = _target_tensors(teacher, device)
    packed = teacher.packed
    if mode == Phase2Mode.DENSE_COMPACT:
        valid_positions = [index for index, valid in enumerate(packed.valid_token) if valid]
        position_map = {original: compact for compact, original in enumerate(valid_positions)}
        compact_ids = input_ids.index_select(1, torch.tensor(valid_positions, device=device))
        compact_targets = torch.tensor(
            [position_map[index] for index in teacher.target_positions], dtype=torch.long, device=device,
        )
        compact_predictions = compact_targets - 1
        return PreparedModelInput(
            model_kwargs={
                "input_ids": compact_ids,
                "attention_mask": torch.ones_like(compact_ids, dtype=torch.bool),
                "position_ids": torch.arange(compact_ids.shape[1], device=device).unsqueeze(0),
                "use_cache": False,
                "logits_to_keep": compact_predictions,
            },
            target_token_ids=targets, prediction_positions=compact_predictions,
            target_positions=compact_targets,
        )

    base_embeddings = model.get_input_embeddings()(input_ids)
    if mode.uses_memory:
        if token_bank is None:
            raise ValidationError(f"{mode.value} requires a memory/router token bank")
        inputs_embeds = token_bank.apply(base_embeddings, teacher)
    else:
        inputs_embeds = base_embeddings
    if model.training and not inputs_embeds.requires_grad:
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)
    common = {
        "inputs_embeds": inputs_embeds,
        "position_ids": torch.arange(packed.layout.sequence_length, device=device).unsqueeze(0),
        "use_cache": False,
        "logits_to_keep": prediction_positions,
    }
    block_mask = None
    if mode == Phase2Mode.DENSE_ALIGNED:
        common["attention_mask"] = torch.tensor(
            packed.valid_token, dtype=torch.bool, device=device,
        ).unsqueeze(0)
    else:
        options = MaskOptions(
            cache_semantics=(
                CacheSemantics.EPISODE_ANCHORED
                if mode == Phase2Mode.BLOCK_DYNAMIC_ANCHOR
                else CacheSemantics.PERSISTENT_ARTIFACT
            ),
            answer_sees_memory=False,
        )
        metadata = TensorMaskMetadata.from_packed([packed], device=device)
        block_mask = build_flex_block_mask(metadata, options)
        common["attention_mask"] = {"full_attention": block_mask}
        common["kernel_options"] = FLEX_KERNEL_OPTIONS
    return PreparedModelInput(
        model_kwargs=common, target_token_ids=targets,
        prediction_positions=prediction_positions, target_positions=target_positions,
        block_mask=block_mask,
    )


def set_model_attention(model: nn.Module, mode: Phase2Mode) -> None:
    model.set_attn_implementation("flex_attention" if mode.uses_flex else "sdpa")


def assert_phase2_model(model: nn.Module) -> None:
    if model.training:
        raise ValidationError("Phase 2 model must be in eval mode")
    if float(model.config.attention_dropout) != 0.0:
        raise ValidationError("Phase 2 requires zero attention dropout")
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cuda"}:
        raise ValidationError(f"all model parameters must be on CUDA, found {sorted(devices)}")
