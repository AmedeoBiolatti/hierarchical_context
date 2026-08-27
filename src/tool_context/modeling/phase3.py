from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn

from .qwen3_block import MemoryRouterTokenBank, Phase2Mode, PreparedModelInput, prepare_model_input, set_model_attention
from ..packing import TeacherForcedEpisode, TokenRole
from ..schema import ValidationError


def _base_causal_lm(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


class OracleRouterHead(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int = 256) -> None:
        super().__init__()
        self.memory_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection_size))
        self.query_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection_size))
        self.no_tool = nn.Linear(projection_size, 1)
        self.scale = math.sqrt(projection_size)

    def forward(self, hidden: Tensor, teacher: TeacherForcedEpisode) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValidationError("Phase 3 router expects a batch of one")
        packed = teacher.packed; device = hidden.device
        roles = torch.tensor([int(role) for role in packed.token_role], device=device)
        memory_vectors = []
        ordered_blocks = list(dict.fromkeys(
            block for role, block in zip(packed.token_role, packed.block_id, strict=True)
            if role == TokenRole.T and block is not None
        ))
        for block in ordered_blocks:
            positions = torch.tensor([
                index for index, (role, item) in enumerate(zip(packed.token_role, packed.block_id, strict=True))
                if role == TokenRole.M and item == block
            ], device=device)
            if positions.numel() == 0:
                raise ValidationError(f"missing memory states for block {block}")
            memory_vectors.append(hidden[0].index_select(0, positions).float().mean(0))
        router_positions = torch.nonzero(roles == int(TokenRole.R), as_tuple=False).flatten()
        if router_positions.numel() == 0:
            raise ValidationError("missing router states")
        query = hidden[0].index_select(0, router_positions).float().mean(0)
        projected_query = self.query_projection(query)
        projected_memory = self.memory_projection(torch.stack(memory_vectors))
        block_logits = projected_memory @ projected_query / self.scale
        return block_logits.unsqueeze(0), self.no_tool(projected_query).reshape(1)


@dataclass(slots=True)
class AdaptationOutput:
    logits: Tensor
    prepared: PreparedModelInput
    block_logits: Tensor | None
    no_tool_logit: Tensor | None


class OracleAdaptationModel(nn.Module):
    def __init__(self, language_model: nn.Module, token_bank: MemoryRouterTokenBank,
                 router_head: OracleRouterHead) -> None:
        super().__init__()
        self.language_model = language_model
        self.token_bank = token_bank
        self.router_head = router_head

    def forward(self, teacher: TeacherForcedEpisode, mode: Phase2Mode) -> AdaptationOutput:
        set_model_attention(self.language_model, mode)
        prepared = prepare_model_input(
            self.language_model, teacher, mode,
            token_bank=self.token_bank if mode.uses_memory else None,
        )
        captured: list[Tensor] = []
        hook = None
        if mode.uses_memory:
            base = _base_causal_lm(self.language_model)
            hook = base.model.norm.register_forward_hook(
                lambda _module, _args, output: captured.append(output)
            )
        try:
            output = self.language_model(**prepared.model_kwargs)
        finally:
            if hook is not None:
                hook.remove()
        block_logits = no_tool_logit = None
        if mode.uses_memory:
            if len(captured) != 1:
                raise RuntimeError(f"expected one final hidden-state capture, found {len(captured)}")
            block_logits, no_tool_logit = self.router_head(captured[0], teacher)
        return AdaptationOutput(output.logits, prepared, block_logits, no_tool_logit)


def build_oracle_adaptation_model(base_model: nn.Module, *, seed: int, rank: int = 16,
                                  alpha: int = 32, dropout: float = 0.05) -> OracleAdaptationModel:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    language_model = get_peft_model(base_model, config)
    language_model.enable_input_require_grads()
    hidden_size = int(base_model.config.hidden_size)
    bank = MemoryRouterTokenBank(
        hidden_size, memory_tokens=8, router_tokens=8,
        initializer_range=float(base_model.config.initializer_range), seed=seed,
    )
    router = OracleRouterHead(hidden_size)
    return OracleAdaptationModel(language_model, bank, router)


def trainable_parameter_audit(model: OracleAdaptationModel) -> dict[str, Any]:
    trainable = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    invalid = [name for name in trainable if not (
        "lora_" in name or name.startswith("token_bank.") or name.startswith("router_head.")
    )]
    if invalid:
        raise ValidationError(f"unexpected trainable parameters: {invalid}")
    return {"parameters": trainable, "total": sum(trainable.values())}
