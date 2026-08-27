from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .distillation import TeacherDistribution, residual_kl
from ..modeling.phase3 import AdaptationOutput
from ..packing import TeacherForcedEpisode, TokenRole


@dataclass(slots=True)
class LossComponents:
    total: Tensor
    answer: Tensor
    distillation: Tensor
    router: Tensor


def router_labels(teacher: TeacherForcedEpisode, device: torch.device) -> tuple[Tensor, Tensor]:
    packed = teacher.packed
    ordered = list(dict.fromkeys(block for role, block in zip(packed.token_role, packed.block_id, strict=True)
                                 if role == TokenRole.T and block is not None))
    selected = {block for role, block, chosen in zip(
        packed.token_role, packed.block_id, packed.selected_block, strict=True,
    ) if role == TokenRole.T and chosen and block is not None}
    labels = torch.tensor([[float(block in selected) for block in ordered]], device=device)
    return labels, torch.tensor([float(not selected)], device=device)


def phase3_loss(output: AdaptationOutput, teacher: TeacherForcedEpisode,
                teacher_distribution: TeacherDistribution, *, kl_weight: float = 0.2,
                router_weight: float = 0.2, positive_weight: float = 4.0,
                no_tool_weight: float = 0.25) -> LossComponents:
    targets = output.prepared.target_token_ids
    answer = functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                      targets.reshape(-1))
    distillation = residual_kl(output.logits, teacher_distribution)
    router = output.logits.new_zeros((), dtype=torch.float32)
    if output.block_logits is not None:
        labels, no_tool = router_labels(teacher, output.block_logits.device)
        router = functional.binary_cross_entropy_with_logits(
            output.block_logits.float(), labels,
            pos_weight=torch.tensor(positive_weight, device=labels.device),
        )
        router += no_tool_weight * functional.binary_cross_entropy_with_logits(
            output.no_tool_logit.float(), no_tool,
        )
    total = answer + kl_weight * distillation + router_weight * router
    return LossComponents(total, answer, distillation, router)
