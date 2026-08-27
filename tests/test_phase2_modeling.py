from __future__ import annotations

import torch

from tool_context.benchmarking import BenchmarkShape, benchmark_packed
from tool_context.modeling.qwen3_block import MemoryRouterTokenBank, Phase2Mode
from tool_context.packing import TeacherForcedEpisode, TokenRole
from tool_context.prefill import _teacher
from tool_context.schema import LayoutSpec


def _episode() -> TeacherForcedEpisode:
    layout = LayoutSpec("unit", 128, 2, 128, 8, 8, 128, 128)
    packed = benchmark_packed(layout, {0})
    return TeacherForcedEpisode(packed, (layout.sequence_length - 1,), (layout.sequence_length - 2,), (3,))


def test_token_bank_is_seeded_and_only_changes_memory_router_positions() -> None:
    teacher = _episode(); base = torch.zeros(1, teacher.packed.layout.sequence_length, 16)
    left = MemoryRouterTokenBank(16, seed=1729); right = MemoryRouterTokenBank(16, seed=1729)
    assert left.checksums() == right.checksums()
    result = left.apply(base, teacher)
    changed = result.ne(base).any(dim=-1).squeeze(0)
    expected = torch.tensor([role in (TokenRole.M, TokenRole.R) for role in teacher.packed.token_role])
    assert torch.equal(changed, expected)


def test_prefill_static_disables_memory_and_router_and_oracle_selects_two() -> None:
    shape = BenchmarkShape("unit", 768, 128, 2, 128, answer_capacity=128)
    static = _teacher(shape, Phase2Mode.BLOCK_STATIC).packed
    assert TokenRole.M not in static.token_role and TokenRole.R not in static.token_role
    oracle = _teacher(shape, Phase2Mode.BLOCK_ORACLE).packed
    selected = {block for role, block, selected in zip(
        oracle.token_role, oracle.block_id, oracle.selected_block, strict=True,
    ) if role == TokenRole.T and selected}
    assert selected == {"tool-0", "tool-1"}
