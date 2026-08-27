from __future__ import annotations

import torch

from tool_context.benchmarking import benchmark_packed
from tool_context.modeling.phase3 import OracleRouterHead
from tool_context.packing import TeacherForcedEpisode
from tool_context.schema import LayoutSpec, ValidationError
from tool_context.training.distillation import (
    TeacherCache, TeacherCacheWriter, compress_teacher_logits, residual_kl,
)
from tool_context.training.losses import router_labels


def _teacher() -> TeacherForcedEpisode:
    layout = LayoutSpec("phase3-unit", 128, 2, 128, 8, 8, 128, 128)
    packed = benchmark_packed(layout, {0})
    return TeacherForcedEpisode(packed, (layout.sequence_length - 1,), (layout.sequence_length - 2,), (3,))


def test_topk_residual_kl_is_zero_for_the_source_distribution() -> None:
    torch.manual_seed(1729); logits = torch.randn(1, 3, 31)
    teacher = compress_teacher_logits(logits, top_k=7)
    assert abs(float(residual_kl(logits, teacher))) < 1e-6
    assert torch.allclose(
        teacher.log_probabilities.float().exp().sum(-1) + teacher.residual_log_probability.exp(),
        torch.ones(3), atol=2e-3,
    )


def test_teacher_cache_round_trip_and_metadata_validation(tmp_path) -> None:
    logits = torch.randn(1, 2, 17); metadata = {"revision": "abc", "top_k": 4}
    writer = TeacherCacheWriter(tmp_path, metadata, shard_size=1)
    writer.add("episode", compress_teacher_logits(logits, 4)); writer.finish()
    cache = TeacherCache(tmp_path, metadata); restored = cache.get("episode")
    assert restored.token_ids.shape == (2, 4)
    try:
        TeacherCache(tmp_path, {"revision": "wrong"})
    except ValidationError:
        pass
    else:
        raise AssertionError("mismatched teacher metadata was accepted")


def test_router_head_and_labels_cover_blocks_and_no_tool() -> None:
    teacher = _teacher(); hidden = torch.randn(1, teacher.packed.layout.sequence_length, 32)
    block_logits, no_tool = OracleRouterHead(32, 8)(hidden, teacher)
    labels, no_tool_label = router_labels(teacher, torch.device("cpu"))
    assert block_logits.shape == labels.shape == (1, 2)
    assert no_tool.shape == no_tool_label.shape == (1,)
    assert labels.tolist() == [[1.0, 0.0]] and no_tool_label.item() == 0.0
