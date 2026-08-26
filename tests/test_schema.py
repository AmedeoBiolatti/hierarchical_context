import json

import pytest

from tool_context.schema import Anchor, EpisodeGraph, LayoutSpec, ToolBlock, ValidationError


def episode(**overrides):
    block = ToolBlock("a", "value alpha")
    values = {
        "episode_id": "example", "anchor": Anchor("find alpha"), "tool_blocks": (block,),
        "expected_answer": "alpha", "acceptable_support_sets": (frozenset({"a"}),),
        "requires_tool": True, "template_family": "fixture/development", "seed": 1,
    }
    values.update(overrides)
    return EpisodeGraph(**values)


def test_schema_round_trip_and_stable_hash():
    original = episode(metadata={"b": 2, "a": 1})
    restored = EpisodeGraph.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.content_hash == original.content_hash


def test_tool_hash_rejects_mismatch():
    with pytest.raises(ValidationError, match="content_hash"):
        ToolBlock("a", "content", content_hash="wrong")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tool_blocks": (ToolBlock("a", "one"), ToolBlock("a", "two"))}, "duplicate"),
        ({"acceptable_support_sets": (frozenset({"missing"}),)}, "unknown"),
        ({"acceptable_support_sets": (frozenset(),)}, "tool-required"),
        ({"requires_tool": False}, "empty support"),
    ],
)
def test_invalid_episodes_fail_early(overrides, message):
    with pytest.raises(ValidationError, match=message):
        episode(**overrides)


def test_standard_layouts_are_fixed_and_aligned():
    layouts = LayoutSpec.standard_layouts()
    assert [(x.global_capacity, x.tool_count, x.tool_capacity) for x in layouts] == [
        (512, 8, 512), (512, 8, 1024), (1024, 16, 2048),
    ]
    assert all(x.sequence_length % 128 == 0 for x in layouts)


def test_layout_rejects_unaligned_regions():
    with pytest.raises(ValidationError, match="divisible"):
        LayoutSpec("bad", 127, 2, 128)

