from tool_context.benchmarking import BenchmarkShape, benchmark_packed
from tool_context.packing import TokenRole


def test_exact_phase1_layouts_and_benchmark_metadata():
    shapes = (
        BenchmarkShape("4k", 4096, 512, 8, 384),
        BenchmarkShape("8k", 8192, 512, 8, 896),
        BenchmarkShape("16k", 16384, 512, 8, 1920),
        BenchmarkShape("32k", 32768, 512, 8, 3968),
    )
    assert [shape.layout().sequence_length for shape in shapes] == [4096, 8192, 16384, 32768]
    item = benchmark_packed(shapes[0].layout(), {1, 6})
    tool_tokens = [index for index, role in enumerate(item.token_role) if role == TokenRole.T]
    selected = sum(item.selected_block[index] for index in tool_tokens)
    assert selected / len(tool_tokens) == 0.25
    assert item.safe_key_index == 0
