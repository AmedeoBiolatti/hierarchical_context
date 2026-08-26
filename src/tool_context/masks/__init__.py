from .reference import MaskOptions, build_reference_mask, visible

__all__ = ["MaskOptions", "build_reference_mask", "visible"]
"""Attention-mask backends.

Import ``tool_context.masks.flex`` explicitly so the Phase 0 reference backend
remains usable without installing the optional PyTorch kernel dependencies.
"""
