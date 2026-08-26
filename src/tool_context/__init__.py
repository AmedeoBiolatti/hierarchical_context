"""Backend-neutral contracts for tool-context attention experiments."""

from .packing import ByteTokenizer, PackedEpisode, TokenRole, pack_episode
from .schema import Anchor, CacheSemantics, EpisodeGraph, LayoutSpec, ToolBlock

__all__ = [
    "Anchor",
    "ByteTokenizer",
    "CacheSemantics",
    "EpisodeGraph",
    "LayoutSpec",
    "PackedEpisode",
    "TokenRole",
    "ToolBlock",
    "pack_episode",
]

