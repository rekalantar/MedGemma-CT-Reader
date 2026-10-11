"""Choosing which slices to send.

Each slice costs 256 vision tokens, and MedGemma caps a volume at 85 slices —
21,760 vision tokens, which has to fit alongside the question and the answer
inside a 32K context. A longer volume is sampled evenly along z rather than
truncated, so the model still sees the whole anatomy, just more coarsely.
"""
from __future__ import annotations

#: MedGemma's documented cap: 85 slices x 256 tokens = 21,760 vision tokens.
MAX_SLICES = 85
TOKENS_PER_SLICE = 256


def sample_indices(depth: int, max_slices: int = MAX_SLICES) -> list[int]:
    """Indices of up to `max_slices` slices spread evenly through a volume."""
    if depth <= 0:
        raise ValueError("volume has no slices")
    if max_slices <= 0:
        raise ValueError("max_slices must be positive")
    if depth <= max_slices:
        return list(range(depth))
    return [round(i / max_slices * (depth - 1)) for i in range(1, max_slices + 1)]


def vision_tokens(n_slices: int) -> int:
    """Vision tokens a given number of slices will occupy."""
    return n_slices * TOKENS_PER_SLICE
