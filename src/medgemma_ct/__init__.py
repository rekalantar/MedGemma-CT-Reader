"""Read a 3D CT volume with MedGemma 1.5.

A vision encoder takes 2D RGB images, so a volume becomes a sequence of axial
slices — each windowed into three channels, each costing 256 vision tokens, and
capped at 85 slices to stay inside the context budget.

    from medgemma_ct import window_rgb, sample_indices, build_messages

    volume = window_rgb(volume_hu)                  # (D, H, W) -> (D, H, W, 3)
    keep = sample_indices(len(volume))              # at most 85, evenly spread
    messages = build_messages(volume[keep], "Describe the liver.")
"""
from .prompt import RADIOLOGIST, build_messages, count_images, encode_slice
from .sampling import MAX_SLICES, TOKENS_PER_SLICE, sample_indices, vision_tokens
from .windowing import WINDOWS, normalise, window_grey, window_rgb

__version__ = "0.1.0"
__all__ = [
    "MAX_SLICES",
    "RADIOLOGIST",
    "TOKENS_PER_SLICE",
    "WINDOWS",
    "build_messages",
    "count_images",
    "encode_slice",
    "normalise",
    "sample_indices",
    "vision_tokens",
    "window_grey",
    "window_rgb",
]
