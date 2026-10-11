"""Pack three Hounsfield windows into the three RGB channels.

A CT stores thousands of Hounsfield values; a vision encoder trained on natural
images accepts 256 levels per channel. The usual fix is to pick one window and
copy it across R, G and B — which spends three channels carrying one view.

MedGemma does something better: each channel carries a *different* window, so a
single RGB image shows bone and lung, soft tissue, and brain/vascular detail at
once. The windows below are the ones MedGemma's own CT notebook uses.
"""
from __future__ import annotations

import numpy as np

#: (min_hu, max_hu) per channel — red, green, blue.
WINDOWS: tuple[tuple[int, int], ...] = (
    (-1024, 1024),   # wide: morphological boundaries across all anatomy
    (-135, 215),     # mediastinum / soft tissue
    (0, 80),         # brain and vascular detail
)


def normalise(volume: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip a Hounsfield array to [low, high] and rescale it to 0-255 as float."""
    if high <= low:
        raise ValueError(f"window must have high > low, got ({low}, {high})")
    out = np.clip(volume, low, high).astype(np.float32)
    out -= low
    out /= high - low
    return out * 255.0


def window_rgb(volume_hu: np.ndarray, windows=WINDOWS) -> np.ndarray:
    """Hounsfield array -> uint8 RGB with one window per channel.

    Accepts a single slice `(H, W)` or a volume `(D, H, W)`; the channel axis is
    appended, giving `(H, W, 3)` or `(D, H, W, 3)`.
    """
    if len(windows) != 3:
        raise ValueError(f"need exactly three windows, got {len(windows)}")
    stacked = np.stack([normalise(volume_hu, lo, hi) for lo, hi in windows], axis=-1)
    return np.round(stacked, 0).astype(np.uint8)


def window_grey(volume_hu: np.ndarray, width: float = 400, level: float = 40) -> np.ndarray:
    """Single-window greyscale, for comparison. This is what most pipelines do."""
    return np.round(
        normalise(volume_hu, level - width / 2, level + width / 2), 0
    ).astype(np.uint8)
