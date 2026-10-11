"""Turning a windowed volume into a chat message MedGemma understands.

A volume is not one input. It becomes an alternating sequence of image blocks
and "SLICE n" labels, bracketed by an instruction and a question — so the model
can refer to slice positions in its answer, which is what makes the output
auditable rather than a wall of prose.

Images are inlined as base64 data URIs, the format the processor's chat template
expects.
"""
from __future__ import annotations

import base64
import io

import numpy as np

RADIOLOGIST = (
    "You are an expert radiologist analysing a contiguous block of axial CT "
    "slices. Review every slice provided below carefully before answering."
)


def encode_slice(slice_rgb: np.ndarray, image_format: str = "jpeg") -> str:
    """One uint8 RGB slice -> a base64 data URI."""
    import PIL.Image

    if slice_rgb.ndim != 3 or slice_rgb.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got shape {slice_rgb.shape}")
    if slice_rgb.dtype != np.uint8:
        raise TypeError(f"expected uint8 (window it first), got {slice_rgb.dtype}")

    with io.BytesIO() as buffer:
        with PIL.Image.fromarray(slice_rgb) as image:
            image.save(buffer, format=image_format)
        encoded = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    return f"data:image/{image_format};base64,{encoded}"


def build_messages(slices_rgb, question: str, instruction: str = RADIOLOGIST,
                   image_format: str = "jpeg") -> list[dict]:
    """Build the `messages` list for `processor.apply_chat_template`.

    Args:
        slices_rgb: iterable of `(H, W, 3)` uint8 slices, in anatomical order
        question:   what to ask about them
        instruction: the system-style framing that opens the message
    """
    slices_rgb = list(slices_rgb)
    if not slices_rgb:
        raise ValueError("no slices to send")

    content: list[dict] = [{"type": "text", "text": instruction}]
    for number, slice_rgb in enumerate(slices_rgb, 1):
        content.append({"type": "image", "image": encode_slice(slice_rgb, image_format)})
        content.append({"type": "text", "text": f"SLICE {number}"})
    content.append({"type": "text", "text": question})

    return [{"role": "user", "content": content}]


def count_images(messages: list[dict]) -> int:
    """How many image blocks a messages list carries."""
    return sum(
        1
        for message in messages
        for block in message.get("content", [])
        if block.get("type") == "image"
    )
