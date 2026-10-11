"""Message construction for the chat template."""
import base64

import numpy as np
import pytest

from medgemma_ct import build_messages, count_images, encode_slice, window_rgb


def _slices(n=3, size=16):
    rng = np.random.default_rng(0)
    return window_rgb(rng.integers(-500, 500, (n, size, size), dtype=np.int16))


def test_encode_produces_a_data_uri():
    uri = encode_slice(_slices(1)[0])
    assert uri.startswith("data:image/jpeg;base64,")
    base64.b64decode(uri.split(",", 1)[1])          # must be valid base64


def test_encode_rejects_unwindowed_input():
    with pytest.raises(TypeError):
        encode_slice(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        encode_slice(np.zeros((8, 8), dtype=np.uint8))


def test_message_structure():
    """instruction, then (image, SLICE n) per slice, then the question."""
    messages = build_messages(_slices(3), "Describe the liver.")
    assert len(messages) == 1
    content = messages[0]["content"]

    assert content[0]["type"] == "text"
    assert content[-1]["text"] == "Describe the liver."
    assert len(content) == 2 + 3 * 2                 # instruction + 3 pairs + question

    labels = [b["text"] for b in content if b.get("type") == "text"]
    assert "SLICE 1" in labels and "SLICE 3" in labels


def test_slice_labels_are_one_based_and_ordered():
    content = build_messages(_slices(5), "?")[0]["content"]
    # image blocks carry no "text" key, hence .get
    labels = [b.get("text", "") for b in content]
    labels = [t for t in labels if t.startswith("SLICE ")]
    assert labels == [f"SLICE {i}" for i in range(1, 6)]


def test_count_images():
    assert count_images(build_messages(_slices(7), "?")) == 7


def test_empty_rejected():
    with pytest.raises(ValueError):
        build_messages([], "?")
