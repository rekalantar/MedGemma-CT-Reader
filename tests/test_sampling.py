"""Slice selection, checked against MedGemma's own sampling formula."""
import pytest

from medgemma_ct import MAX_SLICES, TOKENS_PER_SLICE, sample_indices, vision_tokens


def test_cap_and_token_cost_match_the_technical_report():
    assert MAX_SLICES == 85
    assert TOKENS_PER_SLICE == 256
    assert vision_tokens(MAX_SLICES) == 21_760


def test_short_volume_is_kept_whole():
    assert sample_indices(12) == list(range(12))
    assert sample_indices(85) == list(range(85))


def test_long_volume_is_capped():
    picked = sample_indices(500)
    assert len(picked) == MAX_SLICES
    assert max(picked) == 499                 # the last slice is always included


def test_matches_the_reference_formula():
    """Upstream: [int(round(i / MAX * (D - 1))) for i in range(1, MAX + 1)]."""
    for depth in (86, 120, 333, 1000):
        # int(round(...)) mirrors upstream verbatim; round() alone already returns int
        reference = [int(round(i / MAX_SLICES * (depth - 1)))  # noqa: RUF046
                     for i in range(1, MAX_SLICES + 1)]
        assert sample_indices(depth) == reference, f"depth {depth}"


def test_indices_are_sorted_and_in_range():
    picked = sample_indices(400)
    assert picked == sorted(picked)
    assert all(0 <= i < 400 for i in picked)


def test_spacing_is_even():
    picked = sample_indices(850)
    gaps = {picked[i + 1] - picked[i] for i in range(len(picked) - 1)}
    assert gaps <= {9, 10}                     # evenly spread, +/- rounding


def test_rejects_nonsense():
    with pytest.raises(ValueError):
        sample_indices(0)
    with pytest.raises(ValueError):
        sample_indices(10, max_slices=0)
