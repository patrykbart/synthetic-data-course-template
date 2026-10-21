"""Meeting 1: checks for oversample() and augment() in src/synthdata/balance.py.

These four tests fail until you write both functions.
"""

import random
from collections import Counter

import numpy as np
from PIL import Image

from synthdata.balance import augment, oversample

PHOTOS = [f"Egyptian_Mau_{index}" for index in range(15)]


def test_oversample_returns_budget_names_from_the_minority_photos():
    chosen = oversample(list(PHOTOS), 55, random.Random(0))
    assert len(chosen) == 55
    assert set(chosen) <= set(PHOTOS)


def test_oversample_repeats_photos_evenly():
    counts = Counter(oversample(list(PHOTOS), 55, random.Random(0)))
    assert set(counts) == set(PHOTOS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_oversample_is_reproducible_from_the_rng():
    first = oversample(list(PHOTOS), 55, random.Random(3))
    assert first == oversample(list(PHOTOS), 55, random.Random(3))


def test_augment_returns_a_changed_rgb_image_and_keeps_the_input():
    pixels = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    photo = Image.fromarray(pixels)
    outputs = [augment(photo) for _ in range(5)]
    assert all(isinstance(output, Image.Image) for output in outputs)
    assert all(output.mode == "RGB" for output in outputs)
    assert np.array_equal(np.asarray(photo), pixels), "augment() changed its input"
    assert any(
        output.size != photo.size or not np.array_equal(np.asarray(output), pixels)
        for output in outputs
    )
