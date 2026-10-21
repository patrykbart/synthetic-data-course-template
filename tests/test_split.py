from collections import Counter

from synthdata.config import CLASSES, TARGET
from synthdata.data import load_split, minority_training_photos


def test_split_has_the_course_counts():
    counts = Counter((photo.role, photo.breed) for photo in load_split())
    for breed in CLASSES:
        assert counts[("train", breed)] == (15 if breed == TARGET else 70)
        assert counts[("val", breed)] == 20
        assert counts[("test", breed)] == 30


def test_no_photo_has_two_roles():
    names = [photo.image for photo in load_split()]
    assert len(names) == len(set(names))


def test_file_names_match_breeds():
    for photo in load_split():
        assert photo.image.rsplit("_", 1)[0].replace("_", " ") == photo.breed


def test_generation_sources_are_training_photos_only():
    training = {photo.image for photo in load_split() if photo.role == "train"}
    sources = minority_training_photos()
    assert len(sources) == 15
    assert set(sources) <= training
