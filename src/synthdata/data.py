"""Oxford-IIIT Pet, the fixed course split, and photo loading.

uv run synth-data
"""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from synthdata.config import CLASSES, DATA_DIR, SPLIT_CSV, TARGET

ROLES = ("train", "val", "test")


@dataclass(frozen=True)
class Photo:
    image: str
    breed: str
    role: str

    @property
    def label(self) -> int:
        return CLASSES.index(self.breed)


def load_split() -> list[Photo]:
    with SPLIT_CSV.open(newline="") as stream:
        return [Photo(**row) for row in csv.DictReader(stream)]


def photos(role: str) -> list[Photo]:
    return [photo for photo in load_split() if photo.role == role]


def minority_training_photos() -> list[str]:
    """The 15 Egyptian Mau training photos, the only allowed generation sources."""
    return [photo.image for photo in photos("train") if photo.breed == TARGET]


def image_path(image: str) -> Path:
    return DATA_DIR / "oxford-iiit-pet" / "images" / f"{image}.jpg"


def trimap_path(image: str) -> Path:
    return DATA_DIR / "oxford-iiit-pet" / "annotations" / "trimaps" / f"{image}.png"


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def download() -> None:
    from torchvision.datasets import OxfordIIITPet

    for split in ("trainval", "test"):
        OxfordIIITPet(DATA_DIR, split=split, download=True)


def main() -> None:
    argparse.ArgumentParser(
        description="Download Oxford-IIIT Pet and check the course split."
    ).parse_args()
    split = load_split()
    missing = [photo.image for photo in split if not image_path(photo.image).exists()]
    if missing:
        print(f"Downloading Oxford-IIIT Pet to {DATA_DIR} (about 800 MB)...")
        download()
        missing = [p.image for p in split if not image_path(p.image).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} split photos are missing, e.g. {missing[0]}")

    counts = Counter((photo.role, photo.breed) for photo in split)
    print(f"{'breed':<14}" + "".join(f"{role:>7}" for role in ROLES))
    for breed in CLASSES:
        print(f"{breed:<14}" + "".join(f"{counts[(role, breed)]:>7}" for role in ROLES))
    print(f"\nAll {len(split)} photos are in place. {TARGET} is the minority class.")


if __name__ == "__main__":
    main()
