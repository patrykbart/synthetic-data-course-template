"""Preparing training photos for image-conditioned generation (no GPU needed)."""

import numpy as np
from PIL import Image, ImageFilter

from synthdata.data import image_path, open_rgb, trimap_path

SIZE = 512
# Width in pixels of the photo border that outpainting may blend with new content.
OVERLAP = 8
TRIMAP_PET = 1


def square(image: Image.Image, resample=Image.Resampling.LANCZOS) -> Image.Image:
    """Centre-crop to a square and resize to the generation size."""
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    box = (left, top, left + side, top + side)
    return image.crop(box).resize((SIZE, SIZE), resample)


def source_photo(image: str) -> Image.Image:
    return square(open_rgb(image_path(image)))


def background_mask(image: str) -> Image.Image:
    """White where the model may paint (background and fur edge), black on the cat."""
    with Image.open(trimap_path(image)) as trimap:
        trimap = square(trimap.convert("L"), Image.Resampling.NEAREST)
    pet = np.asarray(trimap) == TRIMAP_PET
    return Image.fromarray(np.where(pet, 0, 255).astype(np.uint8), mode="L")


def keep_cat(photo: Image.Image, generated: Image.Image, mask: Image.Image):
    """Put the original cat pixels back so only the background is synthetic."""
    soft = mask.filter(ImageFilter.GaussianBlur(2))
    return Image.composite(generated, photo, soft)


def outpaint_inputs(photo: Image.Image, scale: float):
    """Shrink the photo into the centre of a larger canvas and mask the border."""
    inner = round(SIZE / scale)
    small = photo.resize((inner, inner), Image.Resampling.LANCZOS)
    offset = (SIZE - inner) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), (127, 127, 127))
    canvas.paste(small, (offset, offset))
    mask = Image.new("L", (SIZE, SIZE), 255)
    keep = (
        offset + OVERLAP,
        offset + OVERLAP,
        offset + inner - OVERLAP,
        offset + inner - OVERLAP,
    )
    mask.paste(0, keep)
    return canvas, mask, small, offset


def paste_back(generated: Image.Image, small: Image.Image, offset: int):
    """Restore the original photo inside the outpainted frame."""
    inner = small.width
    result = generated.copy()
    crop = small.crop((OVERLAP, OVERLAP, inner - OVERLAP, inner - OVERLAP))
    result.paste(crop, (offset + OVERLAP, offset + OVERLAP))
    return result


def overlay(photo: Image.Image, mask: Image.Image) -> Image.Image:
    """Tint the area the model may repaint, for contact sheets."""
    tinted = Image.blend(photo, Image.new("RGB", photo.size, (220, 40, 40)), 0.45)
    return Image.composite(tinted, photo, mask)
