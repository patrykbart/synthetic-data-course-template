"""Contact sheets: a fixed-order grid of pool images for visual inspection."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from synthdata.data import open_rgb
from synthdata.inputs import background_mask, overlay, source_photo

TILE = 192
CAPTION = 18


def thumbnail(image: Image.Image) -> Image.Image:
    return ImageOps.fit(image, (TILE, TILE))


def source_tile(record: dict) -> Image.Image:
    photo = source_photo(record["source"])
    if record["mode"] == "inpaint":
        return overlay(photo, background_mask(record["source"]))
    return photo


def contact_sheet(pool_dir: Path, records: list[dict], limit: int = 24) -> Path:
    """Show the first `limit` attempts; image-conditioned pools show the source."""
    shown = records[:limit]
    paired = any(record.get("source") for record in shown)
    columns = 4 if paired else 6
    cell = TILE * (2 if paired else 1)
    rows = max(1, math.ceil(len(shown) / columns))
    sheet = Image.new("RGB", (columns * cell, rows * (TILE + CAPTION)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(shown):
        x = (index % columns) * cell
        y = (index // columns) * (TILE + CAPTION)
        if paired:
            sheet.paste(thumbnail(source_tile(record)), (x, y))
            x_result = x + TILE
        else:
            x_result = x
        sheet.paste(thumbnail(open_rgb(pool_dir / record["file"])), (x_result, y))
        caption = f"#{index}  seed {record['seed']}"
        if record["status"] != "ok":
            caption += f"  {record['status'].upper()}"
        draw.text((x + 4, y + TILE + 3), caption, fill="black")
    path = pool_dir / "contact-sheet.jpg"
    sheet.save(path, quality=90)
    return path
