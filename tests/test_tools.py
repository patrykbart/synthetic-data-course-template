import json
import random
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from synthdata.inputs import keep_cat, outpaint_inputs, paste_back
from synthdata.metrics import macro_f1, summary, target_recall
from synthdata.pools import draw, read_manifest
from synthdata.report import table

RED = (200, 10, 10)
BLUE = (0, 0, 255)


def records(count: int) -> list[dict]:
    return [
        {"sample_id": f"pool-{index:04d}", "source": f"photo-{index % 15}"}
        for index in range(count)
    ]


def test_draw_spreads_images_evenly_over_source_photos():
    chosen = draw(records(110), 55, random.Random(0))
    counts = Counter(record["source"] for record in chosen)
    assert len({record["sample_id"] for record in chosen}) == 55
    assert max(counts.values()) - min(counts.values()) <= 1


def test_draw_refuses_a_pool_smaller_than_the_budget():
    with pytest.raises(SystemExit):
        draw(records(10), 55, random.Random(0))


def test_manifest_keeps_the_latest_attempt_per_sample(tmp_path):
    lines = [
        {"sample_id": "pool-0001", "status": "safety_blocked"},
        {"sample_id": "pool-0000", "status": "ok"},
        {"sample_id": "pool-0001", "status": "ok"},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    statuses = [record["status"] for record in read_manifest(tmp_path)]
    assert statuses == ["ok", "ok"]


def test_outpainting_keeps_the_photo_in_the_centre():
    photo = Image.new("RGB", (512, 512), RED)
    _, mask, small, offset = outpaint_inputs(photo, 2.0)
    assert small.size == (256, 256)
    assert offset == 128
    assert np.asarray(mask)[0, 0] == 255
    assert np.asarray(mask)[256, 256] == 0
    result = paste_back(Image.new("RGB", (512, 512), BLUE), small, offset)
    assert result.getpixel((256, 256)) == RED
    assert result.getpixel((5, 5)) == BLUE


def test_inpainting_keeps_the_original_cat_pixels():
    photo = Image.new("RGB", (512, 512), RED)
    generated = Image.new("RGB", (512, 512), BLUE)
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, (0, 0, 256, 512))
    result = keep_cat(photo, generated, mask)
    assert result.getpixel((500, 256)) == RED
    assert result.getpixel((10, 256)) == BLUE


def test_metrics_on_a_small_example():
    labels = [4, 4, 4, 4, 0, 1]
    predictions = [4, 4, 0, 0, 0, 1]
    assert target_recall(labels, predictions) == 0.5
    assert 0 < macro_f1(labels, predictions) < 1
    assert summary(labels, predictions)["accuracy"] == pytest.approx(4 / 6)


def test_report_table_averages_seeds():
    runs = {
        "a": [
            {"egyptian_mau_recall": 0.5, "macro_f1": 0.8, "accuracy": 0.8},
            {"egyptian_mau_recall": 0.7, "macro_f1": 0.9, "accuracy": 0.9},
        ]
    }
    row = table(runs).iloc[0]
    assert row["egyptian_mau_recall_mean"] == pytest.approx(0.6)
    assert row["seeds"] == 2
