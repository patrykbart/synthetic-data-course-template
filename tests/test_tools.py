import json
import random
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from synthdata.generate import parse_args, pool_settings
from synthdata.inputs import keep_cat, outpaint_inputs, paste_back
from synthdata.metrics import macro_f1, summary, target_recall
from synthdata.pools import draw, read_manifest
from synthdata.train import parse_tags, write_results

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


def test_tags_keep_numbers_as_numbers():
    tags = parse_tags(["mode=ip-adapter", "scale=0.5", "steps=30"])
    assert tags == {"mode": "ip-adapter", "scale": 0.5, "steps": 30}
    with pytest.raises(ValueError):
        parse_tags(["scale"])


def test_results_keep_every_seed_and_earlier_tags(tmp_path):
    for seed, recall in [(1, 0.7), (0, 0.5)]:
        folder = tmp_path / f"seed-{seed}"
        folder.mkdir()
        metrics = {
            "seed": seed,
            "egyptian_mau_recall": recall,
            "macro_f1": 0.9,
            "accuracy": 0.9,
            "recall": {},
            "confusion": [],
        }
        (folder / "metrics.json").write_text(json.dumps(metrics))
    settings = {"method": "synthetic", "pool": "m3-ip-0.5"}
    write_results(tmp_path, "m3-ip-0.5", settings, {"mode": "ip-adapter"})
    write_results(tmp_path, "m3-ip-0.5", settings, {"scale": 0.5})
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["tags"] == {"mode": "ip-adapter", "scale": 0.5}
    assert [entry["seed"] for entry in results["seeds"]] == [0, 1]
    assert "confusion" not in results["seeds"][0]


def test_source_limits_generation_to_chosen_training_photos():
    args = parse_args(
        [
            "--name",
            "x",
            "--mode",
            "img2img",
            "--strength",
            "0.5",
            "--source",
            "Egyptian_Mau_117",
            "--source",
            "Egyptian_Mau_104",
        ]
    )
    assert pool_settings(args)["sources"] == ["Egyptian_Mau_104", "Egyptian_Mau_117"]


def test_source_refuses_photos_outside_the_minority_training_set():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--name",
                "x",
                "--mode",
                "img2img",
                "--strength",
                "0.5",
                "--source",
                "Abyssinian_1",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(["--name", "x", "--source", "Egyptian_Mau_117"])
