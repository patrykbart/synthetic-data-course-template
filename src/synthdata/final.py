"""Evaluate the frozen comparison once on the test photos (after Meeting 4).

    uv run synth-final --condition real --condition oversample \
        --condition augment --condition "text-to-image=m2-best" \
        --condition "img2img=m3-strength-0.5"

Every condition is trained on the same seeds and evaluated on the 240 test
photos. final/results.json holds the test metrics of every seed and the
difference of every condition from oversampling and augmentation with a 95%
interval, ready for your own tables and plots. The test photos are used once;
a second comparison stops unless --force is given.
"""

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime

import numpy as np

from synthdata import pools
from synthdata.config import BUDGET, CLASSES, FINAL_DIR, SEEDS
from synthdata.data import photos
from synthdata.metrics import macro_f1, target_recall
from synthdata.train import (
    MODELS,
    default_workers,
    environment,
    photo_cache,
    pick_device,
    run_seed,
    write_seed,
)

BUILT_IN = ("real", "oversample", "augment")
LOCK = FINAL_DIR / "TEST_USED.json"
RESULTS = FINAL_DIR / "results.json"
SCORES = {"egyptian_mau_recall": target_recall, "macro_f1": macro_f1}
KEYS = ("seed", "egyptian_mau_recall", "macro_f1", "accuracy", "recall")


def parse_condition(text: str) -> tuple[str, str, str | None]:
    if text in BUILT_IN:
        return text, text, None
    label, separator, pool = text.partition("=")
    if not separator or not label or not pool:
        raise SystemExit(f"--condition {text!r}: use one of {BUILT_IN} or label=pool")
    return label, "synthetic", pool


def bootstrap(labels, first, second, score, iterations=4000, seed=0):
    """Difference of mean scores with a 95% interval over seeds and test photos.

    `first` and `second` hold predictions with one row per seed; seed k of both
    conditions is paired, and so is every resampled test photo.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    by_breed = [np.flatnonzero(labels == index) for index in range(len(CLASSES))]
    count = first.shape[0]
    observed = np.mean([score(labels, row) for row in first]) - np.mean(
        [score(labels, row) for row in second]
    )
    differences = np.empty(iterations)
    for iteration in range(iterations):
        chosen = np.concatenate(
            [rng.choice(group, size=len(group)) for group in by_breed]
        )
        rows = rng.integers(0, count, size=count)
        truth = labels[chosen]
        differences[iteration] = np.mean(
            [
                score(truth, first[row, chosen]) - score(truth, second[row, chosen])
                for row in rows
            ]
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(observed), float(low), float(high)


def read_predictions(folder) -> list[int]:
    with (folder / "predictions.csv").open(newline="") as stream:
        return [CLASSES.index(row["predicted"]) for row in csv.DictReader(stream)]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen conditions and evaluate them once on the "
        "test photos. Results go to final/results.json."
    )
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument(
        "--reference",
        action="append",
        help="conditions to compare against (default: oversample and augment)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--model", choices=MODELS, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--force", action="store_true", help="use the test photos again (ask first)"
    )
    args = parser.parse_args(argv)

    conditions = [parse_condition(text) for text in args.condition]
    labels = [label for label, _, _ in conditions]
    if len(set(labels)) != len(labels):
        parser.error("condition labels must be unique")
    references = args.reference or ["oversample", "augment"]
    references = [name for name in references if name in labels]
    if not references:
        parser.error("include oversample or augment, or name a --reference")
    settings = {
        "conditions": args.condition,
        "seeds": args.seeds,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    FINAL_DIR.mkdir(exist_ok=True)
    started = None
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
        same = {key: lock.get(key) for key in settings} == settings
        if lock.get("finished") and not args.force:
            raise SystemExit(
                f"The test photos were already used on {lock['started']} "
                "(see final/results.json). Ask the instructor before using --force."
            )
        if not lock.get("finished") and not same and not args.force:
            raise SystemExit("an unfinished comparison with other settings exists")
        if same and not args.force:
            started = lock["started"]
    if args.force:
        # A new comparison starts from scratch instead of reusing old seeds.
        for label in labels:
            shutil.rmtree(FINAL_DIR / label, ignore_errors=True)
    started = started or datetime.now(UTC).isoformat(timespec="seconds")
    LOCK.write_text(json.dumps({**settings, "started": started}, indent=2) + "\n")

    device = pick_device(args.device)
    cache = photo_cache(("train", "test"))
    train_settings = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "budget": BUDGET,
    }
    results, predictions = {}, {}
    for label, method, pool in conditions:
        pool_dir = pools.resolve(pool) if pool else None
        results[label] = {
            "method": method,
            "pool": pool_dir.name if pool_dir else None,
            "seeds": [],
        }
        predictions[label] = []
        for seed in args.seeds:
            folder = FINAL_DIR / label / f"seed-{seed}"
            if not (folder / "metrics.json").exists():
                metrics, rows, added, seconds = run_seed(
                    method,
                    pool_dir,
                    seed,
                    "test",
                    train_settings,
                    device,
                    cache,
                    args.workers,
                )
                metrics = {"name": label, "seed": seed, "split": "test", **metrics}
                record = {
                    **train_settings,
                    "condition": label,
                    "method": method,
                    "pool": pool_dir.name if pool_dir else None,
                    "seed": seed,
                    "added": added,
                    "seconds": round(seconds, 1),
                    **environment(device),
                }
                write_seed(folder, metrics, rows, record)
                print(
                    f"{label} seed {seed}: Egyptian Mau recall "
                    f"{metrics['egyptian_mau_recall']:.3f}, macro-F1 "
                    f"{metrics['macro_f1']:.3f}"
                )
            metrics = json.loads((folder / "metrics.json").read_text())
            results[label]["seeds"].append({key: metrics[key] for key in KEYS})
            predictions[label].append(read_predictions(folder))

    truth = [photo.label for photo in photos("test")]
    arrays = {label: np.asarray(rows) for label, rows in predictions.items()}
    differences = []
    for label in labels:
        # A reference is compared only with the references listed before it.
        if label in references:
            compared = references[: references.index(label)]
        else:
            compared = references
        for reference in compared:
            for key, score in SCORES.items():
                difference, low, high = bootstrap(
                    truth, arrays[label], arrays[reference], score
                )
                differences.append(
                    {
                        "condition": label,
                        "compared_with": reference,
                        "metric": key,
                        "difference": round(difference, 4),
                        "interval_95": [round(low, 4), round(high, 4)],
                    }
                )

    finished = datetime.now(UTC).isoformat(timespec="seconds")
    output = {
        **settings,
        "split": "test",
        "test_photos_used": started,
        "conditions": results,
        "differences": differences,
    }
    RESULTS.write_text(json.dumps(output, indent=2) + "\n")
    LOCK.write_text(
        json.dumps({**settings, "started": started, "finished": finished}, indent=2)
        + "\n"
    )
    print(
        f"Test results of every seed and every difference with its 95% interval "
        f"are in {RESULTS}"
    )


if __name__ == "__main__":
    main()
