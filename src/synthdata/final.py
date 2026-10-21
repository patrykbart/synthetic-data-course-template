"""Evaluate the frozen comparison once on the test photos (after Meeting 4).

    uv run synth-final --condition real --condition oversample \
        --condition augment --condition "text-to-image=m2-best" \
        --condition "img2img=m3-strength-0.5"

Every condition is trained on the same seeds and evaluated on the 240 test
photos. Results go to final/: summary.md, results.csv, comparison.png, and
differences.png. The test photos are used once; a second comparison stops
unless --force is given.
"""

import argparse
import csv
import json
from datetime import UTC, datetime

import numpy as np

from synthdata import pools
from synthdata.config import BUDGET, CLASSES, FINAL_DIR, SEEDS
from synthdata.data import photos
from synthdata.metrics import macro_f1, target_recall
from synthdata.report import METRICS, plot, spread
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
SCORES = {"egyptian_mau_recall": target_recall, "macro_f1": macro_f1}


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


def plot_differences(rows: list[dict], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    references = list(dict.fromkeys(row["reference"] for row in rows))
    colours = ["#b45309", "#15803d", "#1f4e79", "#7c3aed"]
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 0.5 * len(conditions) + 1.9), layout="constrained"
    )
    for axis, (key, title) in zip(axes, METRICS.items(), strict=True):
        for number, reference in enumerate(references):
            shift = (number - (len(references) - 1) / 2) * 0.22
            colour = colours[number % len(colours)]
            for row in rows:
                if row["reference"] != reference:
                    continue
                position = conditions.index(row["condition"]) + shift
                mean, low, high = row[key]
                axis.plot([low, high], [position, position], color=colour, lw=2)
                axis.plot(mean, position, "D", color=colour, ms=5)
            axis.plot([], [], "-D", color=colour, label=f"compared with {reference}")
        axis.axvline(0, color="#888888", lw=1, ls="--")
        axis.set_yticks(range(len(conditions)))
        axis.set_yticklabels(conditions if key == "egyptian_mau_recall" else [])
        axis.set_ylim(len(conditions) - 0.5, -0.5)
        axis.set_title(f"Difference in {title}")
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), frameon=False
    )
    figure.suptitle("Differences on the test photos with 95% intervals")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def summary_markdown(results, differences, settings, started) -> str:
    lines = [
        "# Final comparison on the test photos",
        "",
        f"Test photos used on {started}. Seeds: {settings['seeds']}. "
        f"Model: {settings['model']}, {settings['epochs']} epochs.",
        "",
        "| Condition | Egyptian Mau recall | Macro-F1 |",
        "| --- | ---: | ---: |",
    ]
    for label, metrics in results.items():
        cells = [
            f"{np.mean([m[key] for m in metrics]):.3f} ± "
            f"{spread([m[key] for m in metrics]):.3f}"
            for key in METRICS
        ]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## Differences with 95% intervals over seeds and test photos",
        "",
        "| Condition | Compared with | Egyptian Mau recall | Macro-F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in differences:
        cells = [
            f"{row[key][0]:+.3f} [{row[key][1]:+.3f}, {row[key][2]:+.3f}]"
            for key in METRICS
        ]
        lines.append(
            f"| {row['condition']} | {row['reference']} | {cells[0]} | {cells[1]} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen conditions and evaluate them once on the "
        "test photos."
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
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
        same = {key: lock.get(key) for key in settings} == settings
        if lock.get("finished") and not args.force:
            raise SystemExit(
                f"The test photos were already used on {lock['started']} "
                "(see final/summary.md). Ask the instructor before using --force."
            )
        if not lock.get("finished") and not same and not args.force:
            raise SystemExit("an unfinished comparison with other settings exists")
        started = lock["started"] if same and not args.force else None
    else:
        started = None
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
        results[label], predictions[label] = [], []
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
            results[label].append(json.loads((folder / "metrics.json").read_text()))
            predictions[label].append(read_predictions(folder))

    truth = [photo.label for photo in photos("test")]
    arrays = {label: np.asarray(rows) for label, rows in predictions.items()}
    differences = []
    for label in labels:
        # A reference is compared only with the references listed before it.
        compared = (
            references[: references.index(label)] if label in references else references
        )
        for reference in compared:
            row = {"condition": label, "reference": reference}
            for key, score in SCORES.items():
                row[key] = bootstrap(truth, arrays[label], arrays[reference], score)
            differences.append(row)

    with (FINAL_DIR / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "seed", *METRICS, "accuracy"])
        for label, metrics in results.items():
            for item in metrics:
                writer.writerow(
                    [label, item["seed"], *(item[k] for k in METRICS), item["accuracy"]]
                )
    text = summary_markdown(results, differences, settings, started)
    (FINAL_DIR / "summary.md").write_text(text)
    plot(results, FINAL_DIR / "comparison.png", title="Test photos")
    if differences:
        plot_differences(differences, FINAL_DIR / "differences.png")
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    LOCK.write_text(
        json.dumps({**settings, "started": started, "finished": finished}, indent=2)
        + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
