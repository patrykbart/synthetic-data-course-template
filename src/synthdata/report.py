"""Summarise runs as a table and a plot.

    uv run synth-report runs/m1-real runs/m1-oversample runs/m1-augment \
        --out evidence/m1-baselines
    uv run synth-report runs/m3-strength-0.3 runs/m3-strength-0.5 \
        runs/m3-strength-0.7 --x 0.3 0.5 0.7 --xlabel strength --out evidence/m3

Writes <out>.png, <out>.csv, and <out>.md (a table to paste into EXPERIMENTS.md).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import EVIDENCE_DIR, RUNS_DIR

METRICS = {"egyptian_mau_recall": "Egyptian Mau recall", "macro_f1": "Macro-F1"}
POINT = "#9aa5b1"
MEAN = "#1f4e79"


def resolve(run: str) -> Path:
    path = Path(run)
    if not path.exists():
        path = RUNS_DIR / run
    if not list(path.glob("seed-*/metrics.json")):
        raise SystemExit(f"no finished seeds in {path}")
    return path


def load(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("seed-*/metrics.json"))
    return [json.loads(path.read_text()) for path in files]


def spread(values) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def table(runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for label, results in runs.items():
        row = {"run": label, "seeds": len(results)}
        for key in (*METRICS, "accuracy"):
            values = [result[key] for result in results]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_sd"] = spread(values)
        rows.append(row)
    return pd.DataFrame(rows)


def markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| Run | Seeds | Egyptian Mau recall | Macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.run} | {row.seeds} "
            f"| {row.egyptian_mau_recall_mean:.3f} ± {row.egyptian_mau_recall_sd:.3f} "
            f"| {row.macro_f1_mean:.3f} ± {row.macro_f1_sd:.3f} |"
        )
    return "\n".join(lines) + "\n"


def plot(runs: dict[str, list[dict]], path: Path, x=None, xlabel=None, title=None):
    """One panel per metric: grey dots are seeds, blue diamonds are means ± sd."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(runs)
    positions = list(x) if x else list(range(len(labels)))
    width = max(6.0, 1.15 * len(labels) + 3.0) if not x else 8.0
    figure, axes = plt.subplots(1, 2, figsize=(width, 3.6), constrained_layout=True)
    step = (max(positions) - min(positions)) / 40 if x and len(positions) > 1 else 0.06
    for axis, (key, name) in zip(axes, METRICS.items(), strict=True):
        means = []
        for position, label in zip(positions, labels, strict=True):
            values = [result[key] for result in runs[label]]
            offsets = (np.arange(len(values)) - (len(values) - 1) / 2) * step
            axis.scatter(position + offsets, values, s=16, color=POINT, zorder=2)
            means.append(np.mean(values))
            axis.errorbar(
                position,
                means[-1],
                yerr=spread(values),
                fmt="D",
                color=MEAN,
                markersize=6,
                capsize=4,
                zorder=3,
            )
        if x:
            axis.plot(positions, means, color=MEAN, linewidth=1.4, zorder=1)
            axis.set_xticks(positions)
            axis.set_xlabel(xlabel or "")
        else:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20 if len(labels) > 3 else 0)
            axis.set_xlim(-0.6, len(labels) - 0.4)
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    if title:
        figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare runs in a table and a plot.")
    parser.add_argument("runs", nargs="+", help="run folders or names in runs/")
    parser.add_argument("--out", help="output path without extension")
    parser.add_argument("--labels", nargs="+", help="display names, one per run")
    parser.add_argument("--x", type=float, nargs="+", help="numeric x value per run")
    parser.add_argument("--xlabel", help="x-axis label for --x")
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    folders = [resolve(run) for run in args.runs]
    labels = args.labels or [folder.name for folder in folders]
    for option in ("labels", "x"):
        values = getattr(args, option)
        if values and len(values) != len(folders):
            parser.error(f"--{option} needs one value per run")
    runs = {label: load(folder) for label, folder in zip(labels, folders, strict=True)}

    out = Path(args.out) if args.out else EVIDENCE_DIR / folders[0].name
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = table(runs)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(markdown(frame))
    plot(runs, out.with_suffix(".png"), args.x, args.xlabel, args.title)
    print(markdown(frame))
    print(f"Saved {out.with_suffix('.png')}, .csv and .md")


if __name__ == "__main__":
    main()
