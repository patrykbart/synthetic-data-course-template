"""Evaluate the saved models of a run on the test photos.

    uv run synth-test --run m1-real
    uv run synth-test --run m1-real --seeds 0

synth-train saves the model of every seed in runs/<name>/seed-K/model.pt. This
command evaluates those models on the 240 test photos. Every seed folder gets
test-metrics.json and test-predictions.csv, and runs/<name>/test-results.json
holds the test metrics of every seed in the same format as results.json, ready
for your own tables and plots. Test a run only after choosing it on the
validation results.
"""

import argparse
import json

from synthdata.config import RUNS_DIR
from synthdata.train import (
    collect_seeds,
    evaluate,
    load_checkpoint,
    photo_cache,
    pick_device,
    write_predictions,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved models of a run on the test photos. "
        "Results go to runs/<name>/test-results.json."
    )
    parser.add_argument("--run", required=True, help="run name in runs/, e.g. m2-base")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="seeds to test (default: every seed)"
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    args = parser.parse_args(argv)

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "results.json").exists():
        parser.error(f"{run_dir} has no results.json; train the run with synth-train")
    results = json.loads((run_dir / "results.json").read_text())
    seeds = args.seeds or [entry["seed"] for entry in results["seeds"]]
    checkpoints = [run_dir / f"seed-{seed}" / "model.pt" for seed in seeds]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        parser.error(
            f"no saved model: {', '.join(missing)}. Models are not committed, so "
            "test on the computer that trained the run, or train it again under "
            "a new --name"
        )

    device = pick_device(args.device)
    cache = photo_cache(("test",))
    print(f"{args.run}: {len(seeds)} saved models on the test photos, {device}")
    for seed, path in zip(seeds, checkpoints, strict=True):
        metrics, rows = evaluate(load_checkpoint(path, device), "test", device, cache)
        metrics = {"name": args.run, "seed": seed, "split": "test", **metrics}
        folder = path.parent
        (folder / "test-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        write_predictions(folder / "test-predictions.csv", rows)
        print(
            f"  seed {seed}: Egyptian Mau recall "
            f"{metrics['egyptian_mau_recall']:.3f}, macro-F1 "
            f"{metrics['macro_f1']:.3f}"
        )

    output = {
        **results,
        "split": "test",
        "seeds": collect_seeds(run_dir, "test-metrics.json"),
    }
    path = run_dir / "test-results.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Test results of every seed are in {path}")


if __name__ == "__main__":
    main()
