"""Fine-tune a pretrained CNN on the course split and evaluate it.

    uv run synth-train --name m1-real --method real
    uv run synth-train --name m1-oversample --method oversample
    uv run synth-train --name m3-strength-0.5 --method synthetic \
        --pool m3-strength-0.5 --tag mode=img2img --tag strength=0.5

Every run trains one model per seed and evaluates it on the validation photos.
runs/<name>/results.json holds the metrics of every seed and the --tag values,
ready for your own tables and plots. Each seed also saves its model in
seed-K/model.pt, which `synth-test` evaluates on the test photos after Meeting 4.
"""

import argparse
import csv
import json
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from synthdata import balance, pools
from synthdata.config import BUDGET, CLASSES, RUNS_DIR, SEEDS, TARGET_INDEX
from synthdata.data import image_path, minority_training_photos, open_rgb, photos
from synthdata.metrics import summary

METHODS = ("real", "oversample", "augment", "synthetic")
MODELS = ("efficientnet_b0", "resnet18")
SHORT_SIDE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(SHORT_SIDE),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def load_small(path: Path) -> Image.Image:
    """Decode once and keep a small copy in memory for the whole run."""
    image = open_rgb(path)
    scale = SHORT_SIDE / min(image.size)
    if scale < 1:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


class TrainingSet(Dataset):
    def __init__(self, entries: list[tuple[Image.Image, int, str]]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        image, label, kind = self.entries[index]
        if kind == "augment":
            image = balance.augment(image.copy())
        return TRAIN_TRANSFORM(image), label


class EvaluationSet(Dataset):
    def __init__(self, images: list[Image.Image]):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return EVAL_TRANSFORM(self.images[index])


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def call_student(function, *args):
    try:
        return function(*args)
    except NotImplementedError as error:
        raise SystemExit(f"{function.__name__}() is not written yet: {error}") from None


def even_order(names: list[str], budget: int, rng: random.Random) -> list[str]:
    order, chosen = list(names), []
    while len(chosen) < budget:
        rng.shuffle(order)
        chosen.extend(order)
    return chosen[:budget]


def added_examples(method, pool_dir, seed, budget, cache):
    """The extra Egyptian Mau examples a method adds, and their names."""
    minority = minority_training_photos()
    rng = random.Random(seed)
    if method == "real":
        return [], []
    if method == "oversample":
        chosen = call_student(balance.oversample, list(minority), budget, rng)
        if (
            not isinstance(chosen, list)
            or len(chosen) != budget
            or not set(chosen) <= set(minority)
        ):
            raise SystemExit(
                f"oversample() must return a list of {budget} names taken from "
                "the Egyptian Mau training photos"
            )
        return [(cache[name], TARGET_INDEX, "oversample") for name in chosen], chosen
    if method == "augment":
        example = call_student(balance.augment, cache[minority[0]].copy())
        if not isinstance(example, Image.Image):
            raise SystemExit("augment() must return a PIL image")
        chosen = even_order(minority, budget, rng)
        return [(cache[name], TARGET_INDEX, "augment") for name in chosen], chosen
    records = pools.draw(pools.usable(pool_dir), budget, rng)
    entries = [
        (load_small(pool_dir / record["file"]), TARGET_INDEX, "synthetic")
        for record in records
    ]
    return entries, [record["sample_id"] for record in records]


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    else:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def save_checkpoint(path: Path, name: str, model: nn.Module) -> None:
    torch.save({"model": name, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    """A model saved by synth-train, ready for evaluation."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def fit(entries, seed: int, settings: dict, device, workers: int) -> nn.Module:
    seed_everything(seed)
    model = build_model(settings["model"]).to(device)
    epochs = settings["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(
        TrainingSet(entries),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    return model


@torch.inference_mode()
def predict(model: nn.Module, images, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    for batch in DataLoader(EvaluationSet(images), batch_size=64):
        with torch.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
            logits = model(batch.to(device))
        probabilities.append(logits.float().softmax(dim=1).cpu())
    return torch.cat(probabilities).numpy()


def photo_cache(roles: tuple[str, ...]) -> dict[str, Image.Image]:
    wanted = [photo for role in roles for photo in photos(role)]
    return {photo.image: load_small(image_path(photo.image)) for photo in wanted}


def evaluate(model: nn.Module, role: str, device, cache) -> tuple[dict, list]:
    """Metrics and the prediction for every photo of `role`."""
    evaluated = photos(role)
    probabilities = predict(model, [cache[photo.image] for photo in evaluated], device)
    predicted = probabilities.argmax(axis=1)
    metrics = summary([photo.label for photo in evaluated], predicted)
    rows = [
        {
            "image": photo.image,
            "breed": photo.breed,
            "predicted": CLASSES[int(label)],
            "p_egyptian_mau": round(float(row[TARGET_INDEX]), 5),
        }
        for photo, label, row in zip(evaluated, predicted, probabilities, strict=True)
    ]
    return metrics, rows


def run_seed(method, pool_dir, seed, settings, device, cache, workers):
    """Train one model and evaluate it on the validation photos."""
    seed_everything(seed)
    real = [(cache[photo.image], photo.label, "real") for photo in photos("train")]
    extra, added = added_examples(method, pool_dir, seed, settings["budget"], cache)
    started = time.perf_counter()
    model = fit(real + extra, seed, settings, device, workers)
    metrics, rows = evaluate(model, "val", device, cache)
    seconds = time.perf_counter() - started
    return model, metrics, rows, added, seconds


def environment(device: torch.device) -> dict:
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return {
        "device": device.type,
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_seed(folder: Path, model, metrics, rows, record) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    save_checkpoint(folder / "model.pt", record["model"], model)
    (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    write_predictions(folder / "predictions.csv", rows)
    # Written last: metrics.json marks the seed as done.
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def check_settings(run_dir: Path, settings: dict) -> None:
    """Keep one set of settings per run name so seeds are comparable."""
    path = run_dir / "settings.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != settings:
            raise SystemExit(
                f"{run_dir} was created with different settings; "
                "choose a new --name for a new experiment"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n")


def default_workers() -> int:
    return 4 if sys.platform == "linux" else 0


def number_or_text(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_tags(values: list[str]) -> dict:
    """KEY=VALUE pairs; numbers stay numbers so they can go on a plot axis."""
    tags = {}
    for value in values:
        key, separator, text = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--tag {value!r}: use KEY=VALUE, e.g. --tag strength=0.5")
        tags[key.strip()] = number_or_text(text.strip())
    return tags


def collect_seeds(run_dir: Path, filename: str) -> list[dict]:
    """The metrics of every finished seed in seed order, without the confusion."""
    files = sorted(
        run_dir.glob(f"seed-*/{filename}"),
        key=lambda file: int(file.parent.name.removeprefix("seed-")),
    )
    keys = ("seed", "egyptian_mau_recall", "macro_f1", "accuracy", "recall")
    seeds = [json.loads(file.read_text()) for file in files]
    return [{key: metrics[key] for key in keys} for metrics in seeds]


def write_results(run_dir: Path, name: str, settings: dict, tags: dict) -> Path:
    """One JSON file per run with every finished seed and the student's tags."""
    path = run_dir / "results.json"
    saved = json.loads(path.read_text()).get("tags", {}) if path.exists() else {}
    results = {
        "name": name,
        **settings,
        "split": "val",
        "tags": {**saved, **tags},
        "seeds": collect_seeds(run_dir, "metrics.json"),
    }
    path.write_text(json.dumps(results, indent=2) + "\n")
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained CNN and evaluate it on the validation "
        "photos. Results go to runs/<name>/."
    )
    parser.add_argument("--name", required=True, help="run name, e.g. m1-oversample")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--pool", help="generated pool (name in pools/ or a path); --method synthetic"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--model", choices=MODELS, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET,
        help=f"added Egyptian Mau examples (default {BUDGET}); keep it fixed",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="describe the run in results.json, e.g. --tag mode=img2img "
        "--tag strength=0.5; repeat to add more, rerun to add them later",
    )
    args = parser.parse_args(argv)
    if args.method == "synthetic" and not args.pool:
        parser.error("--method synthetic needs --pool")
    if args.method != "synthetic" and args.pool:
        parser.error("--pool is used only with --method synthetic")
    try:
        args.tags = parse_tags(args.tag)
    except ValueError as error:
        parser.error(str(error))
    return args


def settings_from(args, pool_dir) -> dict:
    return {
        "method": args.method,
        "pool": pool_dir.name if pool_dir else None,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "budget": args.budget,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    pool_dir = pools.resolve(args.pool) if args.pool else None
    settings = settings_from(args, pool_dir)
    run_dir = RUNS_DIR / args.name
    check_settings(run_dir, settings)
    device = pick_device(args.device)
    cache = photo_cache(("train", "val"))
    print(
        f"{args.name}: {args.method}, {args.model}, {len(args.seeds)} seeds, {device}"
    )

    for seed in args.seeds:
        folder = run_dir / f"seed-{seed}"
        if (folder / "metrics.json").exists():
            print(f"  seed {seed}: already done")
        else:
            model, metrics, rows, added, seconds = run_seed(
                args.method,
                pool_dir,
                seed,
                settings,
                device,
                cache,
                args.workers,
            )
            metrics = {"name": args.name, "seed": seed, "split": "val", **metrics}
            record = {
                **settings,
                "name": args.name,
                "seed": seed,
                "added": added,
                "seconds": round(seconds, 1),
                "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                **environment(device),
            }
            write_seed(folder, model, metrics, rows, record)
            print(
                f"  seed {seed}: Egyptian Mau recall "
                f"{metrics['egyptian_mau_recall']:.3f}, macro-F1 "
                f"{metrics['macro_f1']:.3f} ({seconds:.0f} s)"
            )

    path = write_results(run_dir, args.name, settings, args.tags)
    print(f"Results of every seed are in {path}")


if __name__ == "__main__":
    main()
