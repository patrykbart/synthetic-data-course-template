"""Generated pools: reading manifests and drawing the images a run adds."""

import json
import random
from collections import defaultdict
from pathlib import Path

from synthdata.config import POOLS_DIR


def resolve(pool: str) -> Path:
    """Accept a pool name from pools/ or a path to a pool folder."""
    path = Path(pool)
    if not (path / "manifest.jsonl").exists():
        path = POOLS_DIR / pool
    if not (path / "manifest.jsonl").exists():
        raise SystemExit(f"no pool {pool!r}: expected {path / 'manifest.jsonl'}")
    return path


def read_manifest(pool_dir: Path) -> list[dict]:
    """One record per sample in sample order; a later attempt replaces an earlier."""
    manifest = pool_dir / "manifest.jsonl"
    if not manifest.exists():
        return []
    records = {}
    for line in manifest.read_text().splitlines():
        if line:
            record = json.loads(line)
            records[record["sample_id"]] = record
    return [records[sample] for sample in sorted(records)]


def usable(pool_dir: Path) -> list[dict]:
    return [
        record
        for record in read_manifest(pool_dir)
        if record["status"] == "ok" and (pool_dir / record["file"]).exists()
    ]


def draw(records: list[dict], budget: int, rng: random.Random) -> list[dict]:
    """Pick `budget` images, spreading the choice evenly over source photos."""
    if len(records) < budget:
        raise SystemExit(
            f"the pool has {len(records)} usable images but a run adds {budget}: "
            "generate more images first"
        )
    groups = defaultdict(list)
    for record in records:
        groups[record.get("source") or ""].append(record)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups)
    rng.shuffle(order)
    chosen = []
    while len(chosen) < budget:
        for key in order:
            if groups[key] and len(chosen) < budget:
                chosen.append(groups[key].pop())
    return chosen
