"""Fixed course settings shared by every command."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SYNTHDATA_DATA_DIR", REPO / "data"))
SPLIT_CSV = REPO / "data" / "split.csv"
POOLS_DIR = REPO / "pools"
RUNS_DIR = REPO / "runs"
FINAL_DIR = REPO / "final"
EVIDENCE_DIR = REPO / "evidence"

CLASSES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "Egyptian Mau",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
]
TARGET = "Egyptian Mau"
TARGET_INDEX = CLASSES.index(TARGET)

# Each majority breed has 70 training photos and Egyptian Mau has 15, so every
# method adds the same 55 Egyptian Mau training examples.
BUDGET = 55
SEEDS = (0, 1, 2, 3, 4)
