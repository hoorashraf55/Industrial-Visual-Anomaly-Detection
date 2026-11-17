
from pathlib import Path


# Project directories

FIG_DIR = Path("experiments/figures")
RES_DIR = Path("experiments/results")

FIG_DIR.mkdir(parents=True, exist_ok=True)


# MVTec AD categories

ALL_15_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]


# Models order

MODELS_ORDER = [
    "baseline",
    "autoencoder",
    "patchcore",
    "efficientad"
]


# Canonical experiment settings

DEFAULT_SEED = 42
MULTI_SEEDS = [42, 123, 2026]
DEFAULT_IMAGE_SIZE = 256


# Error analysis settings

ERROR_ANALYSIS_CATEGORY = "cable"
ERROR_ANALYSIS_SEED = 42
ERROR_ANALYSIS_BATCH_SIZE = 1
