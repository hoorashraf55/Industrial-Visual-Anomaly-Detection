import subprocess
import sys
from pathlib import Path

MODELS = ["baseline", "autoencoder", "patchcore", "efficientad"]
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]
SEEDS = [42, 123, 2026]
RES_DIR = Path("experiments/results")

total = len(MODELS) * len(CATEGORIES) * len(SEEDS)
done = 0

for model in MODELS:
    for cat in CATEGORIES:
        for seed in SEEDS:
            out_file = RES_DIR / f"{model}_{cat}_s{seed}_sz256.json"
            if out_file.exists():
                done += 1
                print(f"[{done}/{total}] SKIP (already exists): {out_file.name}")
                continue

            done += 1
            print(f"\n[{done}/{total}] RUNNING: model={model}, category={cat}, seed={seed}")
            cmd = [
                sys.executable, "-m", "src.train",
                "--model", model,
                "--category", cat,
                "--seed", str(seed)
            ]
            subprocess.run(cmd, check=True)

print("\nAll 180 experiments completed successfully!")