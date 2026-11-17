
import pandas as pd

from src.reporting.config import (
    ALL_15_CATEGORIES,
    MODELS_ORDER,
    MULTI_SEEDS,
    DEFAULT_SEED,
    DEFAULT_IMAGE_SIZE,
)


def generate_tables(df):

    if df.empty:
        print("No experiment results available.")
        return

    # --------------------------------------------------
    # 1. Multi-Seed Stability: Bottle
    # --------------------------------------------------

    bottle_pattern = (
        r"(baseline|autoencoder|patchcore|efficientad)"
        r"_bottle_s(42|123|2026)_sz256\.json"
    )

    bottle_seeds = df[
        df["filename"].str.fullmatch(
            bottle_pattern,
            na=False
        )
    ].copy()

    seed_rows = []

    for model_name in MODELS_ORDER:

        sub = bottle_seeds[
            bottle_seeds["model"] == model_name
        ]

        if sub.empty:
            continue

        actual_seeds = sorted(sub["seed"].dropna().unique())

        if len(actual_seeds) != 3:
            print(
                f"Warning: {model_name} has "
                f"{len(actual_seeds)} seeds instead of 3."
            )

        seed_rows.append({

            "Model": model_name.upper(),

            "Image AUROC (Mean ± Seed SD)": (
                f"{sub['image_auroc'].mean():.4f} ± "
                f"{sub['image_auroc'].std():.4f}"
            ),

            "F1-Score (Mean ± Seed SD)": (
                f"{sub['f1_score'].mean():.4f} ± "
                f"{sub['f1_score'].std():.4f}"
            ),

            "Pixel AUROC (Mean ± Seed SD)": (
                f"{sub['pixel_auroc'].mean():.4f} ± "
                f"{sub['pixel_auroc'].std():.4f}"
            ),

            "Pixel IoU (Mean ± Seed SD)": (
                f"{sub['pixel_iou'].mean():.4f} ± "
                f"{sub['pixel_iou'].std():.4f}"
            ),

            "Params": (
                "N/A"
                if int(sub["parameters"].iloc[0]) == 0
                else int(sub["parameters"].iloc[0])
            ),

            "Checkpoint Size (MB)": round(
                sub["model_size_mb"].iloc[0],
                2
            ),

            "Train Time (s)": round(
                sub["train_time_sec"].mean(),
                2
            ),

            "Inf Time (ms)": round(
                sub["inference_time_ms"].mean(),
                2
            ),
        })

    seed_df = pd.DataFrame(seed_rows)

    # --------------------------------------------------
    # 2. Overall Benchmark: All 15 Categories
    # --------------------------------------------------

    categories_pattern = (
        r"(baseline|autoencoder|patchcore|efficientad)_"
        r"(bottle|cable|capsule|carpet|grid|hazelnut|leather|"
        r"metal_nut|pill|screw|tile|toothbrush|transistor|wood|zipper)"
        r"_s42_sz256\.json"
    )

    all_15_s42 = df[
        df["filename"].str.fullmatch(
            categories_pattern,
            na=False
        )
    ].copy()

    overall_rows = []

    for model_name in MODELS_ORDER:

        sub = all_15_s42[
            all_15_s42["model"] == model_name
        ]

        if sub.empty:
            continue

        overall_rows.append({

            "Model": model_name.upper(),

            "Categories Evaluated": (
                sub["category"].nunique()
            ),

            "Image AUROC (Mean ± Category SD)": (
                f"{sub['image_auroc'].mean():.4f} ± "
                f"{sub['image_auroc'].std():.4f}"
            ),

            "Mean Precision": (
                f"{sub['precision'].mean():.4f}"
            ),

            "Mean Recall": (
                f"{sub['recall'].mean():.4f}"
            ),

            "F1-Score (Mean ± Category SD)": (
                f"{sub['f1_score'].mean():.4f} ± "
                f"{sub['f1_score'].std():.4f}"
            ),

            "Pixel AUROC (Mean ± Category SD)": (
                f"{sub['pixel_auroc'].mean():.4f} ± "
                f"{sub['pixel_auroc'].std():.4f}"
            ),

            "Pixel IoU (Mean ± Category SD)": (
                f"{sub['pixel_iou'].mean():.4f} ± "
                f"{sub['pixel_iou'].std():.4f}"
            ),
        })

    overall_15_df = pd.DataFrame(overall_rows)

    # --------------------------------------------------
    # 3. Per-Class Image AUROC
    # --------------------------------------------------

    pivot_img = all_15_s42.pivot(
        index="category",
        columns="model",
        values="image_auroc"
    )

    pivot_img = pivot_img.reindex(
        index=[
            category
            for category in ALL_15_CATEGORIES
            if category in pivot_img.index
        ],
        columns=MODELS_ORDER
    ).round(4)

    pivot_img.loc["MEAN (15 Categories)"] = (
        pivot_img.mean().round(4)
    )

    # --------------------------------------------------
    # 4. Special Studies
    # --------------------------------------------------

    special_columns = [
        "filename",
        "image_auroc",
        "f1_score",
        "pixel_auroc",
        "pixel_iou",
        "train_time_sec",
        "inference_time_ms"
    ]

    # Study 1: Hyperparameter and Resolution

    hyperparameter_files = [
        "autoencoder_bottle_s42_sz256_lr001.json",
        "autoencoder_bottle_s42_sz256_lr0005.json",
        "autoencoder_bottle_s42_sz128.json",
        "autoencoder_bottle_s42_sz256.json",
        "patchcore_bottle_s42_sz256.json",
        "patchcore_bottle_s42_sz320.json",
    ]

    hyperparameter_df = (
        df[df["filename"].isin(hyperparameter_files)]
        [special_columns]
        .round(4)
    )

    # Study 2: Augmentation

    augmentation_files = [
        "autoencoder_bottle_s42_sz256.json",
        "autoencoder_bottle_s42_sz256_aug.json",
    ]

    augmentation_df = (
        df[df["filename"].isin(augmentation_files)]
        [special_columns]
        .round(4)
    )

    # Study 3: Pretrained vs Scratch

    pretrained_files = [
        "autoencoder_bottle_s42_sz256.json",
        "autoencoder_bottle_s42_sz256_pretrained.json",
    ]

    pretrained_df = (
        df[df["filename"].isin(pretrained_files)]
        [special_columns]
        .round(4)
    )

    # Study 4: EfficientAD Ablation

    ablation_files = [
        "efficientad_bottle_s42_sz256.json",
        "efficientad_bottle_s42_sz256_noAE.json",
    ]

    ablation_df = (
        df[df["filename"].isin(ablation_files)]
        [special_columns]
        .round(4)
    )

    # --------------------------------------------------
    # 5. Save Report Tables
    # --------------------------------------------------

    from pathlib import Path

    report_path = Path("experiments/report_tables.md")

    with open(report_path, "w", encoding="utf-8") as file:

        file.write(
            "# Generated Tables for AI403 Technical Report "
            "(All 15 MVTec AD Categories)\n\n"
        )

        file.write(
            "## Table 4A & Table 7: Multi-Seed Stability "
            "(Mean ± Seed SD on Bottle, 3 Seeds) "
            "& Computational Cost\n\n"
        )

        file.write(seed_df.to_markdown(index=False))

        file.write(
            "\n\n## Table 4B: Overall Benchmark Performance "
            "Across All 15 MVTec AD Categories "
            "(Seed 42) (Mean ± Category SD)\n\n"
        )

        file.write(overall_15_df.to_markdown(index=False))

        file.write(
            "\n\n## Per-Class Image AUROC Across All "
            "15 Categories (Seed 42)\n\n"
        )

        file.write(pivot_img.to_markdown())

        file.write(
            "\n\n## Study 1: Hyperparameter Study "
            "(Learning Rate & Image Resolution)\n\n"
        )

        file.write(hyperparameter_df.to_markdown(index=False))

        file.write(
            "\n\n## Study 2: Data Strategy Study "
            "(Without Augmentation vs With Augmentation)\n\n"
        )

        file.write(augmentation_df.to_markdown(index=False))

        file.write(
            "\n\n## Study 3: Pretrained vs Scratch Encoder "
            "(Autoencoder)\n\n"
        )

        file.write(pretrained_df.to_markdown(index=False))

        file.write(
            "\n\n## Study 4 / Table 5: Ablation Study "
            "(EfficientAD With AE Branch vs Without AE Branch)\n\n"
        )

        file.write(ablation_df.to_markdown(index=False))

        file.write("\n\n## All Raw Experiment Runs\n\n")

        raw_columns = [
            "filename",
            "model",
            "category",
            "seed",
            "image_size",
            "pretrained",
            "no_ae",
            "augmentation",
            "image_auroc",
            "f1_score",
            "pixel_auroc",
            "pixel_iou"
        ]

        available_columns = [
            column
            for column in raw_columns
            if column in df.columns
        ]

        file.write(
            df[available_columns].to_markdown(index=False)
        )

    print(f"Updated report tables saved to {report_path}")

    return all_15_s42
