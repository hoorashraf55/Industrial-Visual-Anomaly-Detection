
import numpy as np
import matplotlib.pyplot as plt

from src.reporting.config import (
    FIG_DIR,
    ALL_15_CATEGORIES,
    MODELS_ORDER
)


def generate_charts(all_15_s42):

    if all_15_s42 is None or all_15_s42.empty:
        print("No benchmark results available for charts.")
        return

    # --------------------------------------------------
    # Chart 1: Overall Model Comparison
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    means_image = []
    means_pixel = []
    means_iou = []

    for model_name in MODELS_ORDER:

        sub = all_15_s42[
            all_15_s42["model"] == model_name
        ]

        means_image.append(
            sub["image_auroc"].mean()
        )

        means_pixel.append(
            sub["pixel_auroc"].mean()
        )

        means_iou.append(
            sub["pixel_iou"].mean()
        )

    x = np.arange(len(MODELS_ORDER))
    width = 0.25

    ax.bar(
        x - width,
        means_image,
        width=width,
        label="Image AUROC"
    )

    ax.bar(
        x,
        means_pixel,
        width=width,
        label="Pixel AUROC"
    )

    ax.bar(
        x + width,
        means_iou,
        width=width,
        label="Pixel IoU"
    )

    ax.set_xticks(x)

    ax.set_xticklabels([
        "Baseline\n(OC-SVM)",
        "Classic DL\n(Autoencoder)",
        "Modern DL\n(PatchCore)",
        "Advanced DL\n(EfficientAD)"
    ])

    ax.set_ylim(0, 1.1)

    ax.set_ylabel("Mean Score (15 Categories)")

    ax.set_title(
        "Overall Model Performance Across "
        "All 15 MVTec AD Categories"
    )

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "overall_15_categories_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Chart 2: Per-Class Image AUROC
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 5.5))

    categories = [
        category
        for category in ALL_15_CATEGORIES
        if category in all_15_s42["category"].unique()
    ]

    x = np.arange(len(categories))
    width = 0.2

    for idx, model_name in enumerate(MODELS_ORDER):

        values = []

        for category in categories:

            result = all_15_s42[
                (all_15_s42["model"] == model_name) &
                (all_15_s42["category"] == category)
            ]["image_auroc"]

            values.append(
                result.iloc[0] if not result.empty else np.nan
            )

        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            label=model_name.upper()
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories,
        rotation=35,
        ha="right"
    )

    ax.set_ylim(0, 1.1)

    ax.set_ylabel("Image AUROC")

    ax.set_title(
        "Per-Class Image AUROC Across "
        "All 15 MVTec AD Categories"
    )

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "per_class_15_categories.png",
        dpi=300
    )

    plt.close()

    print(f"Updated charts saved in {FIG_DIR}")
