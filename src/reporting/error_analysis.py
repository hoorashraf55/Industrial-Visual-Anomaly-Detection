
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.datasets.dataloaders import get_dataloaders
from src.models.efficientad import EfficientADModel

from src.reporting.config import (
    FIG_DIR,
    ERROR_ANALYSIS_CATEGORY,
    ERROR_ANALYSIS_SEED,
    ERROR_ANALYSIS_BATCH_SIZE
)


def generate_error_analysis_examples(
    data_root=r"C:\Users\Dell\Downloads\mvtec_anomaly_detection"
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(
        "checkpoints/efficientad/cable_seed42.pt"
    )

    if not checkpoint_path.exists():

        print(
            "Error analysis skipped: "
            f"Checkpoint not found at {checkpoint_path}"
        )

        return

    model = EfficientADModel().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    _, val_loader, test_loader = get_dataloaders(
        data_root,
        ERROR_ANALYSIS_CATEGORY,
        batch_size=ERROR_ANALYSIS_BATCH_SIZE,
        image_size=256,
        seed=ERROR_ANALYSIS_SEED,
        num_workers=0
    )

    # Calculate threshold using validation data.

    val_scores = []

    with torch.no_grad():

        for imgs, _, _, _ in val_loader:

            scores, _ = model.predict_anomaly(
                imgs.to(device)
            )

            val_scores.extend(
                scores.cpu().numpy().reshape(-1)
            )

    if not val_scores:

        print(
            "Error analysis skipped: "
            "Validation scores are empty."
        )

        return

    threshold = float(
        np.percentile(val_scores, 95)
    )

    false_positive_saved = 0
    false_negative_saved = 0

    mean = np.array([
        0.485,
        0.456,
        0.406
    ])

    std = np.array([
        0.229,
        0.224,
        0.225
    ])

    # Find misclassified test examples.

    with torch.no_grad():

        for imgs, masks, labels, defect_types in test_loader:

            scores, anomaly_maps = model.predict_anomaly(
                imgs.to(device)
            )

            score = float(scores.item())

            label = int(labels.item())

            prediction = (
                1 if score >= threshold else 0
            )

            if prediction == label:
                continue

            if prediction == 1 and false_positive_saved < 2:

                error_type = "False_Positive"
                index = false_positive_saved

                false_positive_saved += 1

            elif prediction == 0 and false_negative_saved < 2:

                error_type = "False_Negative"
                index = false_negative_saved

                false_negative_saved += 1

            else:
                continue

            # Original image

            image_np = (
                imgs.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            image_np = np.clip(
                image_np * std + mean,
                0,
                1
            )

            # Ground truth mask

            mask_np = (
                masks.squeeze()
                .cpu()
                .numpy()
            )

            # Predicted anomaly map

            anomaly_map_np = (
                anomaly_maps.squeeze()
                .cpu()
                .numpy()
            )

            # Plot results

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(11, 3.8)
            )

            axes[0].imshow(image_np)

            axes[0].set_title(
                f"Original ({defect_types[0]})\n"
                f"True: {label} | Pred: {prediction}"
            )

            axes[0].axis("off")

            axes[1].imshow(
                mask_np,
                cmap="gray"
            )

            axes[1].set_title(
                "Ground Truth Mask"
            )

            axes[1].axis("off")

            axes[2].imshow(
                anomaly_map_np,
                cmap="jet"
            )

            axes[2].set_title(
                f"Predicted Map (Score: {score:.2f})"
            )

            axes[2].axis("off")

            plt.suptitle(
                f"Error Analysis Example: {error_type} "
                "(Cable - EfficientAD)"
            )

            plt.tight_layout()

            output_path = (
                FIG_DIR /
                f"error_{error_type}_{index + 1}.png"
            )

            plt.savefig(
                output_path,
                dpi=250
            )

            plt.close()

    print(
        f"Error analysis examples saved in {FIG_DIR}"
    )
