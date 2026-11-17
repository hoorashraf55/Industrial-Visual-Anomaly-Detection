
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)


def find_best_threshold(
    val_labels: np.ndarray,
    val_scores: np.ndarray
) -> float:
    """
    Finds the image-level threshold using validation data only.
    If validation contains only normal images, use the 95th percentile.
    """
    if len(val_scores) == 0:
        raise ValueError("Validation scores are empty.")

    if len(np.unique(val_labels)) < 2:
        return float(np.percentile(val_scores, 95))

    precisions, recalls, thresholds = precision_recall_curve(
        val_labels, val_scores
    )

    if len(thresholds) == 0:
        return float(np.percentile(val_scores, 95))

    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-8)
    )

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def find_pixel_threshold(val_anomaly_maps: np.ndarray) -> float:
    """
    Calculates the pixel-level threshold using normal validation maps only.
    """
    if val_anomaly_maps.size == 0:
        raise ValueError("Validation anomaly maps are empty.")

    return float(np.percentile(val_anomaly_maps, 95))


def compute_image_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> dict:
    """
    Computes Image-level AUROC, Precision, Recall, and F1-score.
    """
    preds = (scores >= threshold).astype(int)

    auroc = (
        roc_auc_score(labels, scores)
        if len(np.unique(labels)) > 1
        else 0.0
    )

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        "image_auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold": float(threshold)
    }


def compute_pixel_metrics(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    pixel_threshold: float
) -> dict:
    """
    Computes Pixel-level AUROC and IoU.

    gt_masks: Binary array of shape (N, H, W).
    anomaly_maps: Continuous array of shape (N, H, W).
    pixel_threshold: Must be calculated from validation anomaly maps.
    """
    if gt_masks.shape != anomaly_maps.shape:
        raise ValueError(
            f"Shape mismatch: gt_masks {gt_masks.shape}, "
            f"anomaly_maps {anomaly_maps.shape}"
        )

    gt_flat = (gt_masks.flatten() > 0.5).astype(int)
    maps_flat = anomaly_maps.flatten()

    pixel_auroc = (
        roc_auc_score(gt_flat, maps_flat)
        if len(np.unique(gt_flat)) > 1
        else 0.0
    )

    pred_flat = (maps_flat >= pixel_threshold).astype(int)

    intersection = np.logical_and(
        gt_flat == 1,
        pred_flat == 1
    ).sum()

    union = np.logical_or(
        gt_flat == 1,
        pred_flat == 1
    ).sum()

    iou = float(intersection / (union + 1e-8))

    return {
        "pixel_auroc": float(pixel_auroc),
        "pixel_iou": iou,
        "pixel_threshold": float(pixel_threshold)
    }
