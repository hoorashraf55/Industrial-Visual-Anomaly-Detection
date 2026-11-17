import time

import joblib
import numpy as np

from src.models.baseline_ocsvm import BaselineOCSVM
from src.evaluate import find_best_threshold, find_pixel_threshold
from src.training.utils import get_model_stats, to_numpy


def train_and_evaluate_baseline(
    train_loader,
    val_loader,
    test_loader,
    ckpt_path,
):
    """Fit the HOG + One-Class SVM baseline and evaluate it."""
    start_time = time.time()

    model = BaselineOCSVM(
        patch_size=32,
        nu=0.05,
    )
    model.fit(train_loader)

    train_time = time.time() - start_time

    joblib.dump(model, ckpt_path)
    stats = get_model_stats(model, ckpt_path)

    # Validation data is normal-only. Thresholds are determined on validation.
    _, val_scores, _, val_maps = model.predict(val_loader)
    val_scores = to_numpy(val_scores)
    val_maps = to_numpy(val_maps)

    val_labels = np.zeros(len(val_scores), dtype=int)
    threshold = find_best_threshold(val_labels, val_scores)
    pixel_threshold = find_pixel_threshold(val_maps)

    # Baseline inference uses the CPU-based scikit-learn pipeline.
    inference_start = time.time()
    test_labels, test_scores, gt_masks, anomaly_maps = model.predict(test_loader)

    inference_time_ms = (
        (time.time() - inference_start) / len(test_loader.dataset)
    ) * 1000.0

    return {
        "model": model,
        "train_time": train_time,
        "inference_time_ms": inference_time_ms,
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "test_labels": to_numpy(test_labels),
        "test_scores": to_numpy(test_scores),
        "gt_masks": to_numpy(gt_masks),
        "anomaly_maps": to_numpy(anomaly_maps),
        "stats": stats,
    }
