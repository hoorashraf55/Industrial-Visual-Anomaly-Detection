import time

import numpy as np
import torch

from src.components.custom_losses import CombinedAnomalyLoss
from src.models.autoencoder import CNNAutoencoder
from src.models.patchcore import PatchCoreModel
from src.models.efficientad import EfficientADModel
from src.evaluate import find_best_threshold, find_pixel_threshold
from src.training.utils import get_model_stats, to_numpy


def _train_autoencoder(model, train_loader, device, args):
    criterion = CombinedAnomalyLoss(alpha=0.85)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, _, _, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            reconstruction = model(images)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {average_loss:.4f}"
        )


def _train_efficientad(model, train_loader, device, args):
    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, _, _, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {average_loss:.4f}"
        )


def _build_and_train_model(args, train_loader, device):
    if args.model == "autoencoder":
        model = CNNAutoencoder(
            latent_dim=128,
            pretrained_encoder=args.pretrained,
        ).to(device)
        _train_autoencoder(model, train_loader, device, args)

    elif args.model == "patchcore":
        model = PatchCoreModel(
            coreset_ratio=0.1,
            n_neighbors=3,
            seed=args.seed,
        )
        model.fit(train_loader, device)

    elif args.model == "efficientad":
        model = EfficientADModel(
            out_channels=128,
            use_ae_branch=not args.no_ae,
        ).to(device)
        _train_efficientad(model, train_loader, device, args)

    else:
        raise ValueError(f"Unsupported deep-learning model: {args.model}")

    return model


def _save_model(model, model_name, ckpt_path):
    if model_name == "patchcore":
        torch.save({"memory_bank": model.memory_bank}, ckpt_path)
    else:
        torch.save(model.state_dict(), ckpt_path)


def _evaluate_validation(model, val_loader, device):
    if hasattr(model, "eval"):
        model.eval()

    val_scores = []
    val_anomaly_maps = []

    with torch.no_grad():
        for images, _, _, _ in val_loader:
            images = images.to(device)
            scores, anomaly_maps_batch = model.predict_anomaly(images)

            val_scores.extend(to_numpy(scores).reshape(-1))
            val_anomaly_maps.append(to_numpy(anomaly_maps_batch))

    val_scores = np.asarray(val_scores)
    val_anomaly_maps = np.concatenate(val_anomaly_maps, axis=0)

    # Validation contains normal images only.
    val_labels = np.zeros(len(val_scores), dtype=int)
    threshold = find_best_threshold(val_labels, val_scores)
    pixel_threshold = find_pixel_threshold(val_anomaly_maps)

    return threshold, pixel_threshold


def _evaluate_test(model, test_loader, device):
    test_labels = []
    test_scores = []
    gt_masks = []
    anomaly_maps = []

    # Synchronize before timing so previously queued GPU work is excluded.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    inference_start = time.time()

    with torch.no_grad():
        for images, masks, labels, _ in test_loader:
            images = images.to(device)
            scores, anomaly_maps_batch = model.predict_anomaly(images)

            test_scores.extend(to_numpy(scores).reshape(-1))
            anomaly_maps.append(to_numpy(anomaly_maps_batch))
            test_labels.extend(to_numpy(labels).reshape(-1))
            gt_masks.append(to_numpy(masks.squeeze(1)))

    # CUDA operations are asynchronous; wait before stopping the timer.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    inference_time_ms = (
        (time.time() - inference_start) / len(test_loader.dataset)
    ) * 1000.0

    return {
        "inference_time_ms": inference_time_ms,
        "test_labels": np.asarray(test_labels),
        "test_scores": np.asarray(test_scores),
        "gt_masks": np.concatenate(gt_masks, axis=0),
        "anomaly_maps": np.concatenate(anomaly_maps, axis=0),
    }


def train_and_evaluate_deep_model(
    args,
    train_loader,
    val_loader,
    test_loader,
    device,
    ckpt_path,
):
    """Train, validate, and test one deep-learning anomaly detector."""
    training_start = time.time()

    model = _build_and_train_model(args, train_loader, device)
    train_time = time.time() - training_start

    _save_model(model, args.model, ckpt_path)
    stats = get_model_stats(model, ckpt_path)

    threshold, pixel_threshold = _evaluate_validation(
        model,
        val_loader,
        device,
    )

    test_result = _evaluate_test(model, test_loader, device)

    return {
        "model": model,
        "train_time": train_time,
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "stats": stats,
        **test_result,
    }