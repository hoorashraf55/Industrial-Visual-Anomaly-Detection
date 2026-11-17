import json
from pathlib import Path

import torch

from src.datasets.dataloaders import get_dataloaders
from src.evaluate import compute_image_metrics, compute_pixel_metrics
from src.training.baseline_trainer import train_and_evaluate_baseline
from src.training.deep_learning_trainer import train_and_evaluate_deep_model
from src.training.utils import set_seed


def run_experiment(args):
    """Run one experiment and save its metrics as a JSON file."""

    set_seed(args.seed)
    augment = getattr(args, "augment", False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"\n--- Running [{args.model}] | "
        f"Category: [{args.category}] | "
        f"Seed: [{args.seed}] | Device: [{device}] ---"
    )

    # Load dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_root,
        category=args.category,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=0,
        augment=augment,
    )

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints") / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build checkpoint name
    checkpoint_tag = f"{args.category}_seed{args.seed}"

    if args.model == "autoencoder" and args.pretrained:
        checkpoint_tag += "_pretrained"

    if args.model == "efficientad" and args.no_ae:
        checkpoint_tag += "_noAE"

    if augment:
        checkpoint_tag += "_aug"

    # Keep the default 256 checkpoint name unchanged
    if args.image_size != 256:
        checkpoint_tag += f"_sz{args.image_size}"

    checkpoint_extension = (
        ".joblib"
        if args.model == "baseline"
        else ".pt"
    )

    checkpoint_path = checkpoint_dir / (
        checkpoint_tag + checkpoint_extension
    )

    # Train and evaluate model
    if args.model == "baseline":
        experiment = train_and_evaluate_baseline(
            train_loader,
            val_loader,
            test_loader,
            checkpoint_path,
        )
    else:
        experiment = train_and_evaluate_deep_model(
            args,
            train_loader,
            val_loader,
            test_loader,
            device,
            checkpoint_path,
        )

    # Calculate image-level metrics
    image_metrics = compute_image_metrics(
        experiment["test_labels"],
        experiment["test_scores"],
        experiment["threshold"],
    )

    # Calculate pixel-level metrics
    pixel_metrics = compute_pixel_metrics(
        experiment["gt_masks"],
        experiment["anomaly_maps"],
        experiment["pixel_threshold"],
    )

    # Save all experiment results
    results = {
        "model": args.model,
        "category": args.category,
        "seed": args.seed,
        "image_size": args.image_size,

        # Experiment configuration
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "pretrained": args.pretrained,
        "no_ae": args.no_ae,
        "augmentation": augment,

        **image_metrics,
        **pixel_metrics,
        **experiment["stats"],

        "train_time_sec": round(
            experiment["train_time"], 2
        ),

        "inference_time_ms": round(
            experiment["inference_time_ms"], 2
        ),

        "hardware": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    }

    # Create results directory
    results_dir = Path("experiments") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build results filename
    tag = (
        f"{args.model}_{args.category}_"
        f"s{args.seed}_sz{args.image_size}"
    )

    if args.pretrained:
        tag += "_pretrained"

    if args.no_ae:
        tag += "_noAE"

    if augment:
        tag += "_aug"

    output_file = results_dir / f"{tag}.json"

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved results to {output_file}")
    print(json.dumps(results, indent=2))

    return results

