import argparse
from pathlib import Path

import yaml

from src.training.experiment_runner import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Industrial visual anomaly detection experiments."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=r"C:\Users\Dell\Downloads\mvtec_anomaly_detection",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "baseline",
            "autoencoder",
            "patchcore",
            "efficientad",
        ],
    )

    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained ResNet18 encoder in Autoencoder.",
    )

    parser.add_argument(
        "--no_ae",
        action="store_true",
        help="Disable EfficientAD autoencoder branch.",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation during training.",
    )

    args = parser.parse_args()

    config_path = (
        Path(args.config)
        if args.config
        else Path("configs") / f"{args.model}.yaml"
    )

    config = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        print(f"Loaded config: {config_path}")
    else:
        print(
            f"Config not found: {config_path}. "
            "Using default values."
        )

    # Command-line values take priority over YAML values.
    args.epochs = (
        args.epochs if args.epochs is not None else config.get("epochs", 15)
    )
    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 4)
    )
    args.image_size = (
        args.image_size
        if args.image_size is not None
        else config.get("image_size", 256)
    )
    args.lr = (
        args.lr
        if args.lr is not None
        else config.get("learning_rate", 1e-3)
    )
    args.pretrained = (
        args.pretrained or config.get("pretrained_encoder", False)
    )
    args.augment = (
        args.augment or config.get("augment", False)
    )

    print("\nExperiment Configuration")
    print("------------------------")
    print(f"Model: {args.model}")
    print(f"Category: {args.category}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Pretrained: {args.pretrained}")
    print(f"No AE: {args.no_ae}")
    print(f"Augmentation: {args.augment}")

    return args


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()