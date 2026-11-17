import json
from pathlib import Path
from torch.utils.data import DataLoader
from src.datasets.mvtec_dataset import MVTecDataset


def get_dataloaders(
    root_dir: str,
    category: str,
    batch_size: int = 4,
    image_size: int = 256,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    augment: bool = False
):
    train_ds = MVTecDataset(
        root_dir,
        category,
        split="train",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        augment=augment
    )
    val_ds = MVTecDataset(
        root_dir,
        category,
        split="val",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        augment=False
    )
    test_ds = MVTecDataset(
        root_dir,
        category,
        split="test",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def save_split_manifest(root_dir: str, categories: list, output_path: str = "data/split_manifest.json", seed: int = 42):
    manifest = {}
    for cat in categories:
        manifest[cat] = {}
        for split in ["train", "val", "test"]:
            ds = MVTecDataset(root_dir, cat, split=split, seed=seed)
            manifest[cat][split] = [str(s["image_path"].relative_to(root_dir)) for s in ds.samples]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Split manifest saved successfully to {output_path}")