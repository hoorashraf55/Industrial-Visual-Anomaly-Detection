import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset Loader.
    - 'train' & 'val' splits are derived strictly from 'train/good' (normal images only) to prevent data leakage.
    - 'test' split uses the official 'test' directory (normal + anomalous) with pixel-level ground-truth masks.
    """
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        image_size: int = 256,
        val_ratio: float = 0.2,
        seed: int = 42,
        augment: bool = False,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        assert category in MVTEC_CATEGORIES, f"Invalid category: {category}. Must be one of {MVTEC_CATEGORIES}"
        assert split in ["train", "val", "test"], f"Invalid split: {split}. Must be 'train', 'val', or 'test'"

        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment

        # Apply data augmentation strictly on training split when enabled
        if transform is not None:
            self.transform = transform
        elif self.split == "train" and self.augment:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.mask_transform = mask_transform or T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.samples: List[Dict] = []
        self._load_samples()

    def _load_samples(self):
        cat_dir = self.root_dir / self.category

        if self.split in ["train", "val"]:
            good_dir = cat_dir / "train" / "good"
            all_images = sorted([p for p in good_dir.glob("*.png")])

            # Deterministic shuffle for reproducible 80/20 split across all models
            rng = random.Random(self.seed)
            indices = list(range(len(all_images)))
            rng.shuffle(indices)

            val_count = int(len(all_images) * self.val_ratio)
            if self.split == "val":
                selected_indices = sorted(indices[:val_count])
            else:
                selected_indices = sorted(indices[val_count:])

            for idx in selected_indices:
                self.samples.append({
                    "image_path": all_images[idx],
                    "mask_path": None,
                    "label": 0,  # 0 = Normal
                    "defect_type": "good"
                })

        elif self.split == "test":
            test_dir = cat_dir / "test"
            gt_dir = cat_dir / "ground_truth"

            for defect_folder in sorted(test_dir.iterdir()):
                if not defect_folder.is_dir():
                    continue
                defect_type = defect_folder.name
                is_normal = (defect_type == "good")

                for img_path in sorted(defect_folder.glob("*.png")):
                    mask_path = None
                    if not is_normal:
                        # MVTec naming convention: 000.png -> 000_mask.png
                        mask_name = f"{img_path.stem}_mask.png"
                        candidate_mask = gt_dir / defect_type / mask_name
                        if not candidate_mask.exists():
                            raise FileNotFoundError(f"Missing ground truth mask: {candidate_mask}")
                        mask_path = candidate_mask

                    self.samples.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "label": 0 if is_normal else 1,  # 0 = Normal, 1 = Anomalous
                        "defect_type": defect_type
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        sample = self.samples[idx]

        # Load RGB image
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transform(image)

        # Load or create binary mask (1, H, W)
        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        return image_tensor, mask_tensor, sample["label"], sample["defect_type"]