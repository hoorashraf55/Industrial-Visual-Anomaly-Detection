
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.components.coreset import greedy_coreset_selection
from src.components.anomaly_scoring import compute_spatial_anomaly_map


class PatchCoreModel(nn.Module):
    """
    Modern DL Approach: PatchCore using a pretrained WideResNet50_2
    backbone, custom Greedy Coreset Memory Bank, and k-Nearest
    Neighbor Distance Scoring.
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        n_neighbors: int = 3,
        seed: int = 42,
    ):
        super().__init__()

        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.seed = seed

        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.DEFAULT
        )

        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )

        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.memory_bank = None

        # PatchCore uses a frozen pretrained backbone.
        # Its weights are not updated during training.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def _extract_patch_embeddings(self, x: torch.Tensor) -> tuple:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        # Align f3 spatial size to f2.
        f3_up = F.interpolate(
            f3,
            size=f2.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        combined = torch.cat(
            [f2, f3_up],
            dim=1,
        )

        # Local neighborhood smoothing.
        combined = F.avg_pool2d(
            combined,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        B, C, H_p, W_p = combined.shape

        embeddings = (
            combined.permute(0, 2, 3, 1)
            .reshape(-1, C)
        )

        return embeddings, (B, H_p, W_p)

    @torch.no_grad()
    def fit(self, train_loader, device: torch.device):
        self.eval().to(device)

        all_embeddings = []

        for images, _, _, _ in train_loader:
            images = images.to(device)

            emb, _ = self._extract_patch_embeddings(images)

            all_embeddings.append(emb.cpu())

        full_bank = torch.cat(
            all_embeddings,
            dim=0,
        )

        # Compress memory bank using the custom coreset algorithm.
        self.memory_bank = greedy_coreset_selection(
            full_bank,
            self.coreset_ratio,
            self.seed,
        ).to(device)

    @torch.no_grad()
    def predict_anomaly(self, x: torch.Tensor) -> tuple:
        self.eval()

        emb, (B, H_p, W_p) = self._extract_patch_embeddings(x)

        # Compute Euclidean distances to the memory bank.
        dists = torch.cdist(
            emb,
            self.memory_bank,
        )

        knn_dists, _ = torch.topk(
            dists,
            k=self.n_neighbors,
            largest=False,
            dim=1,
        )

        patch_scores = (
            knn_dists.mean(dim=1)
            .view(B, H_p, W_p)
        )

        return compute_spatial_anomaly_map(
            patch_scores,
            target_size=(x.shape[2], x.shape[3]),
        )
