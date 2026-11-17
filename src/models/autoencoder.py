import torch
import torch.nn as nn
import torchvision.models as models
from src.components.anomaly_scoring import compute_spatial_anomaly_map

class CNNAutoencoder(nn.Module):
    """
    Classic DL Approach: Convolutional Autoencoder.
    Supports 'scratch' encoder or 'pretrained' ResNet18 encoder for the Pretrained vs Scratch study.
    """
    def __init__(self, latent_dim: int = 128, pretrained_encoder: bool = False):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder

        if pretrained_encoder:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 512, H/32, W/32)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512, latent_dim, kernel_size=1),
                nn.ReLU(True),
                nn.Conv2d(latent_dim, 512, kernel_size=1),
                nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, latent_dim, 4, stride=2, padding=1), nn.ReLU(True),
            )
            self.bottleneck = nn.Identity()
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(self.encoder(x))
        return self.decoder(z)

    @torch.no_grad()
    def predict_anomaly(self, x: torch.Tensor) -> tuple:
        recon = self.forward(x)
        pixel_err = ((x - recon) ** 2).mean(dim=1)  # (B, H, W)
        return compute_spatial_anomaly_map(pixel_err, target_size=(x.shape[2], x.shape[3]))