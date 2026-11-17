import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.components.anomaly_scoring import compute_spatial_anomaly_map


class PDNNetwork(nn.Module):

    def __init__(self, out_channels: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientADModel(nn.Module):

    def __init__(
        self,
        out_channels: int = 128,
        use_ae_branch: bool = True
    ):
        super().__init__()

        self.use_ae_branch = use_ae_branch

        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        # Frozen pretrained Teacher: ResNet18 up to layer2
        self.teacher = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.teacher.eval()

        # Trainable Student
        self.student = PDNNetwork(
            out_channels=out_channels
        )

        # Optional Autoencoder Branch
        if self.use_ae_branch:

            self.ae_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )

            self.ae_decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    64, 128, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(
                    128, 256, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    256,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep the frozen Teacher in evaluation mode
        self.teacher.eval()

        return self

    def _get_teacher_features(self, x: torch.Tensor):

        self.teacher.eval()

        with torch.no_grad():
            teacher_features = self.teacher(x)

            teacher_features = (
                teacher_features
                - teacher_features.mean(
                    dim=(2, 3),
                    keepdim=True
                )
            ) / (
                teacher_features.std(
                    dim=(2, 3),
                    keepdim=True
                ) + 1e-6
            )

        return teacher_features

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:

        teacher_features = self._get_teacher_features(x)

        student_features = self.student(x)

        student_features = F.interpolate(
            student_features,
            size=teacher_features.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss_student_teacher = F.mse_loss(
            student_features,
            teacher_features
        )

        if not self.use_ae_branch:
            return loss_student_teacher

        ae_features = self.ae_decoder(
            self.ae_encoder(x)
        )

        ae_features = F.interpolate(
            ae_features,
            size=teacher_features.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss_ae_teacher = F.mse_loss(
            ae_features,
            teacher_features
        )

        loss_student_ae = F.mse_loss(
            student_features,
            ae_features
        )

        total_loss = (
            loss_student_teacher
            + loss_ae_teacher
            + 0.5 * loss_student_ae
        )

        return total_loss

    @torch.no_grad()
    def predict_anomaly(self, x: torch.Tensor) -> tuple:

        self.eval()

        teacher_features = self._get_teacher_features(x)

        student_features = self.student(x)

        student_features = F.interpolate(
            student_features,
            size=teacher_features.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        student_map = (
            (teacher_features - student_features) ** 2
        ).mean(dim=1)

        if self.use_ae_branch:

            ae_features = self.ae_decoder(
                self.ae_encoder(x)
            )

            ae_features = F.interpolate(
                ae_features,
                size=teacher_features.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            ae_map = (
                (student_features - ae_features) ** 2
            ).mean(dim=1)

            combined_map = 0.5 * (
                student_map + ae_map
            )

        else:
            combined_map = student_map

        return compute_spatial_anomaly_map(
            combined_map,
            target_size=(x.shape[2], x.shape[3])
        )