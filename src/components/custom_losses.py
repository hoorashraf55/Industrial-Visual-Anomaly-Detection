import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedAnomalyLoss(nn.Module):

    def __init__(self, alpha: float = 0.85):
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        self.alpha = alpha
        self.mse = nn.MSELoss()

    def gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)

        return loss_x + loss_y

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        loss_mse = self.mse(pred, target)
        loss_gradient = self.gradient_loss(pred, target)

        return (
            self.alpha * loss_mse
            + (1.0 - self.alpha) * loss_gradient
        )