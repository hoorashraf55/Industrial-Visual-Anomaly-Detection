import torch
import torch.nn.functional as F


def compute_spatial_anomaly_map(
    pixel_errors: torch.Tensor,
    target_size: tuple = (256, 256),
    smooth_kernel: int = 5
) -> tuple:

    if pixel_errors.dim() == 3:
        pixel_errors = pixel_errors.unsqueeze(1)

    upsampled = F.interpolate(
        pixel_errors,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )

    if smooth_kernel > 1:
        pad = smooth_kernel // 2
        upsampled = F.avg_pool2d(
            upsampled,
            kernel_size=smooth_kernel,
            stride=1,
            padding=pad
        )

    anomaly_map = upsampled.squeeze(1)

    batch_size = anomaly_map.shape[0]
    flat_maps = anomaly_map.reshape(batch_size, -1)

    k = max(1, int(flat_maps.shape[1] * 0.01))
    top_k_values, _ = torch.topk(flat_maps, k, dim=1)

    image_scores = top_k_values.mean(dim=1)

    return image_scores, anomaly_map