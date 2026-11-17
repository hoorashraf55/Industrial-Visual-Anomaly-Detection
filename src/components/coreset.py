import torch


def greedy_coreset_selection(
    features: torch.Tensor,
    sampling_ratio: float = 0.1,
    seed: int = 42,
    max_candidates: int = 10000
) -> torch.Tensor:

    if features.ndim != 2:
        raise ValueError("features must have shape (N, D).")

    if not 0 < sampling_ratio <= 1:
        raise ValueError("sampling_ratio must be between 0 and 1.")

    if features.shape[0] == 0:
        raise ValueError("Cannot select a coreset from empty features.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    n_samples = features.shape[0]
    target_size = max(1, int(n_samples * sampling_ratio))

    if n_samples > max_candidates:
        permutation = torch.randperm(
            n_samples,
            generator=generator
        )[:max_candidates]

        features = features[permutation]
        n_samples = features.shape[0]
        target_size = max(
            1,
            int(n_samples * sampling_ratio)
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    features_device = features.to(device)

    first_idx = torch.randint(
        0,
        n_samples,
        (1,),
        generator=generator
    ).item()

    selected_indices = [first_idx]

    min_distances = torch.cdist(
        features_device,
        features_device[first_idx:first_idx + 1]
    ).squeeze(1)

    for _ in range(1, target_size):
        next_idx = torch.argmax(min_distances).item()
        selected_indices.append(next_idx)

        new_distances = torch.cdist(
            features_device,
            features_device[next_idx:next_idx + 1]
        ).squeeze(1)

        min_distances = torch.minimum(
            min_distances,
            new_distances
        )

    return features[selected_indices].cpu()