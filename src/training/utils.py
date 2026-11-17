
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def get_model_stats(model, ckpt_path: Path) -> dict:
    parameters = 0
    trainable_parameters = 0
    model_tensor_bytes = 0

    # Count parameters and their trainable subset.
    if callable(getattr(model, "parameters", None)):
        all_parameters = list(model.parameters())

        parameters = sum(
            p.numel()
            for p in all_parameters
        )

        trainable_parameters = sum(
            p.numel()
            for p in all_parameters
            if p.requires_grad
        )

    # Estimate the in-memory size of model parameters and buffers.
    if callable(getattr(model, "state_dict", None)):
        state = model.state_dict()

        model_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in state.values()
            if isinstance(tensor, torch.Tensor)
        )

    # PatchCore stores its memory bank as a regular tensor,
    # not as a registered parameter or buffer.
    memory_bank_size_mb = 0.0
    memory_bank = getattr(model, "memory_bank", None)

    if memory_bank is not None:
        if isinstance(memory_bank, torch.Tensor):
            memory_bank_bytes = (
                memory_bank.numel()
                * memory_bank.element_size()
            )
        else:
            memory_bank_bytes = np.asarray(memory_bank).nbytes

        memory_bank_size_mb = (
            memory_bank_bytes / (1024 * 1024)
        )

        model_tensor_bytes += memory_bank_bytes

    checkpoint_size_mb = (
        ckpt_path.stat().st_size / (1024 * 1024)
        if ckpt_path.exists()
        else 0.0
    )

    # For models without a PyTorch state_dict (e.g. the
    # scikit-learn baseline), use the serialized file size.
    if model_tensor_bytes > 0:
        estimated_model_size_mb = (
            model_tensor_bytes / (1024 * 1024)
        )
    else:
        estimated_model_size_mb = checkpoint_size_mb

    return {
        "parameters": int(parameters),
        "trainable_parameters": int(trainable_parameters),
        "checkpoint_size_mb": round(checkpoint_size_mb, 3),
        "model_size_mb": round(checkpoint_size_mb, 3),
        "estimated_model_size_mb": round(
            estimated_model_size_mb, 3
        ),
        "memory_bank_size_mb": round(
            memory_bank_size_mb, 3
        ),
    }

