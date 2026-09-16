from pathlib import Path

import torch


FORMAT_NAME = "w4a16-linear"
FORMAT_VERSION = 1


def _validate_quantized_tensors(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[int, int]:
    if packed_weights.dtype != torch.uint8 or packed_weights.ndim != 2:
        raise ValueError("packed_weights must be a uint8 tensor shaped [N, K / 2]")
    if scales.dtype != torch.float16 or scales.ndim != 2:
        raise ValueError("scales must be a float16 tensor shaped [N, K / group_size]")
    if packed_weights.device != scales.device:
        raise ValueError("packed_weights and scales must share a device")
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    output_features = packed_weights.shape[0]
    input_features = packed_weights.shape[1] * 2
    if input_features % group_size != 0:
        raise ValueError("K must be divisible by group_size")

    expected_scale_shape = (output_features, input_features // group_size)
    if scales.shape != expected_scale_shape:
        raise ValueError(f"scales must have shape {expected_scale_shape}")

    return output_features, input_features


def save_quantized_weight_checkpoint(
    path: str | Path,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> None:
    """Save one packed quantized weight matrix and its format metadata."""
    original_shape = _validate_quantized_tensors(
        packed_weights,
        scales,
        group_size,
    )
    checkpoint = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "packed_weights": packed_weights.detach().cpu(),
        "scales": scales.detach().cpu(),
        "group_size": group_size,
        "original_shape": original_shape,
    }
    torch.save(checkpoint, Path(path))


def load_quantized_weight_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load and validate one packed quantized weight matrix."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if checkpoint.get("format") != FORMAT_NAME:
        raise ValueError("checkpoint has an unsupported format")
    if checkpoint.get("version") != FORMAT_VERSION:
        raise ValueError("checkpoint has an unsupported version")

    packed_weights = checkpoint.get("packed_weights")
    scales = checkpoint.get("scales")
    group_size = checkpoint.get("group_size")
    if not isinstance(packed_weights, torch.Tensor):
        raise ValueError("checkpoint is missing packed_weights")
    if not isinstance(scales, torch.Tensor):
        raise ValueError("checkpoint is missing scales")
    if not isinstance(group_size, int):
        raise ValueError("checkpoint is missing group_size")

    original_shape = _validate_quantized_tensors(
        packed_weights,
        scales,
        group_size,
    )
    if tuple(checkpoint.get("original_shape", ())) != original_shape:
        raise ValueError("checkpoint original_shape does not match its tensors")

    return packed_weights, scales, group_size
