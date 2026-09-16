import torch

from w4a16.quantization import unpack_int4


def dequantize_weights(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct floating-point weights for correctness testing."""
    if packed_weights.dtype != torch.uint8:
        raise TypeError("packed_weights must have dtype torch.uint8")
    if scales.dtype != torch.float16:
        raise TypeError("scales must have dtype torch.float16")
    if packed_weights.ndim != 2:
        raise ValueError("packed_weights must have shape [N, K / 2]")
    if scales.ndim != 2:
        raise ValueError("scales must have shape [N, K / group_size]")
    if packed_weights.device != scales.device:
        raise ValueError("packed_weights and scales must be on the same device")
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    quantized_weights = unpack_int4(packed_weights)
    output_features, input_features = quantized_weights.shape

    if input_features % group_size != 0:
        raise ValueError("K must be divisible by group_size")

    number_of_groups = input_features // group_size
    expected_scale_shape = (output_features, number_of_groups)
    if scales.shape != expected_scale_shape:
        raise ValueError(
            f"scales must have shape {expected_scale_shape}, got {tuple(scales.shape)}"
        )

    grouped_weights = quantized_weights.to(torch.float32).reshape(
        output_features,
        number_of_groups,
        group_size,
    )
    grouped_scales = scales.to(torch.float32).unsqueeze(dim=2)
    dequantized_groups = grouped_weights * grouped_scales

    return dequantized_groups.reshape(output_features, input_features)


def linear_reference(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Compute the quantized linear layer using ordinary PyTorch operations."""
    if activations.dtype != torch.float16:
        raise TypeError("activations must have dtype torch.float16")
    if activations.ndim != 2:
        raise ValueError("activations must have shape [M, K]")
    if (
        activations.device != packed_weights.device
        or activations.device != scales.device
    ):
        raise ValueError("activations, packed_weights, and scales must share a device")

    dequantized_weights = dequantize_weights(
        packed_weights,
        scales,
        group_size,
    )

    if activations.shape[1] != dequantized_weights.shape[1]:
        raise ValueError("activations and weights must have the same K dimension")

    output_float = torch.matmul(
        activations.to(torch.float32),
        dequantized_weights.transpose(0, 1),
    )

    return output_float.to(torch.float16)
