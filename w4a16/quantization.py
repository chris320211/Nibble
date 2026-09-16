import torch


def pack_int4(quantized_weights: torch.Tensor) -> torch.Tensor:
    """ Used to pack int4 pair tensor into int8 tensor byte, two per byte """
    if quantized_weights.dtype != torch.int8:
        raise TypeError("quantized_weights int4 value must have int8 dtype")
    if quantized_weights.ndim != 2:
        raise ValueError("quantized_weights shape mismatch, expected: [N,K]")
    if quantized_weights.shape[1] % 2 != 0:
        raise ValueError("K num of weights per row, must be even so two weights = 1 byte")
    
    if torch.any((quantized_weights < -7) | (quantized_weights > 7 )):
        raise ValueError("quantized_weights val must be between [-7, 7]")
    
    # low byte side and high byte, rearrange matrix shape k/2
    low_weights = quantized_weights[:, 0::2]
    high_weights = quantized_weights[:, 1::2]
    
    low_four_bits = torch.bitwise_and(low_weights, 0x0F)
    high_four_bits = torch.bitwise_and(high_weights, 0x0F)
    low_four_bits = low_four_bits.to(torch.uint8)
    high_four_bits = high_four_bits.to(torch.uint8)
    
    shifted_high_bits = torch.bitwise_left_shift(high_four_bits, 4)
    packed_weights = torch.bitwise_or(low_four_bits, shifted_high_bits)
    
    return packed_weights
    

def unpack_int4(packed_weights: torch.Tensor) -> torch.Tensor:
    """ Used to unpack each int8 tensor byte into two signed int4 weights tensor. """
    if packed_weights.dtype != torch.uint8:
        raise TypeError("packed_weights must have uint8 dtype")
    if packed_weights.ndim != 2:
        raise ValueError("packed_weights shape mismatch, should be [N,K]")

    low_four_bits = torch.bitwise_and(packed_weights, 0x0F)
    high_four_bits = torch.bitwise_right_shift(packed_weights, 4)
    low_vals = low_four_bits.to(torch.int8)
    high_vals = high_four_bits.to(torch.int8)
    
    low_vals = torch.where(
        low_vals >= 8,
        low_vals - 16,
        low_vals,
    )
    high_vals = torch.where(
        high_vals >= 8,
        high_vals - 16,
        high_vals,
    )
    
    unpacked_weights = torch.empty(
        (packed_weights.shape[0], packed_weights.shape[1] * 2),
        dtype=torch.int8,
        device=packed_weights.device,
    )
    
    unpacked_weights[:, 0::2] = low_vals
    unpacked_weights[:, 1::2] = high_vals
    
    return unpacked_weights
    
def quantize_groupwise(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """ Quantize original FP16 weights into signed INT4 values and FP16 quantization scales. """
    if weights.dtype not in (torch.float16, torch.float32):
        raise TypeError("weights must have dtype torch.float16 or torch.float32")
    if weights.ndim != 2:
        raise ValueError("weights must have shape [N, K]")
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")
    if weights.shape[1] % group_size != 0:
        raise ValueError("K must be divisible by group_size")

    weights_float = weights.to(torch.float32)
    output_features, input_features = weights_float.shape
    grouped_weights = weights_float.reshape(
        output_features,
        input_features // group_size,
        group_size,
    )

    max_absolute_values = grouped_weights.abs().amax(dim=2)
    scales_float = max_absolute_values / 7.0
    scales_float = torch.where(
        max_absolute_values == 0,
        torch.ones_like(scales_float),
        scales_float,
    )

    quantized_groups = torch.round(
        grouped_weights / scales_float.unsqueeze(dim=2)
    ).clamp(-7, 7)

    quantized_weights = quantized_groups.to(torch.int8).reshape(
        output_features,
        input_features,
    )
    scales = scales_float.to(torch.float16)

    return quantized_weights, scales


def quantize_and_pack_weights(
    weights: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert floating-point weights into packed INT4 weights via pack_int4 and FP16 scales via quantize_groupwise."""
    quantized_weights, scales = quantize_groupwise(weights, group_size)
    packed_weights = pack_int4(quantized_weights)

    return packed_weights, scales
