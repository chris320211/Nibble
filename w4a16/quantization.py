import torch

def pack_int4(quantized_weights: torch.Tensor) -> torch.Tensor:
    """ Used to pack int4 pair into byte, two per byte """
    if quantized_weights.dtype != torch.int8:
        raise TypeError("quantized_weights int4 value must have int8 dtype")
    if quantized_weights.ndim != 2:
        raise ValueError("quantized_weights shape mismatch, expected: [N,K]")
    if quantized_weights.shape[1] % 2 != 0:
        raise ValueError("K num of weights per row, must be even so two weights = 1 byte")
    
    if torch.any((quantized_weights < -7) | (quantized_weights > 7 )):
        raise ValueError("quantized_weights val must be between [-7, 7]
    
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
    
      