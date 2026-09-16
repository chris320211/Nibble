import torch

from w4a16.quantization import pack_int4
from w4a16.reference import dequantize_weights


def test_dequantize_weights_known_values() -> None:
    quantized_weights = torch.tensor(
        [[2, -3, 7, -1, 0, 0, 0, 0]],
        dtype=torch.int8,
    )
    packed_weights = pack_int4(quantized_weights)
    scales = torch.tensor(
        [[0.125, 1.0]],
        dtype=torch.float16,
    )

    result = dequantize_weights(
        packed_weights,
        scales,
        group_size=4,
    )
    expected = torch.tensor(
        [[0.25, -0.375, 0.875, -0.125, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    assert torch.equal(result, expected)
