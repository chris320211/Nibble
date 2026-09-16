import torch

from w4a16.quantization import pack_int4
from w4a16.reference import dequantize_weights, linear_reference


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


def test_linear_reference_known_values() -> None:
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 2.0, 0.5],
        ],
        dtype=torch.float16,
    )
    quantized_weights = torch.tensor(
        [
            [2, -3, 7, -1],
            [1, 2, -1, 4],
        ],
        dtype=torch.int8,
    )
    packed_weights = pack_int4(quantized_weights)
    scales = torch.tensor(
        [
            [0.125],
            [0.25],
        ],
        dtype=torch.float16,
    )

    result = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=4,
    )
    expected = torch.tensor(
        [
            [1.625, 4.5],
            [1.4375, -0.25],
        ],
        dtype=torch.float16,
    )

    assert torch.equal(result, expected)
