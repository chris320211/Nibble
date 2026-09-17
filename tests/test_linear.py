import torch

from w4a16.linear import QuantizedLinear
from w4a16.quantization import quantize_and_pack_weights


def test_quantized_linear_known_values() -> None:
    weights = torch.tensor(
        [
            [0.25, -0.375, 0.875, -0.125],
            [0.25, 0.5, -0.25, 1.75],
        ],
        dtype=torch.float16,
    )
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=4)
    layer = QuantizedLinear(packed_weights, scales, group_size=4)
    activations = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, 0.0, 2.0, 0.5],
            ]
        ],
        dtype=torch.float16,
    )

    result = layer(activations)
    expected = torch.tensor(
        [
            [
                [1.625, 7.5],
                [1.4375, 0.125],
            ]
        ],
        dtype=torch.float16,
    )

    assert torch.equal(result, expected)
    assert "packed_weights" in layer.state_dict()
    assert "scales" in layer.state_dict()


def test_quantized_linear_from_float() -> None:
    layer = torch.nn.Linear(4, 2, bias=True, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.25, -0.375, 0.875, -0.125],
                    [0.25, 0.5, -0.25, 1.75],
                ],
                dtype=torch.float16,
            )
        )
        layer.bias.copy_(torch.tensor([0.5, -0.25], dtype=torch.float16))

    quantized_layer = QuantizedLinear.from_float(layer, group_size=4)
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 2.0, 0.5],
        ],
        dtype=torch.float16,
    )

    result = quantized_layer(activations)
    expected = layer(activations)

    assert torch.equal(result, expected)
    assert torch.equal(quantized_layer.bias, layer.bias)
