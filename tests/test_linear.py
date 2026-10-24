import pytest
import torch
import torch.nn.functional as F

from w4a16.cuda import cuda_extension_available
from w4a16.linear import QuantizedLinear
from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import dequantize_weights, linear_reference


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


def test_quantized_linear_empty_shapes() -> None:
    layer = QuantizedLinear.empty(
        out_features=2,
        in_features=4,
        group_size=4,
        bias=True,
        fused=False,
    )

    assert layer.packed_weights.shape == (2, 2)
    assert layer.packed_weights.dtype == torch.uint8
    assert layer.scales.shape == (2, 1)
    assert layer.scales.dtype == torch.float16
    assert layer.bias is not None
    assert layer.bias.shape == (2,)
    assert layer.fused is False


def test_quantized_linear_empty_rejects_unaligned_in_features() -> None:
    with pytest.raises(ValueError, match="K must be divisible by group_size"):
        QuantizedLinear.empty(out_features=2, in_features=6, group_size=4)


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


def test_quantized_linear_rejects_float32_activations() -> None:
    weights = torch.tensor(
        [
            [0.25, -0.375, 0.875, -0.125],
            [0.25, 0.5, -0.25, 1.75],
        ],
        dtype=torch.float16,
    )
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=4)
    layer = QuantizedLinear(packed_weights, scales, group_size=4)
    activations = torch.ones((2, 4), dtype=torch.float32)

    with pytest.raises(TypeError, match="activations must have dtype torch.float16"):
        layer(activations)


def test_quantized_linear_rejects_requires_grad() -> None:
    weights = torch.tensor(
        [
            [0.25, -0.375, 0.875, -0.125],
            [0.25, 0.5, -0.25, 1.75],
        ],
        dtype=torch.float16,
    )
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=4)
    layer = QuantizedLinear(packed_weights, scales, group_size=4)
    activations = torch.ones((2, 4), dtype=torch.float16, requires_grad=True)

    with pytest.raises(RuntimeError, match="activations must not require grad"):
        layer(activations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_quantized_linear_cuda_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((2, 128), dtype=torch.float16, device=device)
    weights = torch.randn((8, 128), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)
    bias = torch.randn((8,), dtype=torch.float16, device=device)
    layer = QuantizedLinear(
        packed_weights,
        scales,
        group_size=128,
        bias=bias,
    )

    result = layer(activations)
    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    expected = expected + bias

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_quantized_linear_cuda_accepts_noncontiguous_activations() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = torch.randn((8, 128), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)
    layer = QuantizedLinear(packed_weights, scales, group_size=128)
    activations = torch.randn((2, 256), dtype=torch.float16, device=device)[:, :128]
    assert not activations.is_contiguous()

    result = layer(activations)
    expected = linear_reference(
        activations.contiguous(),
        packed_weights,
        scales,
        group_size=128,
    )
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_quantized_linear_cuda_decode_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 128), dtype=torch.float16, device=device)
    weights = torch.randn((8, 128), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)
    layer = QuantizedLinear(packed_weights, scales, group_size=128, fused=True)

    result = layer(activations)
    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_quantized_linear_cuda_prefill_matches_dequant_linear() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    # M > FUSED_MAX_ROWS, so the module takes the dequant + cuBLAS path.
    activations = torch.randn((16, 128), dtype=torch.float16, device=device)
    weights = torch.randn((8, 128), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)
    layer = QuantizedLinear(packed_weights, scales, group_size=128, fused=True)
    weight_fp16 = dequantize_weights(
        packed_weights,
        scales,
        group_size=128,
    ).to(dtype=torch.float16)

    result = layer(activations)
    expected = F.linear(activations, weight_fp16)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_quantized_linear_cuda_dequant_mode_matches_dequant_linear() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 128), dtype=torch.float16, device=device)
    weights = torch.randn((8, 128), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)
    layer = QuantizedLinear(packed_weights, scales, group_size=128, fused=False)
    weight_fp16 = dequantize_weights(
        packed_weights,
        scales,
        group_size=128,
    ).to(dtype=torch.float16)

    result = layer(activations)
    expected = F.linear(activations, weight_fp16)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
