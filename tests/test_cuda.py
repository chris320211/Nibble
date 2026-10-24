import pytest
import torch

from w4a16.cuda import cuda_extension_available, dequantize_cuda, linear_cuda
from w4a16.linear import QuantizedLinear
from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import dequantize_weights, linear_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_naive_cuda_linear_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((4, 256), dtype=torch.float16, device=device)
    weights = torch.randn((8, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_naive_cuda_linear_decode_shape_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 256), dtype=torch.float16, device=device)
    weights = torch.randn((8, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_linear_tinyllama_attention_decode_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 2048), dtype=torch.float16, device=device)
    weights = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_linear_tinyllama_mlp_up_proj_decode_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 2048), dtype=torch.float16, device=device)
    weights = torch.randn((5632, 2048), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_linear_decode_n_not_divisible_by_8_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 256), dtype=torch.float16, device=device)
    weights = torch.randn((7, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_linear_tinyllama_mlp_down_proj_decode_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    # K = 5632 is 176 chunks of 32 weights, so the last warp step is partial.
    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 5632), dtype=torch.float16, device=device)
    weights = torch.randn((2048, 5632), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_linear_decode_misaligned_activations_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    # Offset by one half (2 bytes): still contiguous, but not 16-byte aligned,
    # so the vector-load warp kernels must fall back to the naive kernel.
    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 2049), dtype=torch.float16, device=device)[:, 1:]
    assert activations.is_contiguous()
    assert activations.data_ptr() % 16 != 0
    weights = torch.randn((64, 2048), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_cuda_linear_decode_versions_match_reference(
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 2048), dtype=torch.float16, device=device)
    weights = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    monkeypatch.setenv("W4A16_DECODE", version)
    result = linear_cuda(activations, packed_weights, scales, group_size=128)

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_cuda_linear_decode_all_nibble_values_match_reference(
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    # Random bytes cover all 16 codes, including -8, in both nibbles of every
    # byte position. The quantizer only emits [-7, 7], so this exercises the
    # unpack paths beyond what quantized weights reach.
    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 1024), dtype=torch.float16, device=device)
    packed_weights = torch.randint(
        0, 256, (64, 512), dtype=torch.uint8, device=device
    )
    scales = torch.rand((64, 8), device=device).add(0.01).to(torch.float16)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    monkeypatch.setenv("W4A16_DECODE", version)
    result = linear_cuda(activations, packed_weights, scales, group_size=128)

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("rows_per_warp", ["1", "2", "4"])
@pytest.mark.parametrize(
    ("k", "n"),
    [(256, 7), (1024, 13), (2048, 2048), (5632, 2048), (2048, 5632)],
)
def test_cuda_linear_decode_multirow_matches_reference(
    rows_per_warp: str,
    k: int,
    n: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    # n = 7 and 13 leave a partial last warp (rows past N are not stored).
    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, k), dtype=torch.float16, device=device)
    weights = torch.randn((n, k), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    monkeypatch.setenv("W4A16_DECODE", "v4")
    monkeypatch.setenv("W4A16_ROWS_PER_WARP", rows_per_warp)
    result = linear_cuda(activations, packed_weights, scales, group_size=128)

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_cuda_rejects_bad_rows_per_warp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    device = torch.device("cuda")
    activations = torch.randn((1, 256), dtype=torch.float16, device=device)
    weights = torch.randn((8, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    monkeypatch.setenv("W4A16_DECODE", "v4")
    monkeypatch.setenv("W4A16_ROWS_PER_WARP", "3")
    with pytest.raises(RuntimeError):
        linear_cuda(activations, packed_weights, scales, group_size=128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_cuda_rejects_float32_activations() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((4, 256), dtype=torch.float32, device=device)
    weights = torch.randn((8, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    with pytest.raises(RuntimeError):
        linear_cuda(
            activations,
            packed_weights,
            scales,
            group_size=128,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_cuda_rejects_noncontiguous_activations() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((4, 512), dtype=torch.float16, device=device)[:, :256]
    assert not activations.is_contiguous()
    weights = torch.randn((8, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    with pytest.raises(RuntimeError):
        linear_cuda(
            activations,
            packed_weights,
            scales,
            group_size=128,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_cuda_rejects_cpu_tensors() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    activations = torch.randn((4, 256), dtype=torch.float16)
    weights = torch.randn((8, 256), dtype=torch.float16)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    with pytest.raises(RuntimeError):
        linear_cuda(
            activations,
            packed_weights,
            scales,
            group_size=128,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_cuda_rejects_mismatched_k() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((4, 256), dtype=torch.float16, device=device)
    weights = torch.randn((8, 128), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    with pytest.raises(RuntimeError):
        linear_cuda(
            activations,
            packed_weights,
            scales,
            group_size=128,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_cuda_linear_small_batch_matches_reference(rows: int) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((rows, 2048), dtype=torch.float16, device=device)
    weights = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("rows", [2, 8])
def test_cuda_linear_small_batch_n_not_divisible_by_8_matches_reference(
    rows: int,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((rows, 256), dtype=torch.float16, device=device)
    weights = torch.randn((7, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_linear_above_small_batch_uses_naive_and_matches_reference() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((9, 256), dtype=torch.float16, device=device)
    weights = torch.randn((8, 256), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    expected = linear_reference(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )
    result = linear_cuda(
        activations,
        packed_weights,
        scales,
        group_size=128,
    )

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize(("k", "n"), [(256, 8), (2048, 2048), (2048, 5632), (5632, 2048)])
def test_dequantize_cuda_is_bit_identical_to_reference(
    group_size: int,
    k: int,
    n: int,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = torch.randn((n, k), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=group_size)

    expected = dequantize_weights(packed_weights, scales, group_size).to(torch.float16)
    result = dequantize_cuda(packed_weights, scales, group_size)

    assert result.dtype == torch.float16
    assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dequantize_cuda_all_nibble_values_bit_identical() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    # Random bytes hit all 16 codes (including -8) in every nibble position.
    torch.manual_seed(0)
    device = torch.device("cuda")
    packed_weights = torch.randint(0, 256, (64, 512), dtype=torch.uint8, device=device)
    scales = torch.rand((64, 8), device=device).add(0.01).to(torch.float16)

    expected = dequantize_weights(packed_weights, scales, 128).to(torch.float16)
    result = dequantize_cuda(packed_weights, scales, 128)

    assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dequantize_cuda_rejects_group_size_not_multiple_of_32() -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    device = torch.device("cuda")
    weights = torch.randn((8, 96), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=48)

    with pytest.raises(RuntimeError):
        dequantize_cuda(packed_weights, scales, 48)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("rows", [1, 4, 24])
def test_quantized_linear_traces_under_torch_compile(rows: int) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    # aot_eager traces with fake tensors (needs the register_fake impls in
    # w4a16.cuda) without compiling Triton kernels, so the test stays fast.
    # rows 1 / 4 / 24 cover the decode, small-batch, and dequant + cuBLAS paths.
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = torch.randn((256, 512), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)
    layer = QuantizedLinear(packed_weights, scales, group_size=128)
    activations = torch.randn((rows, 512), dtype=torch.float16, device=device)

    torch._dynamo.reset()
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    with torch.inference_mode():
        expected = layer(activations)
        result = compiled(activations)

    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
