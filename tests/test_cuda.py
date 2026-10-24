import pytest
import torch

from w4a16.cuda import cuda_extension_available, linear_cuda
from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import linear_reference


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
def test_cuda_linear_original_decode_kernel_matches_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_extension_available():
        pytest.skip("W4A16 CUDA extension is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activations = torch.randn((1, 2048), dtype=torch.float16, device=device)
    weights = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=128)

    wide = linear_cuda(activations, packed_weights, scales, group_size=128)
    monkeypatch.setenv("W4A16_DECODE", "v1")
    original = linear_cuda(activations, packed_weights, scales, group_size=128)

    torch.testing.assert_close(wide, original, rtol=1e-2, atol=1e-2)


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
