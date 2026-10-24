import importlib

import torch


try:
    importlib.import_module("w4a16._C")
except (ImportError, OSError) as error:
    _EXTENSION_LOAD_ERROR: Exception | None = error
else:
    _EXTENSION_LOAD_ERROR = None


if _EXTENSION_LOAD_ERROR is None:
    # Fake (meta) implementations: output shape and dtype only. torch.compile
    # traces the model with fake tensors, so it needs these to see through the
    # C++ ops (e.g. generate with a static cache and CUDA graphs).
    @torch.library.register_fake("w4a16::linear")
    def _linear_fake(
        activations: torch.Tensor,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        return activations.new_empty((activations.shape[0], packed_weights.shape[0]))

    @torch.library.register_fake("w4a16::dequantize")
    def _dequantize_fake(
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        return packed_weights.new_empty(
            (packed_weights.shape[0], packed_weights.shape[1] * 2),
            dtype=torch.float16,
        )


def cuda_extension_available() -> bool:
    """Return whether the compiled W4A16 CUDA extension can be loaded."""
    return _EXTENSION_LOAD_ERROR is None


def _require_extension() -> None:
    if _EXTENSION_LOAD_ERROR is not None:
        raise RuntimeError(
            "The W4A16 CUDA extension is not built. Build it with "
            "`python setup.py build_ext --inplace`."
        ) from _EXTENSION_LOAD_ERROR


def linear_cuda(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Run the fused W4A16 CUDA linear operator.

    group_size=128 uses the warp-per-channel kernels (M=1 decode, M=2..8 small
    batch). Other shapes fall back to the naive one-thread-per-output kernel.
    """
    _require_extension()
    return torch.ops.w4a16.linear(
        activations,
        packed_weights,
        scales,
        group_size,
    )


def dequantize_cuda(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Unpack INT4 weights to an FP16 [N, K] matrix in one CUDA pass.

    Bit-identical to ``dequantize_weights(...).to(torch.float16)``. Needs
    group_size to be a multiple of 32 and 16-byte aligned packed weights.
    """
    _require_extension()
    return torch.ops.w4a16.dequantize(packed_weights, scales, group_size)
