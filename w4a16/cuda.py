import importlib

import torch


try:
    importlib.import_module("w4a16._C")
except (ImportError, OSError) as error:
    _EXTENSION_LOAD_ERROR: Exception | None = error
else:
    _EXTENSION_LOAD_ERROR = None


def cuda_extension_available() -> bool:
    """Return whether the compiled W4A16 CUDA extension can be loaded."""
    return _EXTENSION_LOAD_ERROR is None


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
    if _EXTENSION_LOAD_ERROR is not None:
        raise RuntimeError(
            "The W4A16 CUDA extension is not built. Build it with "
            "`python setup.py build_ext --inplace`."
        ) from _EXTENSION_LOAD_ERROR

    return torch.ops.w4a16.linear(
        activations,
        packed_weights,
        scales,
        group_size,
    )
