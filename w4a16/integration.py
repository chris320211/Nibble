import warnings

import torch

from w4a16.linear import QuantizedLinear


def replace_linear_layers(
    module: torch.nn.Module,
    group_size: int = 128,
    prefix: str = "",
    fused: bool = True,
    packed_only: bool = False,
) -> int:
    """Replace compatible torch.nn.Linear children with QuantizedLinear modules.

    ``packed_only=True`` builds empty packed storage from layer shapes and does
    not read the source FP16 weights. The caller must load packed tensors
    afterwards.
    """
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    replacement_count = 0
    for child_name, child in list(module.named_children()):
        qualified_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, QuantizedLinear):
            continue

        if isinstance(child, torch.nn.Linear):
            if child.in_features % group_size != 0:
                warnings.warn(
                    f"Skipping linear layer '{qualified_name}' because "
                    f"in_features={child.in_features} is not divisible by "
                    f"group_size={group_size}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if packed_only:
                quantized_child = QuantizedLinear.empty(
                    out_features=child.out_features,
                    in_features=child.in_features,
                    group_size=group_size,
                    bias=child.bias is not None,
                    fused=fused,
                    device=child.weight.device,
                )
            else:
                quantized_child = QuantizedLinear.from_float(
                    child,
                    group_size=group_size,
                    fused=fused,
                )
            quantized_child.train(child.training)
            setattr(module, child_name, quantized_child)
            replacement_count += 1
            continue

        replacement_count += replace_linear_layers(
            child,
            group_size,
            prefix=qualified_name,
            fused=fused,
            packed_only=packed_only,
        )

    return replacement_count


def _restore_computed_buffers(module: torch.nn.Module) -> None:
    """Fill buffers that ``state_dict`` does not store, such as RoPE ``inv_freq``.

    Those buffers are ``persistent=False``. ``to_empty`` allocates storage for
    them but leaves the values uninitialized.
    """
    for child in module.modules():
        compute = getattr(child, "compute_default_rope_parameters", None)
        if compute is None or not hasattr(child, "config"):
            continue
        inv_freq_buffer = getattr(child, "inv_freq", None)
        if inv_freq_buffer is None:
            continue
        inv_freq, attention_scaling = compute(child.config, inv_freq_buffer.device)
        inv_freq_buffer.copy_(
            inv_freq.to(device=inv_freq_buffer.device, dtype=inv_freq_buffer.dtype)
        )
        original = getattr(child, "original_inv_freq", None)
        if original is not None:
            original.copy_(inv_freq.to(device=original.device, dtype=original.dtype))
        child.attention_scaling = attention_scaling


def load_packed_state_dict(
    module: torch.nn.Module,
    state_dict: dict,
    group_size: int = 128,
    device: str | torch.device = "cpu",
    fused: bool = True,
) -> int:
    """Replace Linear layers with empty packed modules and load INT4 weights.

    ``module`` may live on the meta device so FP16 linear weights are never
    materialized. Remaining non-linear tensors (embeddings, norms) are
    allocated on ``device`` and filled from ``state_dict``.
    """
    replacement_count = replace_linear_layers(
        module,
        group_size=group_size,
        fused=fused,
        packed_only=True,
    )
    module.to_empty(device=device)
    module.load_state_dict(state_dict)
    _restore_computed_buffers(module)
    return replacement_count
