import warnings

import torch

from w4a16.linear import QuantizedLinear


def replace_linear_layers(
    module: torch.nn.Module,
    group_size: int = 128,
    prefix: str = "",
    fused: bool = True,
) -> int:
    """Replace compatible torch.nn.Linear children with QuantizedLinear modules."""
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
        )

    return replacement_count
