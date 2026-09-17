import torch

from w4a16.linear import QuantizedLinear


def replace_linear_layers(
    module: torch.nn.Module,
    group_size: int = 128,
) -> int:
    """Replace compatible torch.nn.Linear children with QuantizedLinear modules."""
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    replacement_count = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, QuantizedLinear):
            continue

        if isinstance(child, torch.nn.Linear):
            if child.in_features % group_size != 0:
                continue

            quantized_child = QuantizedLinear.from_float(
                child,
                group_size=group_size,
            )
            quantized_child.train(child.training)
            setattr(module, child_name, quantized_child)
            replacement_count += 1
            continue

        replacement_count += replace_linear_layers(child, group_size)

    return replacement_count
