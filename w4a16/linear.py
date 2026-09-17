import torch

from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import linear_reference


class QuantizedLinear(torch.nn.Module):
    """Inference-only linear layer backed by packed INT4 weights."""

    def __init__(
        self,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 128,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if packed_weights.dtype != torch.uint8 or packed_weights.ndim != 2:
            raise ValueError("packed_weights must be uint8 with shape [N, K / 2]")
        if scales.dtype != torch.float16 or scales.ndim != 2:
            raise ValueError("scales must be float16 with shape [N, K / group_size]")
        if packed_weights.device != scales.device:
            raise ValueError("packed_weights and scales must share a device")
        if group_size <= 0:
            raise ValueError("group_size must be greater than zero")

        self.out_features = packed_weights.shape[0]
        self.in_features = packed_weights.shape[1] * 2
        self.group_size = group_size

        if self.in_features % group_size != 0:
            raise ValueError("K must be divisible by group_size")
        expected_scale_shape = (
            self.out_features,
            self.in_features // group_size,
        )
        if scales.shape != expected_scale_shape:
            raise ValueError(f"scales must have shape {expected_scale_shape}")

        if bias is not None:
            if bias.dtype != torch.float16 or bias.shape != (self.out_features,):
                raise ValueError("bias must be float16 with shape [N]")
            if bias.device != packed_weights.device:
                raise ValueError("bias and packed_weights must share a device")

        self.register_buffer("packed_weights", packed_weights.detach())
        self.register_buffer("scales", scales.detach())
        self.register_buffer("bias", None if bias is None else bias.detach())

    @classmethod
    def from_float(
        cls,
        layer: torch.nn.Linear,
        group_size: int = 128,
    ) -> "QuantizedLinear":
        """Convert a floating-point PyTorch linear layer into QuantizedLinear."""
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError("layer must be an instance of torch.nn.Linear")

        packed_weights, scales = quantize_and_pack_weights(
            layer.weight.detach(),
            group_size,
        )
        bias = None
        if layer.bias is not None:
            bias = layer.bias.detach().to(torch.float16)

        return cls(
            packed_weights,
            scales,
            group_size=group_size,
            bias=bias,
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim < 1:
            raise ValueError("activations must have at least one dimension")
        if activations.shape[-1] != self.in_features:
            raise ValueError("activations last dimension must equal in_features")

        output_shape = (*activations.shape[:-1], self.out_features)
        flat_activations = activations.reshape(-1, self.in_features)
        flat_output = linear_reference(
            flat_activations,
            self.packed_weights,
            self.scales,
            self.group_size,
        )
        output = flat_output.reshape(output_shape)

        if self.bias is not None:
            output = output + self.bias

        return output
