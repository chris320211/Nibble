import torch
import torch.nn.functional as F

from w4a16.cuda import linear_cuda
from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import dequantize_weights, linear_reference


class QuantizedLinear(torch.nn.Module):
    """Inference-only linear layer backed by packed INT4 weights."""

    def __init__(
        self,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 128,
        bias: torch.Tensor | None = None,
        fused: bool = True,
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
        self.fused = fused

    @classmethod
    def from_float(
        cls,
        layer: torch.nn.Linear,
        group_size: int = 128,
        fused: bool = True,
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
            fused=fused,
        )

    @classmethod
    def empty(
        cls,
        out_features: int,
        in_features: int,
        group_size: int = 128,
        bias: bool = False,
        fused: bool = True,
        device: torch.device | str | None = None,
    ) -> "QuantizedLinear":
        """Allocate packed INT4 storage without reading FP16 weights."""
        if out_features <= 0 or in_features <= 0:
            raise ValueError("out_features and in_features must be greater than zero")
        if in_features % 2 != 0:
            raise ValueError("in_features must be even")
        if group_size <= 0:
            raise ValueError("group_size must be greater than zero")
        if in_features % group_size != 0:
            raise ValueError("K must be divisible by group_size")

        if device is None:
            device = torch.device("cpu")
        packed_weights = torch.empty(
            (out_features, in_features // 2),
            dtype=torch.uint8,
            device=device,
        )
        scales = torch.empty(
            (out_features, in_features // group_size),
            dtype=torch.float16,
            device=device,
        )
        bias_tensor = None
        if bias:
            bias_tensor = torch.empty(
                (out_features,),
                dtype=torch.float16,
                device=device,
            )
        return cls(
            packed_weights,
            scales,
            group_size=group_size,
            bias=bias_tensor,
            fused=fused,
        )

    def _dequant_linear(self, activations: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weights(
            self.packed_weights,
            self.scales,
            self.group_size,
        ).to(dtype=torch.float16)
        return F.linear(activations, weight)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim < 1:
            raise ValueError("activations must have at least one dimension")
        if activations.shape[-1] != self.in_features:
            raise ValueError("activations last dimension must equal in_features")
        if activations.dtype != torch.float16:
            raise TypeError("activations must have dtype torch.float16")
        if activations.requires_grad:
            raise RuntimeError("activations must not require grad")
        if activations.device != self.packed_weights.device:
            raise ValueError("activations and packed_weights must share a device")

        output_shape = (*activations.shape[:-1], self.out_features)
        # HuggingFace generate often yields non-contiguous views; the CUDA op
        # still requires a packed layout, so the module copies here.
        flat_activations = activations.contiguous().reshape(-1, self.in_features)
        rows = flat_activations.shape[0]
        if flat_activations.is_cuda:
            if not self.packed_weights.is_contiguous():
                raise ValueError("packed_weights must be contiguous")
            if not self.scales.is_contiguous():
                raise ValueError("scales must be contiguous")
            # Decode (M=1) uses the fused kernel. Prefill (M>1) dequantizes a
            # temporary FP16 weight matrix and uses cuBLAS so generate is not
            # stuck on the naive fused fallback.
            if self.fused and rows == 1:
                flat_output = linear_cuda(
                    flat_activations,
                    self.packed_weights,
                    self.scales,
                    self.group_size,
                )
            else:
                flat_output = self._dequant_linear(flat_activations)
        else:
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
