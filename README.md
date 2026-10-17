# Fused W4A16 linear (T4)

PyTorch CUDA operator for LLM decode:

```text
FP16 activations [M, K]  x  packed INT4 weights [N, K]  ->  FP16 output [M, N]
```

Unpack, dequantize, and multiply run in one kernel. A full FP16 weight matrix is
never written to global memory. Inference only; no backward. Target GPU: NVIDIA
T4 (`sm_75`). Primary shape is single-token decode (`M = 1`).

## Format

Symmetric group-wise INT4 along `K`. Default `group_size` is 128.

- Codes are signed integers in `[-7, 7]`.
- Two codes per `uint8` byte; even `k` is the low nibble, odd `k` the high nibble.
- Negative codes use four-bit two's complement.
- Per-group scale: `max(abs(w)) / 7`, or `1.0` for an all-zero group.

Stored tensors:

```text
packed_weights: uint8    [N, K / 2]
scales:         float16  [N, K / group_size]
```

## Status

Implemented:

- Python quantize, pack/unpack, and `linear_reference`
- `QuantizedLinear` (CPU uses the reference; CUDA uses the fused op)
- Versioned per-tensor checkpoint (`format=w4a16-linear`, `version=1`)
- `replace_linear_layers`: skips layers with `in_features % group_size != 0` and
  emits a warning
- Naive fused CUDA kernel, plus a fast `M = 1` decode kernel (one warp owns one
  output channel) when `M == 1` and `group_size == 128`

Kernel correctness is match against Python `linear_reference` on the **same**
packed codes and scales. Difference versus the original FP16 `nn.Linear` is
quantization error, not kernel error.

CPU tests run without a GPU. CUDA tests skip if CUDA or the extension is missing.
The fast decode kernel has not been published with T4 numbers yet.

## Install

Requires PyTorch with CUDA and `nvcc`. Intended numbers are for a T4 (`sm_75`);
other Turing GPUs can run the same binary. `setup.py` already passes
`-gencode arch=compute_75,code=sm_75`.

```bash
python setup.py build_ext --inplace
pytest tests/
```

TinyLlama scripts also need `transformers`. Operator benches need only PyTorch and the CUDA extension.

## v1 contract

- Activations: contiguous CUDA `float16`
- Weights: contiguous CUDA `uint8` packed layout
- Scales: contiguous CUDA `float16`
- Real fused path: `K % 128 == 0`, `group_size` default 128
- No autograd (`requires_grad` is rejected)
- Bias, if present, is stored on `QuantizedLinear` and added in Python after the op

## API

From `w4a16`:

| Symbol | Role |
| --- | --- |
| `quantize_and_pack_weights` | FP16/FP32 `[N, K]` -> packed weights + scales |
| `QuantizedLinear` | Inference module; `from_float` converts `nn.Linear` |
| `replace_linear_layers` | Recursively replace compatible `nn.Linear` |
| `linear_cuda` | Fused CUDA op (`torch.ops.w4a16.linear`) |
| `cuda_extension_available` | Whether `w4a16._C` loaded |
| `save_quantized_weight_checkpoint` | Write one packed tensor + metadata |
| `load_quantized_weight_checkpoint` | Load and validate one packed tensor |

CPU path uses `linear_reference` (unpack, dequant to a temporary matrix, matmul).
That reconstructs FP16 weights in memory; the CUDA kernels do not.

## Scripts

These live under `scripts/` and are meant for TinyLlama
(`TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

```bash
python scripts/quantize_tinyllama.py
python scripts/generate.py --mode fp16
python scripts/generate.py --mode fused
python scripts/generate.py --mode fused --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/benchmark.py
```

`generate.py` compares generation quality of fused W4A16 vs FP16. That is not a
kernel-correctness test.

`benchmark.py` runs three isolated baselines, sequentially, with CUDA events:

1. Resident FP16 weights + linear
2. Dequantize packed weights to FP16 on every call + linear
3. Fused W4A16 kernel

It reports median and p95 latency, peak VRAM, and weight storage. Publish numbers
from an on-demand `g4dn.xlarge` (one T4).

## Results

Fill in after a T4 run. Do not copy numbers from other GPUs.

| shape (M, K, N) | baseline | median ms | p95 ms | peak MiB | weight MiB |
| --- | --- | --- | --- | --- | --- |
| | resident FP16 | | | | |
| | dequant every call | | | | |
| | fused kernel | | | | |
