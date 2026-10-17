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
Operator-level T4 numbers for the fast decode kernel are in Results.

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
- `QuantizedLinear` on CUDA: `M = 1` uses the fused decode kernel; `M > 1`
  dequantizes a temporary FP16 weight matrix and uses cuBLAS
- `fused=False` uses dequant + cuBLAS on every call (generate baseline)
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
The fused CUDA decode kernel does not reconstruct a full FP16 weight matrix.
Prefill (`M > 1`) and `fused=False` do, as a temporary cuBLAS operand.

## Scripts

These live under `scripts/` and are meant for TinyLlama
(`TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

```bash
python scripts/quantize_tinyllama.py
python scripts/generate.py --mode fp16
python scripts/generate.py --mode dequant --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/generate.py --mode fused --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/generate.py --mode all --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/benchmark.py
```

`generate.py` times TinyLlama generate for resident FP16, dequant-every-call +
cuBLAS, and fused decode (`M = 1` kernel, `M > 1` dequant + cuBLAS). That is
quality and end-to-end tok/s, not a kernel-correctness test. `--mode all` runs
the three baselines sequentially on the same GPU.

`benchmark.py` runs three isolated baselines, sequentially, with CUDA events:

1. Resident FP16 weights + linear
2. Dequantize packed weights to FP16 on every call + linear
3. Fused W4A16 kernel

It reports median and p95 latency, peak VRAM, and weight storage. `tok/s` in that
script is `1000 / median_ms` for one linear at `M = 1`, not full TinyLlama
generate. Colab notebook: `notebooks/colab_t4.ipynb`.

## Results

Operator-level numbers from Google Colab, NVIDIA Tesla T4 (`sm_75`), `M = 1`,
warmup 5, iters 20, `group_size = 128`. Do not copy numbers from other GPUs.

Weight MiB is resident FP16 storage for `resident FP16`, and packed INT4 storage
(codes + FP16 scales) for the other two.

| shape (M, K, N) | baseline | median ms | p95 ms | peak MiB | weight MiB |
| --- | --- | --- | --- | --- | --- |
| (1, 2048, 2048) attention-like | resident FP16 | 0.0600 | 0.0654 | 16.14 | 8.00 |
| (1, 2048, 2048) attention-like | dequant every call | 0.8120 | 0.8189 | 46.32 | 2.06 |
| (1, 2048, 2048) attention-like | fused kernel | 0.0676 | 0.0737 | 10.20 | 2.06 |
| (1, 2048, 5632) mlp up-like | resident FP16 | 0.1188 | 0.1356 | 30.15 | 22.00 |
| (1, 2048, 5632) mlp up-like | dequant every call | 2.1220 | 2.1291 | 113.16 | 5.67 |
| (1, 2048, 5632) mlp up-like | fused kernel | 0.1509 | 0.1721 | 13.82 | 5.67 |
| (1, 256, 8) small sanity | resident FP16 | 0.0320 | 0.0444 | 8.13 | 0.0039 |
| (1, 256, 8) small sanity | dequant every call | 0.3030 | 0.3152 | 8.15 | 0.0010 |
| (1, 256, 8) small sanity | fused kernel | 0.0473 | 0.1441 | 8.13 | 0.0010 |

On these two TinyLlama-like decode shapes, fused is about 12×–14× faster than
dequant-every-call and about 13%–27% slower than resident FP16 cuBLAS. Packed
weight storage is about 3.9× smaller than FP16. TinyLlama generate quality and
end-to-end tok/s are not in this table; run `scripts/generate.py` on a T4.
