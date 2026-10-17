# Fused W4A16 linear (T4)

CUDA operator for LLM decode on NVIDIA T4 (`sm_75`):

```text
FP16 activations [M, K]  x  packed INT4 weights [N, K]  ->  FP16 output [M, N]
```

Unpack, dequant, and multiply in one kernel. No full FP16 weight matrix in
global memory. Inference only. Primary shape: single-token decode (`M = 1`).

Kernel correctness is match vs Python `linear_reference` on the same packed
codes. Difference vs FP16 `nn.Linear` is quantization error, not kernel error.

## Results (Colab T4)

Google Colab Tesla T4, `group_size = 128`. Do not copy numbers from other GPUs.

### TinyLlama generate

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, greedy, `max_new_tokens = 32`, chat
template. Fused: `M = 1` kernel, `M > 1` dequant+cuBLAS. FP16 loads Hugging Face
weights; fused/dequant load packed weights only (155 layers). `tok/s` =
`1000 * new_tokens / generate_ms` (includes prefill). One run, no generate
warmup.

Prompts: `Write a short haiku about GPUs.` and `What is 2+2?`

| prompt | baseline | tokens | ms | tok/s | allocated MiB | peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| haiku | FP16 | 32 | 2206.9 | 14.50 | 2100.18 | 2111.05 |
| haiku | dequant | 32 | 5898.9 | 5.42 | 642.59 | 1216.48 |
| haiku | fused | 32 | 1326.1 | 24.13 | 642.59 | 751.22 |
| 2+2 | FP16 | 9 | 232.2 | 38.76 | 2100.18 | 2111.05 |
| 2+2 | dequant | 32 | 5614.7 | 5.70 | 642.59 | 1216.48 |
| 2+2 | fused | 32 | 971.5 | 32.94 | 642.59 | 751.22 |

- Packed load: **643 MiB** vs **2100 MiB** FP16 (~3.3× smaller). Fused peak
  **751 MiB** vs **2111 MiB** (~2.8× less). Dequant peaks higher because every
  call unpacks a full matrix (`lm_head` is largest).
- Haiku fused is ~4.5× faster than dequant. Compare haiku rows for tok/s (all
  32 tokens). FP16 `2+2` stopped at 9 tokens.
- FP16 haiku (14.50 tok/s) is a cold first generate after `from_pretrained`;
  the next FP16 prompt is 38.76 tok/s. Fused is not faster than warm FP16.
- Fused and dequant greedy text matched; FP16 wording differed (quantization).

### Operator (`M = 1`)

`scripts/benchmark.py`, warmup 5, iters 20. Weight MiB is resident FP16 storage,
or packed codes + scales.

| shape (M, K, N) | baseline | median ms | p95 ms | peak MiB | weight MiB |
| --- | --- | --- | --- | --- | --- |
| (1, 2048, 2048) attention-like | FP16 | 0.0600 | 0.0654 | 16.14 | 8.00 |
| (1, 2048, 2048) attention-like | dequant | 0.8120 | 0.8189 | 46.32 | 2.06 |
| (1, 2048, 2048) attention-like | fused | 0.0676 | 0.0737 | 10.20 | 2.06 |
| (1, 2048, 5632) mlp up-like | FP16 | 0.1188 | 0.1356 | 30.15 | 22.00 |
| (1, 2048, 5632) mlp up-like | dequant | 2.1220 | 2.1291 | 113.16 | 5.67 |
| (1, 2048, 5632) mlp up-like | fused | 0.1509 | 0.1721 | 13.82 | 5.67 |
| (1, 256, 8) small sanity | FP16 | 0.0320 | 0.0444 | 8.13 | 0.0039 |
| (1, 256, 8) small sanity | dequant | 0.3030 | 0.3152 | 8.15 | 0.0010 |
| (1, 256, 8) small sanity | fused | 0.0473 | 0.1441 | 8.13 | 0.0010 |

On the two TinyLlama-like shapes, fused is 12×–14× faster than dequant and
13%–27% slower than resident FP16 cuBLAS. Packed storage is ~3.9× smaller.
T4 has no INT4 Tensor Cores; the kernel unpacks nibbles and does FP32 math.

Colab: `notebooks/colab_t4.ipynb`.

## Install

PyTorch with CUDA and `nvcc`. `setup.py` targets `sm_75`. TinyLlama scripts
need `transformers`.

```bash
python setup.py build_ext --inplace
pytest tests/
```

CPU tests run without a GPU. CUDA tests skip if CUDA or the extension is missing.

```bash
python scripts/quantize_tinyllama.py
python scripts/generate.py --mode fp16
python scripts/generate.py --mode dequant --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/generate.py --mode fused --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/generate.py --mode all --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/benchmark.py
```

`--mode all` runs FP16, dequant, then fused sequentially. Operator `tok/s` in
`benchmark.py` is `1000 / median_ms` for one `M = 1` linear, not full generate.

## Format

Symmetric group-wise INT4 along `K`. Default `group_size` is 128. Codes in
`[-7, 7]`; two per `uint8` (even `k` low nibble, odd `k` high nibble, four-bit
two's complement). Scale: `max(abs(w)) / 7`, or `1.0` for an all-zero group.

```text
packed_weights: uint8    [N, K / 2]
scales:         float16  [N, K / group_size]
```

## Contract

- CUDA `float16` activations, packed `uint8` weights, `float16` scales; all contiguous
- Fused path: `M = 1`, `K % 128 == 0`, `group_size = 128` (one warp per output channel)
- `M > 1` or `fused=False`: dequant + cuBLAS
- No autograd. Bias, if any, is added in Python after the op

## API

| Symbol | Role |
| --- | --- |
| `quantize_and_pack_weights` | FP16/FP32 `[N, K]` → packed weights + scales |
| `QuantizedLinear` | Inference module; `from_float` / `empty` |
| `replace_linear_layers` | Replace compatible `nn.Linear`; `packed_only=True` skips quantize |
| `load_packed_state_dict` | Empty-replace, `to_empty`, load packed weights |
| `linear_cuda` | Fused CUDA op |
| `cuda_extension_available` | Whether `w4a16._C` loaded |
| `save_quantized_weight_checkpoint` | One packed tensor + metadata |
| `load_quantized_weight_checkpoint` | Load and validate one packed tensor |
