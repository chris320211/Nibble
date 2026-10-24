# Fused W4A16 linear (T4)

CUDA operator for LLM decode on NVIDIA T4 (`sm_75`):

```text
FP16 activations [M, K]  x  packed INT4 weights [N, K]  ->  FP16 output [M, N]
```

Unpack, dequant, and multiply in one kernel. No full FP16 weight matrix in
global memory. Inference only. Primary shape: single-token decode (`M = 1`),
with a small-batch kernel for `M = 2..8`.

Kernel correctness is match vs Python `linear_reference` on the same packed
codes. Difference vs FP16 `nn.Linear` is quantization error, not kernel error.

## Scope: a kernel project, not a quantization method

The quantization here is deliberately simple and **lossy**: plain
round-to-nearest, symmetric, per-group absmax scaling down to 4 bits. It is not
a novel method and not production quality: no calibration data, no outlier
handling, no error compensation (GPTQ, AWQ, and similar do all of that). Model
quality is expected to drop and is not measured here.

The point of the project is the **kernel**: given weights already packed to
INT4, run the linear layer as fast and with as little memory as possible on a
T4. The quantizer only exists to produce packed weights in a known format.

## Results (Colab T4)

Google Colab Tesla T4, `group_size = 128`. Do not copy numbers from other GPUs.

### TinyLlama generate

> **Stale methodology.** The table below is a single cold run with no warmup,
> and FP16 `2+2` stopped at 9 tokens, so the rows are not comparable. It will be
> replaced with numbers from the current `scripts/generate.py` (see
> [Generate benchmark methodology](#generate-benchmark-methodology)).

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
need `transformers`. The bitsandbytes comparison needs `bitsandbytes`
(optional; skipped if missing).

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
python scripts/generate.py --mode bnb-nf4
python scripts/generate.py --mode all --checkpoint checkpoints/tinyllama-w4a16.pt
python scripts/benchmark.py --m 1 2 4 8
python scripts/profile_kernel.py --baselines fused resident_fp16
```

`--mode all` runs FP16, bnb-nf4 (if installed), dequant, then fused
sequentially. Operator `tok/s` in `benchmark.py` is `1000 * M / median_ms` for
one linear, not full generate.

### Generate benchmark methodology

Per mode and prompt, `scripts/generate.py`:

1. Runs one untimed generate that stops at EOS and prints its text. This also
   warms up the CUDA context, cuBLAS, and the caching allocator.
2. Runs `--warmup` more untimed generates (default 1).
3. Times `--runs` generates (default 5) of prefill only (`max_new_tokens = 1`)
   and `--runs` generates forced to exactly `--max-new-tokens` tokens
   (`min_new_tokens` ignores EOS), so every mode decodes the same length.

It reports medians: `prefill_ms`, `decode_ms = total_ms - prefill_ms`,
`decode tok/s = (new_tokens - 1) / decode_s` (the number the `M = 1` kernel
affects), and `e2e tok/s` (includes prefill).

### Comparison with bitsandbytes NF4

`bnb_nf4` (operator) and `--mode bnb-nf4` (generate) use bitsandbytes NF4, the
4-bit path behind Hugging Face `load_in_4bit`. Formats differ, so this compares
kernels, not quantization quality:

| | this repo | bitsandbytes NF4 |
| --- | --- | --- |
| codes | INT4 `[-7, 7]`, round-to-nearest | NF4 lookup table |
| scales | FP16 per 128 weights (~4.1 bits/weight) | FP32 absmax per 64 (~4.5 bits/weight) |
| `lm_head` | quantized | FP16 (transformers default) |

## Profile

Neither script needs Nsight, so both run on Colab, where `ncu` fails with
`ERR_NVGPUCTRPERM`. Rebuild after `setup.py` so NVCC `-lineinfo` is present for
source-mapped ncu later.

- `profile_kernel.py` reports `kernel_us`, GPU time per call over 100
  back-to-back calls replayed from a CUDA graph. Weight copies are rotated so
  every call reads DRAM, not L2 (T4 L2 is 4 MiB, larger than packed attention
  weights). `eager_us` is a single call timed alone, which adds Python dispatch
  and launch latency (about 30 us on Colab). `%peak` is against spec DRAM
  (looked up from the GPU name) and `%copy` against a measured device copy.
- `inspect_kernel.py` reads the built `.so` with `cuobjdump`: registers, spills,
  occupancy, global load widths, barriers, shuffles, and convert/FMA counts per
  kernel. SASS is written to `profile_results/sass/`.

```bash
python scripts/profile_kernel.py --baselines fused resident_fp16
python scripts/inspect_kernel.py
# later, when ncu works:
ncu --target-processes all \
    --profile-from-start off \
    --kernel-name regex:int4_linear_kernel_decode \
    --launch-count 3 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section WarpStateStats --section Occupancy \
    -o profile_results/ncu_decode \
    python scripts/profile_kernel.py --baselines fused --warmup 10 --iters 5 --cuda-profiler
nsys profile -o profile_results/nsys_decode --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    python scripts/profile_kernel.py --baselines fused --cuda-profiler
```

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
- Fused path: `M <= 8`, `K % 128 == 0`, `group_size = 128`, activations and
  packed weights 16-byte aligned (one warp per output channel; `M = 1` wide-load
  decode kernel, `M = 2..8` small-batch kernel that unpacks each weight once and
  reuses it for every row)
- `M = 1` decode: each lane loads 16 packed bytes per step, reads activations
  from L1 with no `__syncthreads`, and the warp reduces once at the end.
  `W4A16_DECODE=v1` selects the original shared-memory decode kernel for A/B runs
- `QuantizedLinear` with `M > 8` or `fused=False`: dequant + cuBLAS. Calling
  `linear_cuda` directly with `M > 8`, another group size, or misaligned
  tensors uses a naive one-thread-per-output kernel
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
