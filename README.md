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

### Operator (`M = 1`)

`scripts/profile_kernel.py`: GPU time per call over 100 back-to-back calls
replayed from a CUDA graph, with weight copies rotated so every call reads DRAM
(as in real decode, where each layer has its own weights). `%copy` is GB/s
against a device-to-device copy measured on the same GPU (about 241 GB/s on a
Colab T4; spec is 320). Weight MiB is resident FP16 storage, or packed codes +
scales.

v4 decode kernel (default, 4 rows per warp) and FP16 cuBLAS, same run. The T4
ran power-capped (`sw-power`, SM 750-1290 MHz of 1590) for nearly every row,
which is normal sustained T4 behavior:

| shape (K, N) | FP16 cuBLAS us | fused us | fused vs FP16 | fused %copy | weight MiB FP16 / packed |
| --- | --- | --- | --- | --- | --- |
| attention-like (2048, 2048) | 38.6 | 12.5 | 3.1x faster | 72% | 8.00 / 2.06 |
| mlp up-like (2048, 5632) | 97.4 | 28.2 | 3.5x faster | 88% | 22.00 / 5.67 |
| mlp down-like (5632, 2048) | 95.7 | 31.6 | 3.0x faster | 78% | 22.00 / 5.67 |

Packed weights are ~3.9x smaller than FP16, which caps the possible speedup for
a memory-bound GEMV at about 3.9x. T4 has no INT4 Tensor Cores; the kernel
unpacks nibbles and does FP32 math.

Decode kernel versions (`W4A16_DECODE`), alternated twice in one session, mean
of the two runs in us (SM clock 690-1290 MHz, power-capped):

| version | attention | mlp up | mlp down |
| --- | --- | --- | --- |
| v3: wide loads + fast unpack, 1 row per warp | 28.0 | 37.1 | 38.7 |
| v4, 1 row per warp (smaller blocks only) | 25.2 | 38.4 | 38.3 |
| v4, 2 rows per warp | 18.3 | 26.9 | 30.7 |
| v4, 4 rows per warp (default) | 18.1 | 27.9 | 30.9 |

Sharing rows cuts 25-35% against v3; smaller blocks alone do not. 2 and 4 rows
per warp tie within noise (an earlier session had 4 ahead by ~10% on the MLP
shapes; this one has them within ~4%).

Earlier steps, each measured against the previous version in its own session:
v1 (original, 2-byte loads, two `__syncthreads` per group) took 43.8 / 100.8 /
93.9 us; v2 (16-byte loads, no barriers, one reduction) took 13.5 / 38.7 /
41.7 us. v3 removed every `I2F` but barely changed time, which showed the
remaining limit was per-warp overhead and per-row activation conversion, not
unpacking; v4 shares those across rows.

T4 lowers its SM clock under its 70 W cap or when hot; memory clock does not
change, so this compute-heavy kernel slows down more than cuBLAS. The table's
`sm_mhz`, `temp_c`, and `throttle` columns flag rows that ran slowed down.

### TinyLlama generate

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, greedy, chat template, `max_new_tokens =
32` forced (every mode decodes 32 tokens), 1 warmup + median of 5 runs (see
[Generate benchmark methodology](#generate-benchmark-methodology)). Fused: v4
`M = 1` kernel for decode, dequant + cuBLAS for prefill (`M > 8`). FP16 and
bnb-nf4 load Hugging Face weights; fused and dequant load packed weights only
(155 layers). bnb-nf4 is from a `--mode all` run in the same session.

| mode | prompt | prefill ms | decode ms | decode tok/s | e2e tok/s | allocated MiB | peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | haiku | 31.4 | 830.4 | 37.33 | 37.13 | 2100.18 | 2111.17 |
| FP16 | 2+2 | 34.3 | 841.3 | 36.85 | 36.54 | 2100.18 | 2111.17 |
| bnb-nf4 | haiku | 48.0 | 1370.2 | 22.62 | 22.56 | 798.00 | 800.87 |
| bnb-nf4 | 2+2 | 50.9 | 1374.1 | 22.56 | 22.46 | 798.00 | 800.87 |
| dequant | haiku | 182.9 | 5508.0 | 5.63 | 5.62 | 642.59 | 1216.60 |
| dequant | 2+2 | 186.2 | 5516.4 | 5.62 | 5.61 | 642.59 | 1216.60 |
| fused | haiku | 171.6 | 827.5 | 37.46 | 32.03 | 642.59 | 751.37 |
| fused | 2+2 | 171.8 | 839.2 | 36.94 | 31.65 | 642.59 | 751.37 |

- **Decode is CPU-bound, so the faster kernel does not show up in tok/s.** One
  token takes ~27 ms, but streaming all FP16 weights takes the GPU only ~8.5 ms
  (~2.0 GB at ~240 GB/s) and the packed weights ~2.5-3 ms. The rest is Python
  and Hugging Face launching hundreds of small ops one at a time while the GPU
  waits, so fused and FP16 decode both run at ~37 tok/s. CUDA graphs
  (`torch.compile(mode="reduce-overhead")` with a static KV cache) would remove
  that overhead and expose the kernel difference.
- Fused decode is ~1.7x bitsandbytes NF4 (37.5 vs 22.6 tok/s) and ~6.7x
  dequant every call, with the least memory: **643 MiB** allocated vs **2100
  MiB** FP16 (~3.3x smaller) and 798 MiB NF4 (NF4 keeps `lm_head` in FP16).
- Fused prefill (~172 ms vs ~31 ms FP16) and the dequant row were measured with
  the old Python dequant (several full-size int8/float32 passes per layer).
  Prefill and dequant mode now use the one-pass CUDA dequantize kernel; these
  rows will be re-measured.
- Fused and dequant greedy text match each other; FP16 and NF4 wording differs
  (quantization).

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
  Nibbles become FP16 through an exponent bit trick (one LOP3 per two weights)
  instead of integer-to-float; multiply-adds stay FP32. Each warp computes
  several output rows (`W4A16_ROWS_PER_WARP` = 1, 2, or 4; default 4) and shares
  each activation load and conversion across them. `W4A16_DECODE` selects the
  version for A/B runs: `v1` original shared-memory kernel, `v2` wide loads,
  `v3` wide loads + fast unpack, `v4` (default) v3 with several rows per warp
- `QuantizedLinear` with `M > 8` or `fused=False`: CUDA dequantize kernel (one
  pass to an FP16 `[N, K]` matrix, bit-identical to the Python
  `dequantize_weights`, needs `group_size % 32 == 0`) + cuBLAS. Calling
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
| `dequantize_cuda` | Packed weights + scales → FP16 `[N, K]` on the GPU |
| `cuda_extension_available` | Whether `w4a16._C` loaded |
| `save_quantized_weight_checkpoint` | One packed tensor + metadata |
| `load_quantized_weight_checkpoint` | Load and validate one packed tensor |
