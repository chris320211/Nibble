# Fused W4A16 CUDA Linear Kernel Roadmap

## Objective

Build a PyTorch-integrated CUDA operator for LLM inference that computes:

```text
FP16 activations x packed INT4 weights -> FP16 output
```

The kernel will unpack, dequantize, and multiply weights locally without writing a
complete FP16 weight matrix to GPU global memory.

This is a CUDA and ML-systems project. The initial quantizer is intentionally
simple; the primary work is the fused execution path, profiling, and real-model
integration.

## Final decisions

| Decision | Answer |
| --- | --- |
| AWS instance | `g4dn.xlarge` |
| GPU | One NVIDIA T4, 16 GB |
| CUDA architecture | Turing, compute capability 7.5 (`sm_75`) |
| Development pricing | Spot when interruption is acceptable |
| Final benchmark pricing | On-demand for consistent measurements |
| Primary workload | Single-token decoding, `M = 1` |
| Secondary workloads | `M = 4, 8, 16, 32` |
| Model | One pinned 1B-class Llama-family checkpoint |
| Quantization | Symmetric group-wise W4A16 |
| Group size | Fixed at 128 for v1 |
| Supported v1 shape | `K` must be divisible by 128 |
| Fast path | Aligned transformer dimensions |
| Accumulation | FP32, with final FP16 output |
| Primary performance goal | Beat per-call dequantization and target resident FP16 cuBLAS at `M = 1` |
| Required optimized implementation | Fast `M = 1` decode kernel |
| Stretch implementation | Small-batch FP16 Tensor Core kernel |

The instance price is a planning estimate and must be checked before launch.
Pin the AWS region used for final measurements.

Compile the CUDA extension for the T4 with:

```text
-gencode arch=compute_75,code=sm_75
```

## Workload

### Primary: M = 1

Single-token decoding performs:

```text
activation[1, K] x weight[N, K]^T -> output[1, N]
```

It reads many weights for relatively little arithmetic. This makes it primarily
weight-bandwidth-bound and is the workload where packed INT4 weights should offer
the clearest advantage.

### Secondary: M = 4-32

Small-batch decoding reuses each weight tile for several activation rows. Compute
becomes more important as `M` grows. A separate Tensor Core kernel may improve
these shapes, but it is a stretch goal rather than a requirement for minimum
project success.

## CUDA cores and Tensor Cores

CUDA cores are general-purpose arithmetic lanes. Ordinary CUDA threads use them
for scalar and vector operations such as bit manipulation, address calculation,
FP32 accumulation, and control flow. The fast `M = 1` kernel will primarily use
CUDA cores because a matrix tile with only one activation row does not naturally
fill a Tensor Core operation.

Tensor Cores are specialized matrix-multiply-accumulate units. A warp collectively
submits fixed-size matrix tiles to them, allowing the GPU to perform many
multiplications and additions per instruction. They provide high throughput when
the problem has enough rows and columns to keep those tiles occupied.

The CUDA-core and Tensor Core paths are two alternative fused implementations of
the same linear operation. They are not two sequential kernels. Runtime dispatch
selects one implementation for a given input shape, and that one kernel performs
all weight loading, local dequantization, multiplication, accumulation, and output
writing.

### CUDA-core data path

Offline quantization stores an integer code `q` and a scale `s` such that an
original weight is approximated by:

```text
w_approx = s * q
```

Unpacking and dequantization are different operations:

- Unpacking extracts two four-bit patterns from one byte and decodes each pattern
  as a signed integer code in `[-7, 7]`.
- Converting the integer code to FP32 or FP16 changes its machine representation,
  but does not yet reconstruct the approximate weight. For example, integer `-3`
  converted to FP32 is still numerically `-3.0`.
- Dequantization applies the saved group scale. If the scale is `0.125`, the
  approximate weight for code `-3` is `0.125 * -3 = -0.375`.

The direct element-by-element formulation is:

```text
w_approx[k] = float(scale[g]) * float(q[k])
output_sum += float(x[k]) * w_approx[k]
```

This is the easiest formulation for the naive correctness kernel. The temporary
`w_approx` is only a register value for the current element; no FP16 weight tensor
is created.

Because all 128 codes in a group share the same scale, the optimized kernel can
use the algebraically equivalent formulation:

```text
group_dot = sum(float(x[k]) * float(q[k])) for k in group
output_sum += float(scale[g]) * group_dot
```

In equation form:

```text
sum(x[k] * (s[g] * q[k])) = s[g] * sum(x[k] * q[k])
```

The scale is still applied and the quantized weight is still dequantized in the
mathematical computation. The optimization merely applies the shared scale once
to the group's partial dot product instead of multiplying the same scale into all
128 individual weights. The two formulations can differ slightly in floating
point rounding, so correctness tests use a documented tolerance.

For the optimized `M = 1` kernel, a warp can own one output channel. Its lanes load
adjacent packed bytes from one weight row, decode their assigned INT4 values,
multiply them by the corresponding activations, and accumulate FP32 lane-local
partial sums. For each group, the group scale is loaded once and broadcast or
otherwise reused by the warp. The warp reduces its lane-local results and writes
one final FP16 output value. Several warps in a block may process several output
channels and optionally share a staged activation tile.

No complete dequantized weight row or matrix exists at any point:

```text
packed bytes in global memory
    -> byte values in registers
    -> signed q values in registers
    -> FP32 partial dot products in registers
    -> group scale applied to partial dot products
    -> final FP16 output in global memory
```

### Optional Tensor Core data path

The T4 supports both FP16 Tensor Core MMA and integer INT4 Tensor Core MMA, but it
does not provide a direct mixed `FP16 x INT4 -> FP32` W4A16 instruction. The T4's
advertised INT4 throughput applies to low-bit integer operands on both sides, not
to FP16 activations with INT4 weights.

Therefore, the proposed W4A16 Tensor Core path is:

```text
packed INT4 weight tile in global memory
    -> CUDA instructions load and unpack nibbles
    -> CUDA instructions convert codes and apply FP16 scales
    -> dequantized FP16 tile exists only in registers/shared memory
    -> FP16 Tensor Core MMA with FP32 accumulation
    -> final FP16 output written to global memory
```

The full FP16 weight matrix is never reconstructed in global memory. The main
engineering question is whether weight unpacking, conversion, shared-memory
traffic, and Tensor Core utilization together beat the FP16 baseline.

This path is one self-contained CUDA kernel. CUDA instructions inside that kernel
perform addressing, unpacking, conversion, scale application, and tile movement;
the same kernel then issues warp-level Tensor Core MMA instructions. It does not
launch a dequantization kernel followed by a matrix-multiplication kernel.

Conceptual runtime dispatch is:

```text
M = 1                  -> launch the CUDA-core decode kernel
small M, if beneficial -> launch the fused Tensor Core kernel
otherwise              -> launch the CUDA-core fallback
```

## Quantized format

Weights are stored logically as `W[N, K]`, matching `torch.nn.Linear`.

- Groups run along `K` independently for every output row.
- `group_size` is fixed at 128 in v1.
- `K % 128 == 0` is required in v1.
- Quantized codes are signed values in `[-7, 7]`.
- Scales are stored as FP16.
- Even `k` uses the low nibble.
- Odd `k` uses the high nibble.
- Negative values use four-bit two's-complement encoding.

Stored tensors:

```text
packed_weights: uint8 [N, K / 2]
scales:         float16 [N, K / 128]
```

Quantization for each group:

```text
scale = max(abs(weights)) / 7
q = clamp(round(weights / scale), -7, 7)
```

For an all-zero group:

```text
scale = 1.0
q = 0
```

Including FP16 scales, the weight storage is approximately:

```text
2 / (0.5 + 2 / 128) = 3.88x smaller than FP16
```

## Operator contract

The initial public operator will be conceptually equivalent to:

```python
output = torch.ops.int4_cuda.linear(
    activations,
    packed_weights,
    scales,
    group_size,
)
```

The v1 contract is:

- Inference only; no backward implementation.
- CUDA tensors only.
- Contiguous FP16 activations.
- Contiguous packed `uint8` weights.
- Contiguous FP16 scales.
- `group_size == 128`.
- `K % 128 == 0`.
- FP32 accumulation and FP16 output.
- Arbitrary leading activation dimensions may be flattened to `[M, K]`.
- Bias is stored separately by `QuantizedLinear` and applied after the operator if
  the selected model contains biased linear layers.
- Unsupported shapes or dtypes fail with clear validation errors.

## Phase 0: Environment and baseline validation

Before kernel development:

- Select and pin the exact model checkpoint and revision.
- Extract all real `(K, N)` linear-layer shapes from the model.
- Pin the AWS region, AMI, NVIDIA driver, CUDA toolkit, compiler, Python, and
  PyTorch versions.
- Verify account quota for `g4dn.xlarge`.
- Compile and execute a trivial `sm_75` PyTorch CUDA extension.
- Verify that Nsight Compute performance counters are accessible.
- Run initial PyTorch FP16 latency and memory measurements.
- Record the reproducible environment in the repository.

Phase 0 is complete when the toolchain, profiler, model, and baseline measurements
all work on the target T4 instance.

## Phase 1: Format, checkpoint, and correctness

Build:

- Python FP16-to-INT4 group-wise quantizer.
- Python packer and unpacker.
- Independent byte-format tests using known nibble patterns.
- Slow PyTorch reference reconstructed from the same codes and scales.
- Naive fused CUDA kernel.
- PyTorch custom-operator wrapper with input validation.
- Initial `QuantizedLinear` module.

### Versioned checkpoint schema

The quantized checkpoint must store:

- A format name and schema version.
- Packed weights.
- FP16 scales.
- Original `N` and `K`.
- Group size.
- Packing order and signed-code convention.
- Original tensor names and model/config metadata needed for replacement.
- Bias tensors when present.
- Source model identifier and revision.

Add tests for:

- Save/load round trips.
- Exact packed-byte preservation.
- CPU loading followed by transfer to CUDA.
- Registration of weights and scales as non-trainable module buffers.
- Replacement of compatible `torch.nn.Linear` modules.
- Rejection of incompatible layers and malformed metadata.

The naive kernel will:

1. Load a packed weight byte.
2. Extract two signed INT4 codes.
3. Load the scale for the current group.
4. Multiply by FP16 activations.
5. Accumulate in FP32.
6. Cast the final output to FP16.

Correctness tests cover:

- Known byte patterns.
- Zero groups.
- Extreme codes.
- Multiple activation rows.
- Representative aligned transformer shapes.
- Operator validation failures.
- Checkpoint conversion and reload.

Phase 1 is complete when the fused CUDA output matches the PyTorch result produced
from the same quantized codes and scales within a documented tolerance. Kernel
correctness is not measured directly against the original FP16 weights because
that would mix implementation error with quantization error.

## Phase 2: Fast M = 1 decoding kernel

Optimize specifically for:

- `M = 1`.
- `group_size = 128`.
- Aligned transformer `K` and `N` dimensions.

Investigate:

- Coalesced packed-weight reads.
- Vectorized global-memory loads.
- Warp and block ownership of output rows.
- Multiple output values per block.
- Register-based nibble unpacking.
- Reuse of each group scale.
- Carefully selected block sizes.
- Minimal branching and synchronization.
- Kernel-specific offline weight layouts when justified by measurements.

Profile with Nsight Compute and inspect:

- DRAM throughput.
- Global-load efficiency.
- Register usage and spills.
- Achieved occupancy.
- Warp-stall reasons.
- Instruction cost of unpacking and conversion.
- Scale-load overhead.

Phase 2 performance targets:

1. Beat packed INT4 weights dequantized into temporary FP16 weights on every call,
   followed by cuBLAS.
2. Approach or beat resident FP16 cuBLAS on a majority of the representative
   `M = 1` model shapes.
3. Preserve approximately 3.88x weight-memory compression.
4. Explain achieved performance using profiler evidence.

Phase 2 is the most important optimization milestone.

## Phase 3: Model integration and end-to-end validation

Replace compatible `torch.nn.Linear` layers with versioned-checkpoint-backed
`QuantizedLinear` modules.

Validate:

- Loading the quantized checkpoint without the original FP16 weights on GPU.
- Correct model forward execution.
- Deterministic layer replacement.
- Generation for representative prompts.
- Kernel dispatch for the model's actual shapes.
- Persistent and peak GPU memory.
- Per-layer and end-to-end generation latency.
- Model-quality change from the original FP16 checkpoint.

Model integration happens before the optional Tensor Core kernel so that the
primary `M = 1` kernel's real value is established early.

## Phase 4: Small-batch Tensor Core kernel (stretch)

If the required milestones are complete, create a separate kernel for:

```text
M = 4, 8, 16, 32
```

The kernel will:

- Load and reuse FP16 activation tiles.
- Load packed INT4 weight tiles.
- Use CUDA instructions to unpack and locally dequantize weights.
- Store temporary FP16 weight tiles only in registers or shared memory.
- Arrange activation and weight fragments for Turing FP16 Tensor Core operations.
- Execute FP16 MMA with FP32 accumulation.
- Overlap loading, dequantization, and computation where practical.

Investigate:

- Shared-memory tiling.
- Double buffering.
- Warp-level fragment ownership.
- Direct `mma.sync` or a suitable WMMA/CUTLASS-based prototype.
- Several tile configurations.
- Runtime dispatch based on `M`.

Conceptual dispatch:

```python
if M == 1:
    use_fast_decode_kernel()
elif M <= 32 and tensor_core_kernel_is_supported:
    use_small_batch_tensor_core_kernel()
else:
    use_fallback()
```

The stretch phase is successful if it demonstrates where Tensor Core execution
improves on the SIMT implementation and honestly identifies where INT4 weight
compression stops providing a latency advantage.

## Phase 5: Final evaluation, packaging, and documentation

Benchmark these baselines:

1. Resident FP16 weights with PyTorch/cuBLAS.
2. Packed INT4 weights dequantized into temporary FP16 weights on every invocation,
   followed by cuBLAS.
3. Packed weights dequantized once and retained as resident FP16 weights.
4. Naive fused W4A16 kernel.
5. Optimized fused W4A16 kernel.
6. Small-batch Tensor Core kernel, if completed.

The once-dequantized baseline may be faster while consuming much more memory.
Report latency and memory together rather than treating either as the sole result.

### Benchmark protocol

- Use CUDA events for kernel and operator latency.
- Warm up before collecting measurements.
- Use a fixed iteration count and report at least median and p95 latency.
- Synchronize consistently across all baselines.
- Distinguish allocation-inclusive and allocation-exclusive timings.
- Run final measurements on an on-demand instance.
- Report every real `(M, K, N)` model shape rather than only favorable shapes.
- Record the exact software environment and compiler flags.

Report:

- Kernel latency by `(M, K, N)`.
- Tokens per second and time per generated token.
- Persistent and peak GPU memory.
- Weight-checkpoint storage.
- Effective memory bandwidth.
- Implementation error versus the quantized reference.
- Quantization error and model-quality change versus original FP16.
- Nsight evidence explaining bottlenecks and speedups.

Package the result with:

- Reproducible build and installation instructions.
- Quantization, checkpoint-loading, and generation examples.
- A documented operator and module API.
- A supported-hardware and supported-shape table.
- Clear diagnostics for unsupported inputs.
- Scripts that reproduce correctness and benchmark results.
- An architecture explanation and profiling report.

## Success levels

### Minimum success

- Correct PyTorch custom-operator integration.
- Well-tested, versioned packed checkpoint format.
- Approximately 3.88x weight-storage reduction.
- Faster execution than per-call dequantization plus cuBLAS.
- Successful real-model generation and honest quality measurements.

### Target success

- Competitive or improved `M = 1` latency versus resident FP16 cuBLAS on most
  representative model shapes.
- End-to-end token-generation improvement.
- Nsight evidence that explains the result.

### Stretch success

- A separate Tensor Core path for `M = 4-32`.
- Runtime dispatch between decode, small-batch, and fallback kernels.
- A clear analysis of when the Tensor Core path wins and when it does not.

Beating a mature Marlin-class kernel is not required. A performant, reproducible,
well-explained T4-specific implementation with real model integration is already a
substantial CUDA and ML-systems project.
