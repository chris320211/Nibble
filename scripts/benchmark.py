"""Operator-level W4A16 linear benchmark (synthetic GEMMs, not generate).

Published numbers should be collected on an NVIDIA T4 (sm_75). This script
does not embed timing results; run it on a T4 to collect numbers.

Baselines run one after another on the same GPU and never overlap:
resident_fp16, dequant_every_call, bnb_nf4, then fused. bnb_nf4 is the
bitsandbytes NF4 4-bit path that Hugging Face uses for load_in_4bit
(blocksize 64, FP32 absmax); it is skipped if bitsandbytes is not installed.
"""

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

import w4a16
from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import dequantize_weights

GROUP_SIZE = 128
BYTES_PER_MIB = 1024 * 1024

# TinyLlama-like (K, N) with group_size=128. M defaults to 1 (decode).
SHAPES: list[tuple[str, int, int]] = [
    ("attention-like", 2048, 2048),
    ("mlp up-like", 2048, 5632),
    ("small sanity", 256, 8),
]

BASELINE_ORDER = ("resident_fp16", "dequant_every_call", "bnb_nf4", "fused")
BNB_BLOCKSIZE = 64


def import_bitsandbytes():
    try:
        import bitsandbytes
    except ImportError:
        return None
    return bitsandbytes


@dataclass(frozen=True)
class Row:
    baseline: str
    shape: str
    m: int
    k: int
    n: int
    median_ms: float
    p95_ms: float
    tokens_per_s: float | None
    peak_mib: float
    fp16_bytes: int
    weight_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time resident FP16, per-call INT4 dequant + FP16 linear, and fused "
            "W4A16 linear. Operator-level synthetic GEMMs; not end-to-end generate."
        )
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Activation rows M to sweep for every shape (default: 1, decode). "
            "The fused path uses dedicated kernels for M <= 8."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Un-timed warmup iterations per baseline/shape (default: 10).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per baseline/shape (default: 50).",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINE_ORDER),
        default=list(BASELINE_ORDER),
        help="Baselines to run, still executed in canonical order and never overlapped.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/latest.txt"),
        help="Write the same table to this path (default: benchmark_results/latest.txt).",
    )
    return parser.parse_args()


def fp16_weight_bytes(n: int, k: int) -> int:
    return n * k * 2


def median_ms(samples_ms: list[float]) -> float:
    ordered = sorted(samples_ms)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def p95_ms(samples_ms: list[float]) -> float:
    ordered = sorted(samples_ms)
    rank = math.ceil(0.95 * len(ordered)) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return ordered[rank]


def measure(
    operation: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[list[float], int]:
    """Warm up, then time with CUDA events. Synchronize after every iteration."""
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = operation()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples_ms: list[float] = []
    for _ in range(iters):
        start_event.record()
        output = operation()
        end_event.record()
        torch.cuda.synchronize()
        samples_ms.append(start_event.elapsed_time(end_event))

    peak_bytes = torch.cuda.max_memory_allocated()
    del output
    return samples_ms, peak_bytes


def make_operation(
    baseline: str,
    x: torch.Tensor,
    source_weight: torch.Tensor,
    group_size: int,
) -> tuple[Callable[[], torch.Tensor], int]:
    """Build the timed operation and return it with its stored weight bytes."""
    if baseline == "resident_fp16":
        weight_fp16 = source_weight

        def resident_fp16() -> torch.Tensor:
            return F.linear(x, weight_fp16)

        return resident_fp16, weight_fp16.numel() * weight_fp16.element_size()

    if baseline == "bnb_nf4":
        bnb = import_bitsandbytes()
        assert bnb is not None
        qweight, quant_state = bnb.functional.quantize_4bit(
            source_weight,
            blocksize=BNB_BLOCKSIZE,
            quant_type="nf4",
        )
        stored_bytes = (
            qweight.numel() * qweight.element_size()
            + quant_state.absmax.numel() * quant_state.absmax.element_size()
        )

        def bnb_nf4() -> torch.Tensor:
            return bnb.matmul_4bit(x, qweight.t(), quant_state=quant_state)

        return bnb_nf4, stored_bytes

    packed, scales = quantize_and_pack_weights(source_weight, group_size)
    packed = packed.contiguous()
    scales = scales.contiguous()
    stored_bytes = packed.numel() + scales.numel() * 2

    if baseline == "dequant_every_call":

        def dequant_every_call() -> torch.Tensor:
            # dequantize_weights returns float32; cast to FP16 to match W4A16.
            reconstructed = dequantize_weights(packed, scales, group_size)
            return F.linear(x, reconstructed.to(torch.float16))

        return dequant_every_call, stored_bytes

    if baseline == "fused":

        def fused() -> torch.Tensor:
            return w4a16.linear_cuda(x, packed, scales, group_size=group_size)

        return fused, stored_bytes

    raise ValueError(f"unknown baseline: {baseline}")


def run_one(
    baseline: str,
    shape_name: str,
    m: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Row:
    x = torch.randn((m, k), dtype=torch.float16, device=device)
    source_weight = torch.randn((n, k), dtype=torch.float16, device=device)
    operation, stored_bytes = make_operation(baseline, x, source_weight, GROUP_SIZE)
    if baseline != "resident_fp16":
        # Quantized baselines must not keep the FP16 source resident.
        del source_weight
        source_weight = None
        torch.cuda.empty_cache()

    samples_ms, peak_bytes = measure(operation, warmup, iters)
    med = median_ms(samples_ms)
    tokens = 1000.0 * m / med if med > 0.0 else None

    del operation, x, source_weight
    torch.cuda.synchronize()

    return Row(
        baseline=baseline,
        shape=shape_name,
        m=m,
        k=k,
        n=n,
        median_ms=med,
        p95_ms=p95_ms(samples_ms),
        tokens_per_s=tokens,
        peak_mib=peak_bytes / BYTES_PER_MIB,
        fp16_bytes=fp16_weight_bytes(n, k),
        weight_bytes=stored_bytes,
    )


def format_tokens(tokens_per_s: float | None) -> str:
    if tokens_per_s is None:
        return "n/a"
    return f"{tokens_per_s:.2f}"


def format_table(rows: list[Row]) -> str:
    headers = (
        "baseline",
        "shape",
        "M",
        "K",
        "N",
        "median_ms",
        "p95_ms",
        "tok/s",
        "peak_MiB",
        "fp16_bytes",
        "weight_bytes",
    )
    table_rows = [
        (
            row.baseline,
            row.shape,
            str(row.m),
            str(row.k),
            str(row.n),
            f"{row.median_ms:.4f}",
            f"{row.p95_ms:.4f}",
            format_tokens(row.tokens_per_s),
            f"{row.peak_mib:.2f}",
            str(row.fp16_bytes),
            str(row.weight_bytes),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), max((len(row[index]) for row in table_rows), default=0))
        for index in range(len(headers))
    ]
    header_line = "  ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    )
    rule = "  ".join("-" * widths[index] for index in range(len(headers)))
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in table_rows
    ]
    return "\n".join([header_line, rule, *body])


def main() -> None:
    args = parse_args()
    if any(m < 1 for m in args.m):
        sys.exit("--m must be >= 1")
    if args.warmup < 0:
        sys.exit("--warmup must be >= 0")
    if args.iters < 1:
        sys.exit("--iters must be >= 1")

    if not torch.cuda.is_available():
        sys.exit("CUDA is required for this benchmark, but no GPU is available.")

    requested = [name for name in BASELINE_ORDER if name in set(args.baselines)]
    lines: list[str] = []

    if "fused" in requested and not w4a16.cuda_extension_available():
        skip = (
            "Skipping fused: W4A16 CUDA extension is not built. "
            "Build it with `python setup.py build_ext --inplace`. "
            "resident_fp16 and dequant_every_call still run."
        )
        print(skip)
        lines.append(skip)
        requested = [name for name in requested if name != "fused"]

    if "bnb_nf4" in requested and import_bitsandbytes() is None:
        skip = "Skipping bnb_nf4: `pip install bitsandbytes` to compare against it."
        print(skip)
        lines.append(skip)
        requested = [name for name in requested if name != "bnb_nf4"]

    if not requested:
        sys.exit("No baselines left to run.")

    torch.manual_seed(0)
    device = torch.device("cuda")

    header = [
        "W4A16 operator-level linear benchmark (synthetic GEMMs, not full generate).",
        "Collect published numbers on an NVIDIA T4 (sm_75).",
        f"GPU: {torch.cuda.get_device_name(0)}",
        (
            f"M={','.join(str(m) for m in args.m)}  warmup={args.warmup}  "
            f"iters={args.iters}  group_size={GROUP_SIZE}"
        ),
        "Baselines run one after another on this GPU and never overlap.",
        "tok/s = 1000 * M / median_ms; operator-level, not full generate.",
        "fp16_bytes = N*K*2. weight_bytes = what the baseline keeps resident:",
        "  fused/dequant: packed codes + FP16 scales (group 128, ~4.1 bits/weight)",
        "  bnb_nf4: packed codes + FP32 absmax (block 64, ~4.5 bits/weight)",
        "Peak MiB is torch.cuda.max_memory_allocated() after the timed loop.",
    ]
    for line in header:
        print(line)
        lines.append(line)

    rows: list[Row] = []
    with torch.inference_mode():
        for baseline in requested:
            print(f"\nRunning {baseline} ...")
            for m in args.m:
                for shape_name, k, n in SHAPES:
                    row = run_one(
                        baseline,
                        shape_name,
                        m,
                        k,
                        n,
                        args.warmup,
                        args.iters,
                        device,
                    )
                    rows.append(row)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    table = format_table(rows)
    print()
    print(table)
    lines.append("")
    lines.append(table)
    lines.append("")

    output_text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text)


if __name__ == "__main__":
    main()
