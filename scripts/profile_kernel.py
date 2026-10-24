"""Roofline profile for the M=1 decode kernel (synthetic GEMMs, not generate).

CUDA-event timing works on Colab T4 without Nsight. Use --cuda-profiler and
--profiler torch when capturing ncu/nsys/Chrome traces later.

kernel_us is GPU time per call: --calls-per-sample back-to-back calls replayed
from a CUDA graph (no Python launch gaps), divided by the call count. Each call
reads a different weight copy, so weights come from DRAM, not L2, as they do in
a real decode step where every layer has its own weights. eager_us is one call
timed alone; it adds Python dispatch and launch latency on top of kernel_us.

Unique payload GB/s is not DRAM traffic. Fused payload is packed weights +
scales + activations + output. FP16 payload is N*K*2 + activations + output.
"""

import argparse
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.cuda.profiler as cuda_profiler
import torch.nn.functional as F

import w4a16
from w4a16.quantization import quantize_and_pack_weights
from w4a16.reference import dequantize_weights

GROUP_SIZE = 128
BYTES_PER_GB = 1e9
BYTES_PER_MIB = 1024 * 1024
DEFAULT_L2_BYTES = 4 * BYTES_PER_MIB
MAX_WEIGHT_COPIES = 256
COPY_BENCH_MIB = 256

# Spec DRAM GB/s for sm_75 cards we may rent. Matched by substring of the
# device name; longer names first so "RTX 2080 Ti" wins over "RTX 2080".
SPEC_DRAM_GBS: list[tuple[str, float]] = [
    ("Tesla T4", 320.0),
    ("RTX 2060 SUPER", 448.0),
    ("RTX 2070 SUPER", 448.0),
    ("RTX 2080 SUPER", 496.0),
    ("RTX 2080 Ti", 616.0),
    ("RTX 2080", 448.0),
    ("RTX 2070", 448.0),
    ("RTX 2060", 336.0),
    ("TITAN RTX", 672.0),
    ("Quadro RTX 6000", 672.0),
    ("Quadro RTX 5000", 448.0),
    ("Quadro RTX 4000", 416.0),
]

# TinyLlama-like (K, N) with group_size=128. M defaults to 1 (decode).
SHAPES: list[tuple[str, int, int]] = [
    ("attention-like", 2048, 2048),
    ("mlp up-like", 2048, 5632),
    ("mlp down-like", 5632, 2048),
    ("small sanity", 256, 8),
]

BASELINE_ORDER = ("resident_fp16", "dequant_every_call", "fused")

# GPU state is sampled with nvidia-smi while extra calls keep the GPU busy, so
# the clock reflects load, not idle. T4 drops its SM clock under its 70 W cap or
# when hot; memory clock stays put, so compute-heavy kernels slow down more.
LOAD_SECONDS = 0.5
SAMPLE_DELAY_SECONDS = 0.2
NVIDIA_SMI_FIELDS = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
# Newer drivers renamed clocks_throttle_reasons to clocks_event_reasons.
NVIDIA_SMI_REASON_FIELDS = (
    "clocks_event_reasons.active",
    "clocks_throttle_reasons.active",
)
THROTTLE_REASONS = {
    0x2: "app-clocks",
    0x4: "sw-power",
    0x8: "hw-slowdown",
    0x20: "sw-thermal",
    0x40: "hw-thermal",
    0x80: "power-brake",
}
NVTX_NAMES = {
    "resident_fp16": "w4a16_fp16",
    "dequant_every_call": "w4a16_dequant",
    "fused": "w4a16_fused",
}


@dataclass(frozen=True)
class GpuState:
    sm_mhz: int
    max_sm_mhz: int
    temperature_c: int
    power_w: float
    throttle: str


@dataclass(frozen=True)
class Row:
    baseline: str
    shape: str
    m: int
    k: int
    n: int
    copies: int
    timing: str
    kernel_us: float
    eager_us: float
    payload_bytes: int
    gbs: float
    pct_peak: float | None
    pct_copy: float
    intensity: float
    gpu_state: GpuState | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roofline-profile resident FP16, per-call INT4 dequant + FP16 linear, "
            "and fused W4A16 linear. Operator-level synthetic GEMMs; not generate."
        )
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="Activation rows M for every shape (default: 1, decode).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Un-timed warmup calls per weight copy (default: 10).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed samples per baseline/shape; the median is reported (default: 20).",
    )
    parser.add_argument(
        "--calls-per-sample",
        type=int,
        default=100,
        help="Back-to-back calls per timed sample (default: 100; use 1-5 under ncu).",
    )
    parser.add_argument(
        "--hot-l2",
        action="store_true",
        help=(
            "Reuse one weight copy so small weights stay in L2. Default rotates "
            "copies totalling > 2x L2 so every call reads DRAM like real decode."
        ),
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Launch calls from Python instead of replaying a CUDA graph.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINE_ORDER),
        default=["fused", "resident_fp16"],
        help="Baselines to run, still executed in canonical order and never overlapped.",
    )
    parser.add_argument(
        "--dram-gbs",
        type=float,
        default=None,
        help=(
            "Spec DRAM GB/s for %%peak. Default: looked up from the GPU name "
            "(T4 = 320); %%peak is blank for unknown GPUs."
        ),
    )
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help=(
            "Wrap the timed loop in cudaProfilerStart/Stop. Pair with "
            "ncu/nsys --profile-from-start off."
        ),
    )
    parser.add_argument(
        "--profiler",
        choices=("none", "torch"),
        default="none",
        help="Optional Chrome trace. 'torch' writes profile_results/*.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_results/latest.txt"),
        help="Write the same table to this path (default: profile_results/latest.txt).",
    )
    return parser.parse_args()


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def spec_dram_gbs(device_name: str) -> float | None:
    for name, gbs in SPEC_DRAM_GBS:
        if name.lower() in device_name.lower():
            return gbs
    return None


def l2_cache_bytes() -> int:
    properties = torch.cuda.get_device_properties(0)
    return int(getattr(properties, "L2_cache_size", 0)) or DEFAULT_L2_BYTES


def decode_throttle(mask: int) -> str:
    names = [name for bit, name in THROTTLE_REASONS.items() if mask & bit]
    return "+".join(names) if names else "none"


def query_gpu_state() -> GpuState | None:
    """One nvidia-smi sample of SM clock, temperature, power, throttle reasons."""
    for reason_field in NVIDIA_SMI_REASON_FIELDS:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={NVIDIA_SMI_FIELDS},{reason_field}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            sm, max_sm, temperature, power, reasons = (
                field.strip() for field in result.stdout.strip().split(",")
            )
            return GpuState(
                sm_mhz=int(sm),
                max_sm_mhz=int(max_sm),
                temperature_c=int(temperature),
                # power.draw reads "[N/A]" on some boards; keep the other fields.
                power_w=float(power) if power[:1].isdigit() else math.nan,
                throttle=decode_throttle(int(reasons, 16)),
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return None


def sample_gpu_state_under_load(launch: Callable[[], None], launch_us: float) -> GpuState | None:
    """Keep the GPU busy for LOAD_SECONDS and sample nvidia-smi partway through."""
    samples: list[GpuState | None] = []

    def sample_later() -> None:
        time.sleep(SAMPLE_DELAY_SECONDS)
        samples.append(query_gpu_state())

    sampler = threading.Thread(target=sample_later)
    sampler.start()
    deadline = time.perf_counter() + LOAD_SECONDS
    # Launch in small batches and sync so the queue never runs far past the deadline.
    batch = max(1, int(10_000 / max(launch_us, 1.0)))
    while time.perf_counter() < deadline:
        for _ in range(batch):
            launch()
        torch.cuda.synchronize()
    sampler.join()
    return samples[0] if samples else None


def payload_bytes(baseline: str, m: int, k: int, n: int, group_size: int) -> int:
    activation_bytes = m * k * 2
    output_bytes = m * n * 2
    if baseline == "resident_fp16":
        return n * k * 2 + activation_bytes + output_bytes
    packed_bytes = n * (k // 2)
    scale_bytes = n * (k // group_size) * 2
    unique = packed_bytes + scale_bytes + activation_bytes + output_bytes
    if baseline == "dequant_every_call":
        # Dequant materializes an FP16 matrix that cuBLAS then reads.
        return unique + n * k * 2
    return unique


def stored_weight_bytes(baseline: str, k: int, n: int, group_size: int) -> int:
    if baseline == "resident_fp16":
        return n * k * 2
    return n * (k // 2) + n * (k // group_size) * 2


def weight_copies(weight_bytes: int, l2_bytes: int, hot_l2: bool) -> int:
    """Copies needed so one pass over them evicts L2 at least twice."""
    if hot_l2:
        return 1
    needed = math.ceil(2 * l2_bytes / max(weight_bytes, 1))
    return min(max(needed, 1), MAX_WEIGHT_COPIES)


def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * n * k


def wrap_nvtx(name: str, operation: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
    def launched() -> torch.Tensor:
        torch.cuda.nvtx.range_push(name)
        try:
            return operation()
        finally:
            torch.cuda.nvtx.range_pop()

    return launched


def make_operation(
    baseline: str,
    x: torch.Tensor,
    weight_fp16: torch.Tensor | None,
    packed: torch.Tensor | None,
    scales: torch.Tensor | None,
    group_size: int,
) -> Callable[[], torch.Tensor]:
    if baseline == "resident_fp16":
        assert weight_fp16 is not None

        def resident_fp16() -> torch.Tensor:
            return F.linear(x, weight_fp16)

        inner = resident_fp16
    elif baseline == "dequant_every_call":
        assert packed is not None and scales is not None

        def dequant_every_call() -> torch.Tensor:
            reconstructed = dequantize_weights(packed, scales, group_size)
            return F.linear(x, reconstructed.to(torch.float16))

        inner = dequant_every_call
    elif baseline == "fused":
        assert packed is not None and scales is not None

        def fused() -> torch.Tensor:
            return w4a16.linear_cuda(x, packed, scales, group_size=group_size)

        inner = fused
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    return wrap_nvtx(NVTX_NAMES[baseline], inner)


def make_operations(
    baseline: str,
    x: torch.Tensor,
    k: int,
    n: int,
    copies: int,
    device: torch.device,
) -> list[Callable[[], torch.Tensor]]:
    """One operation per weight copy, all sharing the same activations."""
    operations: list[Callable[[], torch.Tensor]] = []
    for _ in range(copies):
        if baseline == "resident_fp16":
            weight_fp16 = torch.randn((n, k), dtype=torch.float16, device=device)
            operations.append(
                make_operation(baseline, x, weight_fp16, None, None, GROUP_SIZE)
            )
            continue
        source_weight = torch.randn((n, k), dtype=torch.float16, device=device)
        packed, scales = quantize_and_pack_weights(source_weight, GROUP_SIZE)
        del source_weight
        operations.append(
            make_operation(
                baseline,
                x,
                None,
                packed.contiguous(),
                scales.contiguous(),
                GROUP_SIZE,
            )
        )
    return operations


def run_calls(operations: list[Callable[[], torch.Tensor]], calls: int) -> None:
    for index in range(calls):
        operations[index % len(operations)]()


def capture_graph(
    operations: list[Callable[[], torch.Tensor]],
    calls: int,
) -> torch.cuda.CUDAGraph:
    # Warm up on a side stream before capture, as torch.cuda.graph requires.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run_calls(operations, len(operations))
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_calls(operations, calls)
    torch.cuda.synchronize()
    return graph


def time_samples_us(
    launch: Callable[[], None],
    calls: int,
    iters: int,
    use_cuda_profiler: bool,
) -> list[float]:
    """Time `launch` (which issues `calls` calls) and return microseconds per call."""
    if use_cuda_profiler:
        torch.cuda.synchronize()
        cuda_profiler.start()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples_us: list[float] = []
    for _ in range(iters):
        start_event.record()
        launch()
        end_event.record()
        torch.cuda.synchronize()
        samples_us.append(1000.0 * start_event.elapsed_time(end_event) / calls)

    if use_cuda_profiler:
        torch.cuda.synchronize()
        cuda_profiler.stop()

    return samples_us


def measure(
    operations: list[Callable[[], torch.Tensor]],
    warmup: int,
    iters: int,
    calls: int,
    use_graph: bool,
    use_cuda_profiler: bool,
) -> tuple[float, float, str, GpuState | None]:
    """Return (kernel_us, eager_us, timing mode, GPU state under load).

    Warmup and the GPU-state sample are outside the profiler window.
    """
    for _ in range(warmup):
        run_calls(operations, len(operations))
    torch.cuda.synchronize()

    # Old-style single-call latency: includes Python dispatch and launch gaps.
    eager_us = median(
        time_samples_us(lambda: operations[0](), 1, iters, use_cuda_profiler=False)
    )

    timing = "stream"
    launch: Callable[[], None] = lambda: run_calls(operations, calls)
    if use_graph:
        try:
            graph = capture_graph(operations, calls)
        except RuntimeError as error:
            print(f"  CUDA graph capture failed ({error}); timing Python launches.")
        else:
            timing = "graph"
            launch = graph.replay

    launch()
    torch.cuda.synchronize()
    kernel_us = median(time_samples_us(launch, calls, iters, use_cuda_profiler))
    gpu_state = sample_gpu_state_under_load(launch, kernel_us * calls)
    return kernel_us, eager_us, timing, gpu_state


def measure_copy_gbs(device: torch.device, iters: int = 20) -> float:
    """Device-to-device copy GB/s (read + write): what DRAM actually sustains."""
    size = COPY_BENCH_MIB * BYTES_PER_MIB
    source = torch.empty(size, dtype=torch.uint8, device=device)
    destination = torch.empty_like(source)
    for _ in range(3):
        destination.copy_(source)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        destination.copy_(source)
    end_event.record()
    torch.cuda.synchronize()
    seconds = start_event.elapsed_time(end_event) / 1000.0
    del source, destination
    torch.cuda.empty_cache()
    return 2 * size * iters / seconds / BYTES_PER_GB


def capture_torch_trace(
    operation: Callable[[], torch.Tensor],
    iters: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(iters):
            operation()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    print(f"Wrote Chrome trace {path}")


def run_one(
    baseline: str,
    shape_name: str,
    m: int,
    k: int,
    n: int,
    args: argparse.Namespace,
    dram_gbs: float | None,
    copy_gbs: float,
    l2_bytes: int,
    device: torch.device,
) -> Row:
    x = torch.randn((m, k), dtype=torch.float16, device=device)
    copies = weight_copies(
        stored_weight_bytes(baseline, k, n, GROUP_SIZE),
        l2_bytes,
        args.hot_l2,
    )
    operations = make_operations(baseline, x, k, n, copies, device)

    kernel_us, eager_us, timing, gpu_state = measure(
        operations,
        args.warmup,
        args.iters,
        args.calls_per_sample,
        not args.no_graph,
        args.cuda_profiler,
    )
    bytes_moved = payload_bytes(baseline, m, k, n, GROUP_SIZE)
    seconds = kernel_us / 1e6
    gbs = bytes_moved / seconds / BYTES_PER_GB if seconds > 0.0 else 0.0
    pct_peak = 100.0 * gbs / dram_gbs if dram_gbs else None
    pct_copy = 100.0 * gbs / copy_gbs if copy_gbs > 0.0 else 0.0
    flops = matmul_flops(m, k, n)
    intensity = flops / bytes_moved if bytes_moved > 0 else 0.0

    if args.profiler == "torch":
        safe_shape = shape_name.replace(" ", "_")
        trace_path = Path("profile_results") / f"torch_{baseline}_{safe_shape}.json"
        capture_torch_trace(operations[0], args.iters, trace_path)

    del operations, x
    torch.cuda.synchronize()

    return Row(
        baseline=baseline,
        shape=shape_name,
        m=m,
        k=k,
        n=n,
        copies=copies,
        timing=timing,
        kernel_us=kernel_us,
        eager_us=eager_us,
        payload_bytes=bytes_moved,
        gbs=gbs,
        pct_peak=pct_peak,
        pct_copy=pct_copy,
        intensity=intensity,
        gpu_state=gpu_state,
    )


def format_table(rows: list[Row]) -> str:
    headers = (
        "baseline",
        "shape",
        "M",
        "K",
        "N",
        "copies",
        "timing",
        "kernel_us",
        "eager_us",
        "payload_bytes",
        "GB/s",
        "%peak",
        "%copy",
        "AI",
        "sm_mhz",
        "temp_c",
        "throttle",
    )
    table_rows = [
        (
            row.baseline,
            row.shape,
            str(row.m),
            str(row.k),
            str(row.n),
            str(row.copies),
            row.timing,
            f"{row.kernel_us:.2f}",
            f"{row.eager_us:.2f}",
            str(row.payload_bytes),
            f"{row.gbs:.1f}",
            "-" if row.pct_peak is None else f"{row.pct_peak:.1f}",
            f"{row.pct_copy:.1f}",
            f"{row.intensity:.2f}",
            "-" if row.gpu_state is None else str(row.gpu_state.sm_mhz),
            "-" if row.gpu_state is None else str(row.gpu_state.temperature_c),
            "-" if row.gpu_state is None else row.gpu_state.throttle,
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
    if args.m < 1:
        sys.exit("--m must be >= 1")
    if args.warmup < 0:
        sys.exit("--warmup must be >= 0")
    if args.iters < 1:
        sys.exit("--iters must be >= 1")
    if args.calls_per_sample < 1:
        sys.exit("--calls-per-sample must be >= 1")
    if args.dram_gbs is not None and args.dram_gbs <= 0.0:
        sys.exit("--dram-gbs must be > 0")

    if not torch.cuda.is_available():
        sys.exit("CUDA is required for this profile, but no GPU is available.")

    requested = [name for name in BASELINE_ORDER if name in set(args.baselines)]
    lines: list[str] = []

    if "fused" in requested and not w4a16.cuda_extension_available():
        skip = (
            "Skipping fused: W4A16 CUDA extension is not built. "
            "Build it with `python setup.py build_ext --inplace`."
        )
        print(skip)
        lines.append(skip)
        requested = [name for name in requested if name != "fused"]

    if not requested:
        sys.exit("No baselines left to run.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    dram_gbs = args.dram_gbs if args.dram_gbs is not None else spec_dram_gbs(device_name)
    l2_bytes = l2_cache_bytes()
    copy_gbs = measure_copy_gbs(device)
    idle_state = query_gpu_state()
    if idle_state is None:
        gpu_state_line = "GPU state: nvidia-smi unavailable; sm_mhz/temp_c/throttle left blank."
    else:
        gpu_state_line = (
            f"GPU state at start: SM {idle_state.sm_mhz}/{idle_state.max_sm_mhz} MHz  "
            f"{idle_state.temperature_c} C  {idle_state.power_w:.0f} W  "
            f"throttle {idle_state.throttle}"
        )

    header = [
        "W4A16 decode-kernel roofline (synthetic GEMMs, not full generate).",
        f"GPU: {device_name}",
        (
            f"M={args.m}  warmup={args.warmup}  iters={args.iters}  "
            f"calls_per_sample={args.calls_per_sample}  group_size={GROUP_SIZE}"
        ),
        (
            f"spec DRAM: {'unknown' if dram_gbs is None else f'{dram_gbs:g} GB/s'}  "
            f"measured copy: {copy_gbs:.1f} GB/s  "
            f"L2: {l2_bytes / BYTES_PER_MIB:g} MiB  "
            f"L2: {'hot (one weight copy)' if args.hot_l2 else 'cold (rotating copies)'}"
        ),
        gpu_state_line,
        "Baselines run one after another on this GPU and never overlap.",
        "kernel_us = GPU time per call over back-to-back calls (CUDA graph by default).",
        "eager_us = one call timed alone; includes Python dispatch + launch latency.",
        "copies = weight copies rotated so each call reads DRAM, not L2.",
        "payload_bytes is unique traffic, not measured DRAM.",
        "fused payload = packed + scales + activations + output.",
        "fp16 payload = N*K*2 + activations + output.",
        "GB/s = payload_bytes / kernel time.  %peak vs spec DRAM, %copy vs measured copy.",
        "AI = FLOPs / payload_bytes, FLOPs = 2*M*N*K.",
        (
            "sm_mhz/temp_c/throttle: nvidia-smi sampled under load after each row. "
            "Low sm_mhz or sw-power/thermal throttle means that row ran slowed down."
        ),
    ]
    for line in header:
        print(line)
        lines.append(line)

    rows: list[Row] = []
    with torch.inference_mode():
        for baseline in requested:
            print(f"\nProfiling {baseline} ...")
            for shape_name, k, n in SHAPES:
                row = run_one(
                    baseline,
                    shape_name,
                    args.m,
                    k,
                    n,
                    args,
                    dram_gbs,
                    copy_gbs,
                    l2_bytes,
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
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
