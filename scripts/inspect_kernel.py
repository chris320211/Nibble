"""Static view of the compiled W4A16 kernels: registers, occupancy, SASS mix.

Reads the built extension with cuobjdump, so it needs the CUDA toolkit but no
GPU performance counters (works on Colab where ncu fails). Opcode counts are
static (instructions in the binary), not executed counts: an instruction inside
the K loop runs once per iteration. Read the dumped SASS to see the loop body.

Occupancy assumes sm_75 limits and the launch's threads per block (256).
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

KERNEL_PREFIX = "int4_linear_kernel"
CUDA_BIN = Path("/usr/local/cuda/bin")

# sm_75 (Turing) per-SM limits.
SM75_MAX_WARPS = 32
SM75_MAX_BLOCKS = 16
SM75_REGISTERS = 65536
SM75_REGISTER_UNIT_PER_WARP = 256
SM75_SHARED_BYTES = 65536
SM75_SHARED_UNIT = 256

# Opcode families worth watching in a memory-bound GEMV.
TRACKED = (
    ("LDG", "global loads"),
    ("LDS", "shared loads"),
    ("STS", "shared stores"),
    ("BAR", "barriers"),
    ("SHFL", "warp shuffles"),
    ("I2F", "int->float"),
    ("HADD2.F32", "half->float"),
    ("FFMA", "fp32 fma"),
    ("HFMA2", "half2 fma"),
    ("LOP3", "bit ops"),
    ("PRMT", "byte permute"),
)

INSTRUCTION = re.compile(
    r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)"
)
RESOURCE = re.compile(
    r"Function\s+(\S+?):\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
SASS_FUNCTION = re.compile(r"Function\s*:\s*(\S+)")


@dataclass(frozen=True)
class Resources:
    registers: int
    stack: int
    shared: int
    local: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Path to the built extension (default: find w4a16/_C*.so).",
    )
    parser.add_argument(
        "--threads-per-block",
        type=int,
        default=256,
        help="Threads per block used by the launches (default: 256).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only show kernels whose short name contains this text (e.g. decode).",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("profile_results/sass"),
        help="Write each kernel's SASS here (default: profile_results/sass).",
    )
    return parser.parse_args()


def find_tool(name: str) -> str:
    found = shutil.which(name) or (
        str(CUDA_BIN / name) if (CUDA_BIN / name).is_file() else None
    )
    if found is None:
        sys.exit(f"{name} not found; install the CUDA toolkit or add it to PATH.")
    return found


def find_library(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    matches = sorted((_REPO_ROOT / "w4a16").glob("_C*.so"))
    if not matches:
        sys.exit("No w4a16/_C*.so. Build it with `python setup.py build_ext --inplace`.")
    return matches[0]


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def demangle(names: list[str]) -> dict[str, str]:
    tool = shutil.which("cu++filt") or shutil.which("c++filt")
    if tool is None and (CUDA_BIN / "cu++filt").is_file():
        tool = str(CUDA_BIN / "cu++filt")
    if tool is None:
        return {name: name for name in names}
    result = subprocess.run(
        [tool], input="\n".join(names), capture_output=True, text=True, check=True
    )
    return dict(zip(names, result.stdout.splitlines()))


def short_name(demangled: str, mangled: str) -> str:
    # Require "(" so the source file name in the anonymous namespace is skipped.
    match = re.search(rf"({KERNEL_PREFIX}\w*)(<[^>]*>)?\(", demangled)
    if match:
        return match.group(1) + (match.group(2) or "")
    # No demangler: Itanium names are length-prefixed, e.g. 25int4_linear_kernel_decode.
    # The digits before the name may run into a namespace hash, so try each suffix
    # and keep the length whose name is followed by E (end of name) or I (template).
    for match in re.finditer(rf"(\d+)(?={KERNEL_PREFIX})", mangled):
        digits, start = match.group(1), match.end()
        for index in range(len(digits)):
            end = start + int(digits[index:])
            if end < len(mangled) and mangled[end] in "EI":
                identifier = mangled[start:end]
                template = re.match(r"IL[ib](\d+)EE", mangled[end:])
                return identifier + (f"<{template.group(1)}>" if template else "")
    return demangled


def split_sass(sass: str) -> dict[str, list[str]]:
    functions: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in sass.splitlines():
        header = SASS_FUNCTION.search(line)
        if header:
            current = functions.setdefault(header.group(1), [])
            continue
        if current is not None:
            current.append(line)
    return functions


def opcode_counts(lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        match = INSTRUCTION.search(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def family_count(counts: Counter[str], family: str) -> int:
    return sum(
        count
        for opcode, count in counts.items()
        if opcode == family or opcode.startswith(family + ".")
    )


def global_load_widths(counts: Counter[str]) -> str:
    widths = {
        opcode: count for opcode, count in counts.items() if opcode.startswith("LDG")
    }
    return ", ".join(f"{opcode} x{count}" for opcode, count in sorted(widths.items()))


def occupancy(resources: Resources, threads_per_block: int) -> tuple[int, float, str]:
    """Return (blocks per SM, occupancy fraction, limiting resource)."""
    warps_per_block = math.ceil(threads_per_block / 32)
    registers_per_warp = (
        math.ceil(resources.registers * 32 / SM75_REGISTER_UNIT_PER_WARP)
        * SM75_REGISTER_UNIT_PER_WARP
    )
    limits = {
        "warps": SM75_MAX_WARPS // warps_per_block,
        "blocks": SM75_MAX_BLOCKS,
        "registers": (SM75_REGISTERS // max(registers_per_warp, 1)) // warps_per_block,
    }
    if resources.shared > 0:
        shared_per_block = math.ceil(resources.shared / SM75_SHARED_UNIT) * SM75_SHARED_UNIT
        limits["shared"] = SM75_SHARED_BYTES // shared_per_block
    limiter = min(limits, key=limits.get)
    blocks = limits[limiter]
    return blocks, blocks * warps_per_block / SM75_MAX_WARPS, limiter


def main() -> None:
    args = parse_args()
    cuobjdump = find_tool("cuobjdump")
    library = find_library(args.library)
    print(f"Library: {library}")

    resources: dict[str, Resources] = {}
    for match in RESOURCE.finditer(run([cuobjdump, "-res-usage", str(library)])):
        name, registers, stack, shared, local = match.groups()
        if KERNEL_PREFIX in name:
            resources[name] = Resources(int(registers), int(stack), int(shared), int(local))

    functions = {
        name: lines
        for name, lines in split_sass(run([cuobjdump, "-sass", str(library)])).items()
        if KERNEL_PREFIX in name
    }
    if not functions:
        sys.exit("No int4_linear_kernel* functions in the SASS dump.")

    names = demangle(sorted(set(functions) | set(resources)))
    args.dump_dir.mkdir(parents=True, exist_ok=True)

    for mangled in sorted(functions, key=lambda name: short_name(names[name], name)):
        kernel = short_name(names[mangled], mangled)
        if args.filter not in kernel:
            continue
        counts = opcode_counts(functions[mangled])
        dump_path = args.dump_dir / f"{re.sub(r'[^A-Za-z0-9_]', '_', kernel)}.sass"
        dump_path.write_text("\n".join(functions[mangled]) + "\n")

        print(f"\n== {kernel}")
        resource = resources.get(mangled)
        if resource is not None:
            blocks, fraction, limiter = occupancy(resource, args.threads_per_block)
            print(
                f"  registers {resource.registers}  shared {resource.shared} B  "
                f"stack {resource.stack} B  local (spills) {resource.local} B"
            )
            print(
                f"  occupancy at {args.threads_per_block} threads/block: "
                f"{blocks} blocks/SM, {100 * fraction:.0f}% warps (limited by {limiter})"
            )
        print(f"  instructions {sum(counts.values())}  (static)")
        for family, label in TRACKED:
            count = family_count(counts, family)
            if count:
                print(f"  {family:<10} {count:>5}  {label}")
        widths = global_load_widths(counts)
        if widths:
            print(f"  global load widths: {widths}")
        print(f"  SASS: {dump_path}")


if __name__ == "__main__":
    main()
