"""Generate TinyLlama text in FP16, bitsandbytes NF4, fused INT4, or dequant mode.

Fused decode uses the M=1 kernel; prefill (M>1) dequantizes a temporary FP16
weight and uses cuBLAS. dequant mode uses cuBLAS on every call, including
decode. Quantized modes load packed INT4 weights onto an empty model so the
original FP16 Linear weights never occupy GPU memory. FP16 mode still loads
the Hugging Face checkpoint as a separate baseline. bnb-nf4 is Hugging Face
load_in_4bit (bitsandbytes NF4, FP16 compute); like transformers' default it
keeps lm_head in FP16, while this repo quantizes lm_head too. This compares generate
output, tok/s, and resident VRAM, not kernel correctness. Kernel correctness
is tests vs linear_reference.

Timing methodology, per mode and prompt:
1. One untimed generate that stops at EOS. Its text is printed, and it warms
   up CUDA context, cuBLAS handles, and the caching allocator.
2. --warmup more untimed generates.
3. --runs timed generates with max_new_tokens=1 (prefill + first token) and
   --runs timed generates forced to exactly --max-new-tokens tokens, so every
   mode decodes the same number of tokens. Medians are reported.
decode_ms = total_ms - prefill_ms. decode tok/s = (new_tokens - 1) / decode_s.

Eager Hugging Face decode is CPU-bound on a T4: each token launches hundreds of
small ops from Python, and the GPU waits between them, so a faster linear
kernel barely changes tok/s. --cuda-graphs passes cache_implementation="static";
transformers then torch.compiles the decode step with mode="reduce-overhead",
which replays each token as one CUDA graph. Prefill stays eager. The first
generate per prompt compiles (slow); the untimed text run and warmup absorb it.
transformers marks bitsandbytes 4-bit models as not compileable and silently
decodes them eagerly, so with --cuda-graphs bnb-nf4 overrides that flag;
bitsandbytes 4-bit ops are torch.compile-traceable since 0.46.
"""

import argparse
import gc
import statistics
import sys
import time
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from w4a16 import cuda_extension_available, load_packed_state_dict

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPTS = [
    "Write a short haiku about GPUs.",
    "What is 2+2?",
]
CHECKPOINT_FORMAT = "w4a16-tinyllama"
CHECKPOINT_VERSION = 1
BYTES_PER_MIB = 1024 * 1024
GENERATE_MODES = ("fp16", "bnb-nf4", "dequant", "fused")
QUANTIZED_MODES = ("dequant", "fused")
DISPATCH_LABELS = {
    "fp16": "resident FP16 cuBLAS",
    "bnb-nf4": "bitsandbytes NF4 matmul_4bit (lm_head FP16)",
    "dequant": "dequant+cuBLAS every call",
    "fused": "M<=8 fused kernels, M>8 dequant+cuBLAS",
}


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        choices=[*GENERATE_MODES, "all"],
        required=True,
        help=(
            "all runs fp16, bnb-nf4 (if bitsandbytes is installed), dequant, "
            "then fused sequentially on this GPU."
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Extra untimed generates per prompt after the text run (default: 1).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Timed generates per prompt for prefill and for total (default: 5).",
    )
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help=(
            "Static KV cache + torch.compile(mode='reduce-overhead') for decode "
            "steps, removing per-op Python launch overhead."
        ),
    )
    return parser.parse_args()


def load_fp16_model(model_id: str, device: str) -> torch.nn.Module:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
    return model.to(device)


def load_bnb_nf4_model(model_id: str, device: str) -> torch.nn.Module:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("bnb-nf4 requires CUDA")
    if not bitsandbytes_available():
        raise RuntimeError("bnb-nf4 requires `pip install bitsandbytes`")
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
    )


def allow_auto_compile(model: torch.nn.Module) -> None:
    """Let generate auto-compile a model whose quantizer says it cannot.

    Swaps this one quantizer instance to a subclass whose is_compileable is
    True; other models and the transformers class are untouched.
    """
    quantizer = model.hf_quantizer
    quantizer.__class__ = type(
        f"Compileable{type(quantizer).__name__}",
        (type(quantizer),),
        {"is_compileable": property(lambda self: True)},
    )


def load_empty_causal_lm(model_id: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        try:
            return AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        except TypeError:
            return AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.float16,
            )


def encode_prompt(tokenizer: AutoTokenizer, prompt: str, device: str) -> dict:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            encoded = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def load_tinyllama_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint has an unsupported format")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint has an unsupported version")
    if not isinstance(checkpoint.get("group_size"), int):
        raise ValueError("checkpoint is missing group_size")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("checkpoint is missing state_dict")
    return checkpoint


def load_packed_model(
    model_id: str,
    checkpoint_path: str,
    device: str,
    fused: bool,
) -> tuple[torch.nn.Module, int]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    if fused and device == "cuda" and not cuda_extension_available():
        raise RuntimeError(
            "The W4A16 CUDA extension is not built. Build it with "
            "`python setup.py build_ext --inplace`."
        )

    checkpoint = load_tinyllama_checkpoint(checkpoint_path)
    model = load_empty_causal_lm(model_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        replacement_count = load_packed_state_dict(
            model,
            checkpoint["state_dict"],
            group_size=checkpoint["group_size"],
            device=device,
            fused=fused,
        )
    for warning in caught:
        if issubclass(warning.category, UserWarning):
            print(warning.message)
    return model, replacement_count


def timed_generate(
    model: torch.nn.Module,
    encoded: dict,
    max_new_tokens: int,
    fixed_length: bool,
    pad_token_id: int,
    device: str,
    cuda_graphs: bool,
) -> tuple[torch.Tensor, float]:
    """Run one greedy generate and return output ids and elapsed milliseconds."""
    kwargs = {}
    if fixed_length:
        # Ignore EOS so every mode decodes exactly max_new_tokens tokens.
        kwargs["min_new_tokens"] = max_new_tokens
    if cuda_graphs:
        # A static cache makes decode shapes fixed, which is what lets
        # transformers auto-compile the decode step into a CUDA graph.
        kwargs["cache_implementation"] = "static"

    if device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    output_ids = generated.sequences if hasattr(generated, "sequences") else generated
    return output_ids, elapsed_ms


def benchmark_prompt(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    warmup: int,
    runs: int,
    device: str,
    cuda_graphs: bool,
) -> dict:
    encoded = encode_prompt(tokenizer, prompt, device)
    prompt_length = encoded["input_ids"].shape[-1]
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    # Untimed text run (stops at EOS); doubles as the first warmup.
    output_ids, _ = timed_generate(
        model, encoded, max_new_tokens, False, pad_token_id, device, cuda_graphs
    )
    text = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    for _ in range(warmup):
        timed_generate(
            model, encoded, max_new_tokens, True, pad_token_id, device, cuda_graphs
        )

    prefill_samples = [
        timed_generate(model, encoded, 1, True, pad_token_id, device, cuda_graphs)[1]
        for _ in range(runs)
    ]
    total_samples: list[float] = []
    new_tokens = 0
    for _ in range(runs):
        output_ids, elapsed_ms = timed_generate(
            model, encoded, max_new_tokens, True, pad_token_id, device, cuda_graphs
        )
        new_tokens = int(output_ids.shape[-1] - prompt_length)
        total_samples.append(elapsed_ms)

    prefill_ms = statistics.median(prefill_samples)
    total_ms = statistics.median(total_samples)
    decode_ms = max(total_ms - prefill_ms, 0.0)
    decode_tok_s = 1000.0 * (new_tokens - 1) / decode_ms if decode_ms > 0.0 else 0.0
    e2e_tok_s = 1000.0 * new_tokens / total_ms if total_ms > 0.0 else 0.0
    return {
        "prompt": prompt,
        "text": text,
        "prompt_tokens": int(prompt_length),
        "new_tokens": new_tokens,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "total_min_ms": min(total_samples),
        "total_max_ms": max(total_samples),
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": e2e_tok_s,
    }


def format_summary(rows: list[dict]) -> str:
    headers = (
        "mode",
        "prompt",
        "prompt_tok",
        "new_tok",
        "prefill_ms",
        "decode_ms",
        "decode_tok/s",
        "e2e_tok/s",
        "alloc_mib",
        "peak_mib",
    )

    def mib(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    table_rows = [
        (
            row["mode"],
            row["prompt"],
            str(row["prompt_tokens"]),
            str(row["new_tokens"]),
            f"{row['prefill_ms']:.1f}",
            f"{row['decode_ms']:.1f}",
            f"{row['decode_tok_s']:.2f}",
            f"{row['e2e_tok_s']:.2f}",
            mib(row["allocated_mib"]),
            mib(row["peak_mib"]),
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


def run_mode(
    mode: str,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    prompts: list[str],
) -> list[dict]:
    if mode in QUANTIZED_MODES:
        if args.checkpoint is None:
            raise ValueError("checkpoint is required for fused and dequant")
        model, replacement_count = load_packed_model(
            args.model,
            args.checkpoint,
            args.device,
            fused=(mode == "fused"),
        )
        weight_load = "packed-only checkpoint"
    elif mode == "bnb-nf4":
        model = load_bnb_nf4_model(args.model, args.device)
        replacement_count = None
        weight_load = "from_pretrained with BitsAndBytesConfig(load_in_4bit)"
        if args.cuda_graphs and not model.hf_quantizer.is_compileable:
            allow_auto_compile(model)
            weight_load += " (is_compileable forced on for --cuda-graphs)"
    else:
        model = load_fp16_model(args.model, args.device)
        replacement_count = None
        weight_load = "resident FP16 from_pretrained"

    model.eval()
    print(f"mode: {mode}")
    print(f"device: {args.device}")
    print(f"weight_load: {weight_load}")
    print(f"linear_dispatch: {DISPATCH_LABELS[mode]}")
    if replacement_count is not None:
        print(f"replacement_count: {replacement_count}")
    allocated_mib = None
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        allocated_mib = torch.cuda.memory_allocated() / BYTES_PER_MIB
        print(f"allocated_mib: {allocated_mib:.2f}")
        torch.cuda.reset_peak_memory_stats()
    print(f"warmup: {args.warmup}  runs: {args.runs}  (medians reported)")
    print(
        "cuda_graphs: "
        + ("requested (static cache, compiled decode)" if args.cuda_graphs else "off")
    )

    rows: list[dict] = []
    for prompt in prompts:
        row = benchmark_prompt(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.warmup,
            args.runs,
            args.device,
            args.cuda_graphs,
        )
        print(f"prompt: {prompt}")
        print(f"output: {row['text']}")
        print(
            f"prompt_tokens: {row['prompt_tokens']}  "
            f"new_tokens: {row['new_tokens']}  "
            f"prefill_ms: {row['prefill_ms']:.1f}  "
            f"decode_ms: {row['decode_ms']:.1f}  "
            f"decode tok/s: {row['decode_tok_s']:.2f}  "
            f"e2e tok/s: {row['e2e_tok_s']:.2f}  "
            f"total_ms min/med/max: {row['total_min_ms']:.1f}/"
            f"{row['total_ms']:.1f}/{row['total_max_ms']:.1f}"
        )
        print()
        row["mode"] = mode
        row["allocated_mib"] = allocated_mib
        row["peak_mib"] = None
        rows.append(row)

    if args.cuda_graphs:
        # transformers stores the compiled forward here once it auto-compiles;
        # if it is missing, decode ran eagerly (e.g. a quantizer that is not
        # compileable) and the timings are not CUDA-graph numbers.
        compiled = hasattr(model, "_compiled_call")
        print(f"decode_compiled: {'yes' if compiled else 'no (ran eager)'}")
        for row in rows:
            row["mode"] = f"{mode}+graphs" if compiled else f"{mode}+static"

    peak_mib = None
    if args.device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / BYTES_PER_MIB
        print(f"peak_mib: {peak_mib:.2f}")
    for row in rows:
        row["peak_mib"] = peak_mib

    del model
    # Reference cycles (e.g. bitsandbytes quant states) keep a model alive past
    # `del`; collect them so --mode all does not count the last mode's weights
    # in the next mode's allocated/peak MiB.
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return rows


def main() -> None:
    args = parse_args()
    # Hugging Face warns on every generate call that both max_new_tokens and the
    # checkpoint's max_length are set; that buries the timing output.
    transformers.logging.set_verbosity_error()
    if args.max_new_tokens < 2:
        raise ValueError("max_new_tokens must be at least 2 to time decode")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.mode in ("fp16", "bnb-nf4") and args.checkpoint is not None:
        raise ValueError("checkpoint is only valid with fused, dequant, or all")
    if args.mode not in ("fp16", "bnb-nf4") and args.checkpoint is None:
        raise ValueError("checkpoint is required for fused, dequant, and all")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    modes = GENERATE_MODES if args.mode == "all" else (args.mode,)
    if args.mode == "all" and not bitsandbytes_available():
        print("Skipping bnb-nf4: `pip install bitsandbytes` to compare against it.")
        modes = tuple(mode for mode in modes if mode != "bnb-nf4")
    rows: list[dict] = []
    for mode in modes:
        rows.extend(run_mode(mode, args, tokenizer, prompts))
        print()

    print("generate summary (sequential baselines, never overlapped)")
    print(format_summary(rows))


if __name__ == "__main__":
    main()
