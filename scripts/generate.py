"""Generate TinyLlama text in FP16, fused INT4, or dequant-every-call mode.

Fused decode uses the M=1 kernel; prefill (M>1) dequantizes a temporary FP16
weight and uses cuBLAS. dequant mode uses cuBLAS on every call, including
decode. Quantized modes load packed INT4 weights onto an empty model so the
original FP16 Linear weights never occupy GPU memory. FP16 mode still loads
the Hugging Face checkpoint as a separate baseline. This compares generate
QUALITY, tok/s, and resident VRAM, not kernel correctness. Kernel correctness
is tests vs linear_reference.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
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
GENERATE_MODES = ("fp16", "dequant", "fused")
QUANTIZED_MODES = ("dequant", "fused")
DISPATCH_LABELS = {
    "fp16": "resident FP16 cuBLAS",
    "dequant": "dequant+cuBLAS every call",
    "fused": "M=1 fused kernel, M>1 dequant+cuBLAS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        choices=[*GENERATE_MODES, "all"],
        required=True,
        help="all runs fp16, dequant, then fused sequentially on this GPU.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", action="append", dest="prompts")
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


def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[str, int, float]:
    encoded = encode_prompt(tokenizer, prompt, device)
    prompt_length = encoded["input_ids"].shape[-1]
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

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
        )

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    output_ids = generated.sequences if hasattr(generated, "sequences") else generated
    new_tokens = int(output_ids.shape[-1] - prompt_length)
    decoded = tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    )
    return decoded, new_tokens, elapsed_ms


def format_summary(rows: list[dict]) -> str:
    headers = ("mode", "prompt", "new_tokens", "ms", "tok/s", "peak_mib")
    table_rows = [
        (
            row["mode"],
            row["prompt"],
            str(row["new_tokens"]),
            f"{row['ms']:.1f}",
            f"{row['tok_s']:.2f}",
            f"{row['peak_mib']:.2f}" if row["peak_mib"] is not None else "n/a",
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
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        allocated_mib = torch.cuda.memory_allocated() / BYTES_PER_MIB
        print(f"allocated_mib: {allocated_mib:.2f}")
        torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    for prompt in prompts:
        decoded, new_tokens, elapsed_ms = generate_text(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.device,
        )
        tokens_per_s = (
            1000.0 * new_tokens / elapsed_ms if elapsed_ms > 0.0 else 0.0
        )
        print(f"prompt: {prompt}")
        print(f"output: {decoded}")
        print(
            f"new_tokens: {new_tokens}  "
            f"ms: {elapsed_ms:.1f}  "
            f"tok/s: {tokens_per_s:.2f}"
        )
        print()
        rows.append(
            {
                "mode": mode,
                "prompt": prompt,
                "new_tokens": new_tokens,
                "ms": elapsed_ms,
                "tok_s": tokens_per_s,
                "peak_mib": None,
            }
        )

    peak_mib = None
    if args.device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / BYTES_PER_MIB
        print(f"peak_mib: {peak_mib:.2f}")
    for row in rows:
        row["peak_mib"] = peak_mib

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return rows


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if args.mode == "fp16" and args.checkpoint is not None:
        raise ValueError("checkpoint is only valid with fused, dequant, or all")
    if args.mode != "fp16" and args.checkpoint is None:
        raise ValueError("checkpoint is required for fused, dequant, and all")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    modes = GENERATE_MODES if args.mode == "all" else (args.mode,)
    rows: list[dict] = []
    for mode in modes:
        rows.extend(run_mode(mode, args, tokenizer, prompts))
        print()

    if len(modes) > 1:
        print("generate summary (sequential baselines, never overlapped)")
        print(format_summary(rows))


if __name__ == "__main__":
    main()
