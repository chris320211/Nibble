"""Generate TinyLlama text in FP16 or fused INT4 mode.

This compares QUALITY (quantization error vs FP16), not kernel correctness.
Kernel correctness is tests vs linear_reference.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from w4a16 import cuda_extension_available, replace_linear_layers

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPTS = [
    "Write a short haiku about GPUs.",
    "What is 2+2?",
]
CHECKPOINT_FORMAT = "w4a16-tinyllama"
CHECKPOINT_VERSION = 1
BYTES_PER_MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["fp16", "fused"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", action="append", dest="prompts")
    return parser.parse_args()


def load_model(model_id: str, device: str) -> torch.nn.Module:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    return model.to(device)


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


def apply_fused_weights(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    device: str,
) -> int:
    if device == "cuda" and not cuda_extension_available():
        raise RuntimeError(
            "The W4A16 CUDA extension is not built. Build it with "
            "`python setup.py build_ext --inplace`."
        )

    group_size = 128
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = load_tinyllama_checkpoint(checkpoint_path)
        group_size = checkpoint["group_size"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        replacement_count = replace_linear_layers(model, group_size=group_size)
    for warning in caught:
        if issubclass(warning.category, UserWarning):
            print(warning.message)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
    return replacement_count


def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[str, int, float]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[-1]

    if device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    new_tokens = int(output_ids.shape[-1] - prompt_length)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded, new_tokens, elapsed_ms


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if args.mode == "fp16" and args.checkpoint is not None:
        raise ValueError("checkpoint is only valid with fused mode")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model, args.device)
    model.eval()
    print(f"mode: {args.mode}")
    print(f"device: {args.device}")
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    if args.mode == "fused":
        replacement_count = apply_fused_weights(
            model,
            args.checkpoint,
            args.device,
        )
        print(f"replacement_count: {replacement_count}")

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
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
        print(decoded)
        print(
            f"new_tokens: {new_tokens}  "
            f"ms: {elapsed_ms:.1f}  "
            f"tok/s: {tokens_per_s:.2f}"
        )
        print()

    if args.device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / BYTES_PER_MIB
        print(f"peak_mib: {peak_mib:.2f}")


if __name__ == "__main__":
    main()
