"""Quantize TinyLlama linear layers to packed W4A16 and save a model checkpoint."""

import argparse
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM

from w4a16 import replace_linear_layers

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT = "checkpoints/tinyllama-w4a16.pt"
CHECKPOINT_FORMAT = "w4a16-tinyllama"
CHECKPOINT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def load_model(model_id: str, device: str) -> torch.nn.Module:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    return model.to(device)


def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    model = load_model(args.model, args.device)
    model.eval()

    with torch.inference_mode():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            replacement_count = replace_linear_layers(
                model,
                group_size=args.group_size,
            )

    print(f"replacement_count: {replacement_count}")
    for warning in caught:
        if issubclass(warning.category, UserWarning):
            print(warning.message)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "source_model": args.model,
            "group_size": args.group_size,
            "state_dict": {
                key: tensor.detach().cpu()
                for key, tensor in model.state_dict().items()
            },
        },
        output_path,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
