import pytest
import torch

from w4a16.integration import load_packed_state_dict, replace_linear_layers
from w4a16.linear import QuantizedLinear


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direct = torch.nn.Linear(4, 2, bias=True, dtype=torch.float16)
        self.nested = torch.nn.Sequential(
            torch.nn.Linear(4, 2, bias=False, dtype=torch.float16)
        )

        weights = torch.tensor(
            [
                [0.25, -0.375, 0.875, -0.125],
                [0.25, 0.5, -0.25, 1.75],
            ],
            dtype=torch.float16,
        )
        with torch.no_grad():
            self.direct.weight.copy_(weights)
            self.direct.bias.copy_(
                torch.tensor([0.5, -0.25], dtype=torch.float16)
            )
            self.nested[0].weight.copy_(weights)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.direct(activations) + self.nested(activations)


def test_replace_linear_layers() -> None:
    model = ToyModel()
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 2.0, 0.5],
        ],
        dtype=torch.float16,
    )
    expected = model(activations)

    replacement_count = replace_linear_layers(model, group_size=4)
    result = model(activations)

    assert replacement_count == 2
    assert isinstance(model.direct, QuantizedLinear)
    assert isinstance(model.nested[0], QuantizedLinear)
    assert torch.equal(result, expected)


class MixedAlignmentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aligned = torch.nn.Linear(4, 2, bias=True, dtype=torch.float16)
        self.nested = torch.nn.Sequential(
            torch.nn.Linear(6, 2, bias=False, dtype=torch.float16)
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.aligned(activations)


def test_replace_linear_layers_warns_on_unaligned_in_features() -> None:
    model = MixedAlignmentModel()

    with pytest.warns(UserWarning, match=r"nested\.0"):
        replacement_count = replace_linear_layers(model, group_size=4)

    assert replacement_count == 1
    assert isinstance(model.aligned, QuantizedLinear)
    assert isinstance(model.nested[0], torch.nn.Linear)


def test_replace_linear_layers_fused_flag() -> None:
    model = ToyModel()

    replacement_count = replace_linear_layers(model, group_size=4, fused=False)

    assert replacement_count == 2
    assert model.direct.fused is False
    assert model.nested[0].fused is False


class ShapeOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direct = torch.nn.Linear(4, 2, bias=True, dtype=torch.float16)
        self.nested = torch.nn.Sequential(
            torch.nn.Linear(4, 2, bias=False, dtype=torch.float16)
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.direct(activations) + self.nested(activations)


def _toy_activations() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 2.0, 0.5],
        ],
        dtype=torch.float16,
    )


def test_replace_linear_layers_packed_only_roundtrip() -> None:
    source = ToyModel()
    activations = _toy_activations()
    replace_linear_layers(source, group_size=4)
    expected = source(activations)
    state = {key: tensor.clone() for key, tensor in source.state_dict().items()}

    target = ToyModel()
    with torch.no_grad():
        target.direct.weight.fill_(3.0)
        target.nested[0].weight.fill_(3.0)
        target.direct.bias.fill_(3.0)

    replacement_count = replace_linear_layers(
        target,
        group_size=4,
        packed_only=True,
    )
    target.load_state_dict(state)

    assert replacement_count == 2
    assert torch.equal(target(activations), expected)


def test_load_packed_state_dict_from_meta() -> None:
    source = ToyModel()
    activations = _toy_activations()
    replace_linear_layers(source, group_size=4, fused=False)
    expected = source(activations)
    state = {key: tensor.clone() for key, tensor in source.state_dict().items()}

    with torch.device("meta"):
        target = ShapeOnlyModel()
    replacement_count = load_packed_state_dict(
        target,
        state,
        group_size=4,
        device="cpu",
        fused=False,
    )

    assert replacement_count == 2
    assert target.direct.packed_weights.device.type == "cpu"
    assert "weight" not in target.direct.state_dict()
    assert target.direct.fused is False
    assert torch.equal(target(activations), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_load_packed_state_dict_does_not_materialize_fp16_weights() -> None:
    in_features = 2048
    out_features = 2048
    group_size = 128
    source = torch.nn.Linear(
        in_features,
        out_features,
        bias=False,
        dtype=torch.float16,
    )
    quantized = QuantizedLinear.from_float(source, group_size=group_size)
    state = {
        "layer.packed_weights": quantized.packed_weights.detach().cpu().clone(),
        "layer.scales": quantized.scales.detach().cpu().clone(),
    }

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(
                in_features,
                out_features,
                bias=False,
                dtype=torch.float16,
            )

    with torch.device("meta"):
        model = Wrapper()

    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    replacement_count = load_packed_state_dict(
        model,
        state,
        group_size=group_size,
        device="cuda",
    )
    delta = torch.cuda.memory_allocated() - before
    fp16_bytes = in_features * out_features * 2
    packed_bytes = (
        quantized.packed_weights.numel() + quantized.scales.numel() * 2
    )

    assert replacement_count == 1
    assert model.layer.packed_weights.device.type == "cuda"
    assert delta >= packed_bytes
    assert delta < fp16_bytes


def test_load_packed_state_dict_tiny_llama_roundtrip() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    try:
        source = transformers.AutoModelForCausalLM.from_config(
            config,
            dtype=torch.float16,
        )
    except TypeError:
        source = transformers.AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
        )
    source.eval()
    replace_linear_layers(source, group_size=64, fused=False)
    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in source.state_dict().items()
    }
    input_ids = torch.randint(0, 32, (1, 4))

    with torch.device("meta"):
        try:
            target = transformers.AutoModelForCausalLM.from_config(
                config,
                dtype=torch.float16,
            )
        except TypeError:
            target = transformers.AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.float16,
            )
    replacement_count = load_packed_state_dict(
        target,
        state,
        group_size=64,
        device="cpu",
        fused=False,
    )
    target.eval()

    with torch.inference_mode():
        expected = source(input_ids=input_ids).logits
        result = target(input_ids=input_ids).logits

    assert replacement_count > 0
    assert not any(
        isinstance(module, torch.nn.Linear)
        for module in target.modules()
        if module is not target
    )
    assert torch.equal(
        source.model.rotary_emb.inv_freq,
        target.model.rotary_emb.inv_freq,
    )
    assert torch.equal(result, expected)
