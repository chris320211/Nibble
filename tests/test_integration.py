import torch

from w4a16.integration import replace_linear_layers
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
