import torch

from w4a16.checkpoint import (
    load_quantized_weight_checkpoint,
    save_quantized_weight_checkpoint,
)
from w4a16.quantization import quantize_and_pack_weights


def test_quantized_weight_checkpoint_round_trip(tmp_path) -> None:
    weights = torch.tensor(
        [[0.25, -0.375, 0.875, -0.125]],
        dtype=torch.float16,
    )
    packed_weights, scales = quantize_and_pack_weights(weights, group_size=4)
    checkpoint_path = tmp_path / "quantized_weights.pt"

    save_quantized_weight_checkpoint(
        checkpoint_path,
        packed_weights,
        scales,
        group_size=4,
    )
    loaded_weights, loaded_scales, loaded_group_size = (
        load_quantized_weight_checkpoint(checkpoint_path)
    )

    assert torch.equal(loaded_weights, packed_weights)
    assert torch.equal(loaded_scales, scales)
    assert loaded_group_size == 4
