import torch

from w4a16.quantization import pack_int4, unpack_int4

# Test four vals within int4 range in int8 tensor
def test_pack_int4_known_vals() -> None:
    quantized_weights = torch.tensor(
        [[2, -3, 7, 1]],
        dtype=torch.int8,
    )
    result = pack_int4(quantized_weights)
    expected_result = torch.tensor(
        [[0xD2, 0x17]],
        dtype=torch.uint8,
    )
    
    assert torch.equal(result, expected_result)

# Test int8 tensor byte unpacking into two signed int4 weights
def test_unpack_int4_known_vals() -> None:
    packed_weights = torch.tensor(
        [[0xD2, 0x17]],
        dtype=torch.uint8,
    )
    result = unpack_int4(packed_weights)
    expected_result = torch.tensor(
        [[2, -3, 7, 1]],
        dtype=torch.int8,
    )
    
    assert torch.equal(result, expected_result)

