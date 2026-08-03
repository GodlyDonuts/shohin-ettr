import pytest
import torch

from capability_floor_byte_rail import (
    ASCII_CARDINALITY,
    BYTE_ROLE_WIDTH,
    CapabilityFloorByteRailError,
    byte_role_features,
)
from ettr_il_v2_token_native_surface import CODEWORD_BYTES


def test_byte_role_features_preserve_byte_and_within_atom_position() -> None:
    source = tuple(b"ABCDEFGH12345678")
    masks = (
        tuple(index < CODEWORD_BYTES for index in range(len(source))),
        tuple(index >= CODEWORD_BYTES for index in range(len(source))),
        tuple(False for _ in source),
        tuple(False for _ in source),
    )
    features, present = byte_role_features(source, masks)
    assert features.shape == (4, BYTE_ROLE_WIDTH)
    assert present.tolist() == [True, True, False, False]
    assert int(features[0].sum()) == CODEWORD_BYTES
    assert int(features[1].sum()) == CODEWORD_BYTES
    for position, byte in enumerate(b"ABCDEFGH"):
        assert features[0, position * ASCII_CARDINALITY + byte] == 1
    assert not torch.equal(features[0], features[1])


def test_byte_role_features_reject_partial_public_atom() -> None:
    source = tuple(b"ABCDEFGH")
    masks = (
        (True, False, False, False, False, False, False, False),
        tuple(False for _ in source),
        tuple(False for _ in source),
        tuple(False for _ in source),
    )
    with pytest.raises(CapabilityFloorByteRailError, match="one public atom"):
        byte_role_features(source, masks)
