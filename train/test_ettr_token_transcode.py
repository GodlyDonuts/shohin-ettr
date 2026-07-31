import pytest
import torch

from ettr_episode import ETTREpisodeSegment
from ettr_token_transcode import (
    ETTRTokenTranscodeError,
    transcode_segment,
)


def test_segment_transcode_preserves_width_mask_and_causal_targets() -> None:
    segment = ETTREpisodeSegment.from_tokens(
        torch.tensor([[1, 2, 3, 0], [3, 2, 0, 0]]),
        attention_mask=torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
    )
    mapping = torch.tensor([9, 7, 8, 6])
    result = transcode_segment(segment, mapping)
    assert result.tokens.tolist() == [[7, 8, 6, 9], [6, 8, 9, 9]]
    assert torch.equal(result.attention_mask, segment.attention_mask)
    assert result.targets.tolist() == [[8, 6, -1, -1], [8, -1, -1, -1]]


def test_segment_transcode_rejects_unmapped_visible_token() -> None:
    segment = ETTREpisodeSegment.from_tokens(torch.tensor([[1, 2, 3]]))
    with pytest.raises(ETTRTokenTranscodeError, match="unmapped"):
        transcode_segment(segment, torch.tensor([0, 10, -1, 12]))
