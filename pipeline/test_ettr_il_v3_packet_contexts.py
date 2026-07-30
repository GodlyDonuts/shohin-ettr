from __future__ import annotations

from pathlib import Path
import sys

import pytest
from tokenizers import Tokenizer


_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _ROOT / "train"
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from ettr_il_v2_token_native_surface import DEFAULT_TOKENIZER_PATH  # noqa: E402
from ettr_il_v3_materialize import (  # noqa: E402
    materialize_candidate,
    rematerialize_record,
)
from ettr_il_v3_packet_contexts import (  # noqa: E402
    compact_packet_context_rows,
)
from ettr_packet_index import compact_packet_batch  # noqa: E402
from test_ettr_il_v3_materialize import _row  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH))


@pytest.mark.parametrize("family", ("horn", "resource", "local_rewrite"))
def test_fast_packet_contexts_match_full_tensor_projection(
    family: str,
    tokenizer: Tokenizer,
) -> None:
    record = materialize_candidate(_row(family), tokenizer)
    full = compact_packet_batch(rematerialize_record(record, tokenizer))
    fast = compact_packet_context_rows(record, tokenizer)
    assert len(fast) == 64
    assert fast == full.rows
