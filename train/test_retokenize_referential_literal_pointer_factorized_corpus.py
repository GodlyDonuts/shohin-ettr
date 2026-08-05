from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from retokenize_referential_literal_pointer_factorized_corpus import (
    retokenize_row,
    semantic_payload,
)


def test_retokenization_preserves_semantic_row() -> None:
    tokenizer = Tokenizer(WordLevel(
        vocab={"[UNK]": 0, "Move": 1, "left": 2, "now": 3},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = Whitespace()
    row = {
        "id": "row-1",
        "question": "Move left now",
        "program": [{"kind": "left"}],
        "spans": {
            "op0.kind": {
                "start": 5,
                "end": 9,
                "text": "left",
                "token_positions": [9],
                "token_ids": [999],
            }
        },
        "token_count": 99,
        "token_ids_sha256": "old",
        "token_bag": [[999, 1]],
    }
    derived = retokenize_row(row, tokenizer)
    assert semantic_payload(derived) == semantic_payload(row)
    assert derived["spans"]["op0.kind"]["token_positions"] == [1]
    assert derived["spans"]["op0.kind"]["token_ids"] == [2]
    assert derived["token_count"] == 3
