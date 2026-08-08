import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pipeline.compare_phase2_tokenizers import compare


def _tokenizer(path: Path, vocabulary: dict[str, int]) -> Path:
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))
    return path


def test_comparison_is_text_free_hash_bound_and_counts_tokens(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"review_text": text}) + "\n"
            for text in ("alpha beta", "alpha gamma delta")
        ),
        encoding="utf-8",
    )
    first = _tokenizer(
        tmp_path / "first.json",
        {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4},
    )
    second = _tokenizer(
        tmp_path / "second.json",
        {"[UNK]": 0, "alpha": 1, "beta": 2},
    )
    report = compare(
        tokenizers=[("first", first), ("second", second)],
        sources=[("sample", source)],
        maximum_documents=10,
    )
    assert report["contains_document_text"] is False
    assert report["tokenizers"]["first"]["aggregate"]["tokens"] == 5
    assert report["tokenizers"]["second"]["aggregate"]["tokens"] == 5
    assert report["sources"]["sample"]["sampled_documents"] == 2
    assert "alpha beta" not in json.dumps(report)
