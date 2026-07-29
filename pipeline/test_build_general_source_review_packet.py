import gzip
import hashlib
import json

from pipeline.build_general_source_review_packet import (
    materialize_review_rows,
    select_review_rows,
)
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_SCHEMA,
    exact_text_hash,
    stable_document_identity,
)


def ledger_row(index: int, *, license_value: str, tokens: int, domain: str):
    text = f"Document {index} explains a useful technical concept."
    document_sha256 = exact_text_hash(text).hex()
    source = {"id": f"paper-{index}", "text": text}
    return source, {
        "schema": DOCUMENT_LEDGER_SCHEMA,
        "source_row_index": index,
        "stable_identity_sha256": stable_document_identity(
            source,
            document_sha256,
        ),
        "document_sha256": document_sha256,
        "domain": domain,
        "allowed_value": license_value,
        "chars": len(text),
        "tokens": tokens,
        "shard": "shard_00000.u16.zst",
        "token_start": index * tokens,
        "token_end": (index + 1) * tokens,
        "token_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
    }


def test_review_selection_is_deterministic_and_covers_strata():
    pairs = [
        ledger_row(
            index,
            license_value=("CCBY" if index % 2 else "pd"),
            tokens=(1_000 if index % 3 else 10_000),
            domain=f"domain-{index % 7}.example",
        )
        for index in range(100)
    ]
    rows = [row for _source, row in pairs]
    first = select_review_rows(rows, count=20)
    second = select_review_rows(reversed(rows), count=20)
    assert first == second
    assert {row["allowed_value"] for row in first} == {"CCBY", "pd"}
    assert len({row["stable_identity_sha256"] for row in first}) == 20
    assert len({row["domain"] for row in first}) >= 5


def test_materialized_review_rechecks_exact_source_identity(tmp_path):
    pairs = [
        ledger_row(
            index,
            license_value="CCBY",
            tokens=1_000,
            domain="example.org",
        )
        for index in range(4)
    ]
    source_path = tmp_path / "source.json.gz"
    with gzip.open(source_path, "wt") as output:
        for source, _ledger in pairs:
            output.write(json.dumps(source) + "\n")
    with gzip.open(source_path, "rt") as source:
        source_rows = [json.loads(line) for line in source]
    selected = [pairs[1][1], pairs[3][1]]
    reviews = materialize_review_rows(
        source_rows,
        selected,
        dataset="local/test",
        config="default",
        filters={"text_col": "text", "text_cols": None},
        max_review_chars=1_000,
    )
    assert len(reviews) == 2
    assert all(row["review_text"] for row in reviews)
    assert {
        row["stable_identity_sha256"] for row in reviews
    } == {
        row["stable_identity_sha256"] for row in selected
    }
