#!/usr/bin/env python3
"""Regression checks for deterministic general-source quality filters."""

import json
import io

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from pipeline.tokenize_shards import (
    boilerplate_marker_count,
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    DocumentLedgerWriter,
    file_receipt,
    domain_value,
    exact_text_hash,
    extraction_quality,
    field_value,
    max_line_repeat_fraction,
    local_input_format,
    parse_required_allowed_values,
    parse_required_minimums,
    parse_required_values,
    required_allowed_values_match,
    required_minimums_match,
    required_values_match,
    resolve_local_inputs,
    stable_document_identity,
    verify_file_receipt,
)


def test_exact_hash_is_stable_and_text_sensitive():
    assert exact_text_hash("alpha") == exact_text_hash("alpha")
    assert exact_text_hash("alpha") != exact_text_hash("Alpha")


def test_stable_document_identity_prefers_source_identity_and_falls_back_to_text():
    document_hash = exact_text_hash("alpha").hex()
    assert stable_document_identity({"id": "paper-1"}, document_hash) == (
        stable_document_identity({"id": "paper-1"}, document_hash)
    )
    assert stable_document_identity({"id": "paper-1"}, document_hash) != (
        stable_document_identity({"id": "paper-2"}, document_hash)
    )
    assert stable_document_identity({}, document_hash) == stable_document_identity(
        {},
        document_hash,
    )


def test_document_ledger_writer_is_text_free_and_hash_bound(tmp_path):
    path = tmp_path / DOCUMENT_LEDGER_NAME
    writer = DocumentLedgerWriter(path)
    writer.write(
        {
            "schema": DOCUMENT_LEDGER_SCHEMA,
            "source_row_index": 3,
            "stable_identity_sha256": "a" * 64,
            "document_sha256": "b" * 64,
            "domain": "example.org",
            "allowed_value": "CCBY",
            "chars": 120,
            "tokens": 4,
            "shard": "shard_00000.u16.zst",
            "token_start": 0,
            "token_end": 4,
            "token_sha256": "c" * 64,
        }
    )
    receipt = writer.close()
    assert receipt["rows"] == 1
    assert receipt["tokens"] == 4
    assert receipt["contains_document_text"] is False
    assert receipt["sha256"] == file_receipt(path)["sha256"]
    with zstd.ZstdDecompressor().stream_reader(
        io.BytesIO(path.read_bytes())
    ) as reader:
        rows = [
            json.loads(line)
            for line in reader.read().decode("ascii").splitlines()
        ]
    assert rows[0]["stable_identity_sha256"] == "a" * 64
    assert "text" not in rows[0]


def test_line_repetition_uses_normalized_nonempty_lines():
    assert max_line_repeat_fraction("same\nsame\nother\n") == 2 / 3
    assert max_line_repeat_fraction("\n\n") == 1.0


def test_boilerplate_markers_are_bounded():
    assert boilerplate_marker_count("Accept cookies. Privacy policy.") == 2
    assert boilerplate_marker_count("A useful educational explanation.") == 0


def test_extraction_quality_counts_only_nonwhitespace_controls():
    clean = extraction_quality("Alpha beta\nGamma")
    assert clean["control_fraction"] == 0
    assert clean["replacement_fraction"] == 0
    assert clean["alpha_fraction"] > 0.8

    damaged = extraction_quality("Alpha\ufffd\u0000")
    assert damaged["replacement_fraction"] == 1 / 7
    assert damaged["control_fraction"] == 1 / 7


def test_field_value_reads_nested_license_metadata():
    row = {"metadata": {"oa_license": "CCBY"}}
    assert field_value(row, "metadata.oa_license") == "CCBY"
    assert field_value(row, "metadata.missing") is None


def test_required_values_are_exact_conjunctive_and_case_insensitive():
    predicates = parse_required_values(
        [
            "full_doc_lid=eng_Latn",
            "is_truncated=false",
            "extractor=rolmOCR",
        ]
    )
    assert required_values_match(
        {
            "full_doc_lid": "eng_latn",
            "is_truncated": False,
            "extractor": "ROLmOCR",
        },
        predicates,
    )
    assert not required_values_match(
        {
            "full_doc_lid": "eng_Latn",
            "is_truncated": True,
            "extractor": "rolmOCR",
        },
        predicates,
    )
    assert not required_values_match(
        {
            "full_doc_lid": "eng_Latn",
            "is_truncated": False,
        },
        predicates,
    )


@pytest.mark.parametrize(
    "specifications",
    [
        ["missing-delimiter"],
        ["=value"],
        ["field="],
        ["field=one", "field=two"],
    ],
)
def test_required_values_reject_ambiguous_contracts(specifications):
    with pytest.raises(ValueError, match="required-value"):
        parse_required_values(specifications)


def test_required_allowed_values_are_grouped_or_then_conjunctive():
    predicates = parse_required_allowed_values(
        [
            "eai_taxonomy.reasoning_depth.primary.code=3",
            "eai_taxonomy.reasoning_depth.primary.code=4",
            "eai_taxonomy.reasoning_depth.primary.code=5",
            "eai_taxonomy.technical_correctness.primary.code=4",
            "eai_taxonomy.technical_correctness.primary.code=5",
        ]
    )
    assert required_allowed_values_match(
        {
            "eai_taxonomy": {
                "reasoning_depth": {"primary": {"code": "4"}},
                "technical_correctness": {"primary": {"code": 5}},
            },
        },
        predicates,
    )
    assert not required_allowed_values_match(
        {
            "eai_taxonomy": {
                "reasoning_depth": {"primary": {"code": "6"}},
                "technical_correctness": {"primary": {"code": "5"}},
            },
        },
        predicates,
    )
    assert not required_allowed_values_match(
        {
            "eai_taxonomy": {
                "reasoning_depth": {"primary": {"code": "4"}},
                "technical_correctness": {"primary": {"code": "6"}},
            },
        },
        predicates,
    )


@pytest.mark.parametrize(
    "specifications",
    [
        ["missing-delimiter"],
        ["=1"],
        ["field="],
        ["field=one", "field=ONE"],
    ],
)
def test_required_allowed_values_reject_ambiguous_contracts(specifications):
    with pytest.raises(ValueError, match="required-allowed-value"):
        parse_required_allowed_values(specifications)


def test_required_minimums_are_numeric_dotted_and_conjunctive():
    predicates = parse_required_minimums(
        [
            "quality_signals.fasttext.english=0.9",
            "eai_taxonomy.reasoning_depth.primary.code=3",
            "eai_taxonomy.education_level.primary.code=2",
        ]
    )
    assert required_minimums_match(
        {
            "quality_signals": {"fasttext": {"english": 0.95}},
            "eai_taxonomy": {
                "reasoning_depth": {"primary": {"code": 4}},
                "education_level": {"primary": {"code": "2"}},
            },
        },
        predicates,
    )
    assert not required_minimums_match(
        {
            "quality_signals": {"fasttext": {"english": 0.89}},
            "eai_taxonomy": {
                "reasoning_depth": {"primary": {"code": 4}},
                "education_level": {"primary": {"code": 3}},
            },
        },
        predicates,
    )
    assert not required_minimums_match(
        {
            "quality_signals": {"fasttext": {"english": 0.95}},
            "eai_taxonomy": {
                "reasoning_depth": {"primary": {"code": 4}},
            },
        },
        predicates,
    )


@pytest.mark.parametrize(
    "specifications",
    [
        ["missing-delimiter"],
        ["=1"],
        ["field="],
        ["field=not-a-number"],
        ["field=nan"],
        ["field=inf"],
        ["field=1", "field=2"],
    ],
)
def test_required_minimums_reject_ambiguous_contracts(specifications):
    with pytest.raises(ValueError, match="required-min-number"):
        parse_required_minimums(specifications)


def test_required_minimums_reject_nonfinite_row_values():
    predicates = {"score": 0.5}
    assert not required_minimums_match({"score": "nan"}, predicates)
    assert not required_minimums_match({"score": "inf"}, predicates)


def test_domain_value_handles_urls_and_direct_fields():
    assert domain_value({"url": "https://Docs.Example.org/path"}, "url") == (
        "docs.example.org"
    )
    assert domain_value({"source": "Wikipedia"}, "source") == "wikipedia"
    assert domain_value({}, "url") == "<missing>"


def test_file_receipt_detects_midrun_input_mutation(tmp_path):
    path = tmp_path / "frozen-input"
    path.write_text("first")
    receipt = file_receipt(path)
    verify_file_receipt(receipt)
    path.write_text("second")
    try:
        verify_file_receipt(receipt)
    except RuntimeError as exc:
        assert "changed during tokenization" in str(exc)
    else:
        raise AssertionError("mutated input should fail its receipt")


def test_file_receipt_rejects_symlink_and_hardlink(tmp_path):
    source = tmp_path / "source"
    source.write_text("frozen")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(source)
    with pytest.raises(RuntimeError, match="non-symlink"):
        file_receipt(symlink)

    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(source)
    with pytest.raises(RuntimeError, match="single-link"):
        file_receipt(source)


def test_local_inputs_require_revision_and_are_sorted(tmp_path):
    second = tmp_path / "second.json.gz"
    first = tmp_path / "first.json.gz"
    second.write_bytes(b"second")
    first.write_bytes(b"first")

    with pytest.raises(ValueError, match="explicit --revision"):
        resolve_local_inputs([second], None)

    paths, receipts = resolve_local_inputs([second, first], "pinned-revision")
    assert paths == [str(first.resolve()), str(second.resolve())]
    assert [receipt["path"] for receipt in receipts] == paths
    assert all(len(receipt["sha256"]) == 64 for receipt in receipts)

    with pytest.raises(ValueError, match="duplicate paths"):
        resolve_local_inputs([first, first], "pinned-revision")


def test_local_input_format_supports_pinned_json_and_parquet():
    assert local_input_format(["a.json.gz", "b.jsonl"]) == "json"
    assert local_input_format(["a.parquet", "b.parquet"]) == "parquet"
    assert local_input_format(None) is None
    with pytest.raises(ValueError, match="homogeneous"):
        local_input_format(["a.parquet", "b.json.gz"])
    with pytest.raises(ValueError, match="unsupported"):
        local_input_format(["a.csv"])


def test_pinned_local_parquet_streams_through_detected_loader(tmp_path):
    source = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["first retained document", "second retained document"],
                "int_score": [4, 5],
            }
        ),
        source,
    )
    paths, receipts = resolve_local_inputs([source], "pinned-revision")
    rows = list(
        load_dataset(
            local_input_format(paths),
            data_files=paths,
            split="train",
            streaming=True,
        )
    )
    assert rows == [
        {"text": "first retained document", "int_score": 4},
        {"text": "second retained document", "int_score": 5},
    ]
    assert receipts == [file_receipt(source)]
