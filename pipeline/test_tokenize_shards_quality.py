#!/usr/bin/env python3
"""Regression checks for deterministic general-source quality filters."""

import pytest

from pipeline.tokenize_shards import (
    boilerplate_marker_count,
    file_receipt,
    domain_value,
    exact_text_hash,
    extraction_quality,
    field_value,
    max_line_repeat_fraction,
    verify_file_receipt,
)


def test_exact_hash_is_stable_and_text_sensitive():
    assert exact_text_hash("alpha") == exact_text_hash("alpha")
    assert exact_text_hash("alpha") != exact_text_hash("Alpha")


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
