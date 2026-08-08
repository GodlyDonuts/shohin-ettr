#!/usr/bin/env python3
"""Architecture receipts for the evidence-supported Phase-2 Shohin scales."""

from train import CONFIGS, parameter_count_for_config


def test_scale_candidate_geometry_and_parameter_bands():
    expected = {
        "shohin_390m": (380_000_000, 400_000_000),
        "shohin_920m": (900_000_000, 940_000_000),
    }
    for name, (lower, upper) in expected.items():
        config = CONFIGS[name]
        assert config["seq_len"] == 4096
        assert config["d_model"] % config["n_head"] == 0
        assert config["n_head"] % config["n_kv_head"] == 0
        count = parameter_count_for_config(config, vocab_size=49_152)
        assert lower <= count <= upper


def test_parameter_receipt_matches_tied_gqa_formula():
    config = {
        "n_layer": 1,
        "n_head": 4,
        "n_kv_head": 2,
        "d_model": 16,
        "d_ff": 32,
    }
    # Embedding 160 + attention 768 + MLP 1,536 + block norms 40 + final norm 16.
    assert parameter_count_for_config(config, vocab_size=10) == 2_520


def test_parameter_receipt_rejects_invalid_head_geometry():
    invalid_width = dict(CONFIGS["shohin_390m"], d_model=1025)
    try:
        parameter_count_for_config(invalid_width, vocab_size=49_152)
    except ValueError as exc:
        assert "divisible by n_head" in str(exc)
    else:
        raise AssertionError("invalid width was accepted")

    invalid_gqa = dict(CONFIGS["shohin_390m"], n_kv_head=3)
    try:
        parameter_count_for_config(invalid_gqa, vocab_size=49_152)
    except ValueError as exc:
        assert "divisible by n_kv_head" in str(exc)
    else:
        raise AssertionError("invalid GQA geometry was accepted")
