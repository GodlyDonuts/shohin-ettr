from __future__ import annotations

import hashlib

from diverge_iem1_data import canonical_sha256, _symbol_role_ids
from diverge_rrg1_data import (
    RRG1DataError,
    SCHEMA,
    render_training_text,
    validate_training_record,
)


def _record(*, stage: str, order: int) -> dict[str, object]:
    family = 3
    clause_form = 1
    original_identity = "1" * 64
    pair = canonical_sha256(
        {
            "schema": SCHEMA,
            "stage": stage,
            "original_identity_sha256": original_identity,
            "family": family,
            "clause_form": clause_form,
        }
    )
    nonce = "abcdefghij"
    # render_training_text derives no state from the nonce, so use the exact
    # deterministic mapping expected by the validator.
    digest = hashlib.sha256(pair.encode("ascii")).hexdigest()[:10]
    nonce = "".join(chr(ord("a") + int(value, 16)) for value in digest)
    symbols = ["alpha", "bravo", "charlie", "delta", "echo"]
    text = render_training_text(
        stage,  # type: ignore[arg-type]
        family=family,
        clause_form=clause_form,
        role_order=order,
        nonce=nonce,
        target="alpha",
        distractor="charlie",
        step=7 if stage == "EVIDENCE" else None,
        value="-3/2" if stage == "EVIDENCE" else None,
    )
    record: dict[str, object] = {
        "schema": SCHEMA,
        "split": "train",
        "stage": stage,
        "family": family,
        "clause_form": clause_form,
        "role_order": order,
        "pair_identity_sha256": pair,
        "original_identity_sha256": original_identity,
        "original_source_sha256": "2" * 64,
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "symbols": symbols,
        "target": "alpha",
        "distractor": "charlie",
        "symbol_role_ids": _symbol_role_ids(
            text,
            symbols,
            target="alpha",
            distractor="charlie",
        ),
    }
    if stage == "EVIDENCE":
        record.update({"step_ordinal": 7, "value": "-3/2"})
    record["identity_sha256"] = canonical_sha256(record)
    return record


def _assert_record_is_exact_and_counterfactual(stage: str, order: int) -> None:
    record = _record(stage=stage, order=order)
    validate_training_record(record)
    expected = [0, 1] if order == 0 else [1, 0]
    assert record["symbol_role_ids"] == expected


def test_rrg1_records_are_exact_and_counterfactual() -> None:
    for stage in ("EVIDENCE", "QUERY"):
        for order in (0, 1):
            _assert_record_is_exact_and_counterfactual(stage, order)


def test_rrg1_record_fails_closed_on_role_change() -> None:
    record = _record(stage="QUERY", order=0)
    record["symbol_role_ids"] = [1, 0]
    record["identity_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "identity_sha256"}
    )
    try:
        validate_training_record(record)
    except RRG1DataError as error:
        assert "role assignment" in str(error)
    else:
        raise AssertionError("RRG1 accepted a changed role assignment")


def main() -> None:
    test_rrg1_records_are_exact_and_counterfactual()
    test_rrg1_record_fails_closed_on_role_change()
    print("DIVERGE-RRG1 data tests passed")


if __name__ == "__main__":
    main()
