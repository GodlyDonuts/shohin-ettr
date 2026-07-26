from __future__ import annotations

import hashlib

import pytest

from ettr_il_v2_canary import (
    CanaryError,
    _contains_source_bytes,
    canonical_json_bytes,
)
from ettr_il_v2_semantics import Ontology


def test_canary_helpers_are_deterministic_and_source_sensitive() -> None:
    value = {"z": 1, "a": ["x"]}
    assert canonical_json_bytes(value) == b'{"a":["x"],"z":1}\n'
    assert hashlib.sha256(canonical_json_bytes(value)).hexdigest() == (
        hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    )
    assert _contains_source_bytes(({"x": b"secret"},))
    assert not _contains_source_bytes(({"x": "digest"},))


def test_canary_rejects_invalid_ontology_before_search() -> None:
    with pytest.raises(CanaryError, match="ontology"):
        from ettr_il_v2_canary import run_canary

        run_canary("horn", object())  # type: ignore[arg-type]


def test_ontology_iteration_is_frozen() -> None:
    assert tuple(ontology.value for ontology in Ontology) == (
        "horn",
        "rewrite",
        "resource",
    )
