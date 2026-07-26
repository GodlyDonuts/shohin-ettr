from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from audit_ettr_il_v2_specs import (
    ARMS_SPEC,
    COMPONENT_SPECS,
    CUSTODY_SPEC,
    SEMANTIC_SPEC,
    SpecAuditError,
    audit_specs,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_specs(tmp_path: Path) -> Path:
    for name in COMPONENT_SPECS:
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def test_current_v2_specs_pass_integration_audit() -> None:
    result = audit_specs(ROOT)
    assert result.split_spec_sha256 == (
        "a09f82684c8a118a633b0bb23e244de961166ebdd3593485d897c8c27deb9747"
    )
    assert len(result.checks) >= 10
    assert set(result.component_sha256) == set(COMPONENT_SPECS)


def test_retired_causal_schedule_fails(tmp_path: Path) -> None:
    root = _copy_specs(tmp_path)
    path = root / SEMANTIC_SPEC
    path.write_text(
        path.read_text(encoding="ascii")
        + "\nA global update has eight causal rectangles = 32 rows.\n",
        encoding="ascii",
    )
    with pytest.raises(SpecAuditError, match="retired causal-rectangle schedule"):
        audit_specs(root)


def test_eight_position_transaction_horizon_fails(tmp_path: Path) -> None:
    root = _copy_specs(tmp_path)
    path = root / ARMS_SPEC
    path.write_text(
        path.read_text(encoding="ascii")
        + "\nTransaction targets have exactly eight positions.\n",
        encoding="ascii",
    )
    with pytest.raises(SpecAuditError, match="retired eight-position horizon"):
        audit_specs(root)


def test_split_preimage_mutation_fails(tmp_path: Path) -> None:
    root = _copy_specs(tmp_path)
    path = root / CUSTODY_SPEC
    text = path.read_text(encoding="ascii")
    path.write_text(
        text.replace(
            '"renderer_ids":[0,1,2,3]',
            '"renderer_ids":[0,1,2]',
            1,
        ),
        encoding="ascii",
    )
    with pytest.raises(SpecAuditError, match="split byte count"):
        audit_specs(root)
