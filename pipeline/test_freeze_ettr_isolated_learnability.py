from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from freeze_ettr_isolated_learnability import (
    MASTER_COMMITMENT,
    PREREG_NAME,
    ROW_SCHEMA,
    FreezeError,
    Geometry,
    audit_rows,
    build_dry_run_report,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    dataset_file_record,
    dataset_sha256,
    read_canonical_jsonl,
)


ROOT = Path(__file__).parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _candidate(seed: str) -> dict[str, str]:
    # Fourteen unique words guarantee at least two normalized 13-grams.
    words = " ".join(f"x{seed}{index:02d}" for index in range(14)).encode("ascii")
    return {
        "world_hex": words.hex(),
        "command_hex": f"c{seed}".encode("ascii").hex(),
        "query_hex": f"q{seed}".encode("ascii").hex(),
    }


def _rectangle(
    rectangle: str,
    *,
    split: str = "train",
    ontology: str = "horn",
    stratum: str = "seen_id",
    fold: int = 1,
    theory: int = 1,
    depth: int = 1,
    renderer: int = 0,
    presentation: str = "base",
) -> list[dict]:
    rows = []
    for world in range(2):
        for command in range(2):
            packet = _digest(f"{rectangle}:packet:{world}:{command}")
            terminal = _digest(f"{rectangle}:terminal:{world}:{command}")
            # Both axes flip at least one target and every balance group is 50/50.
            targets = (bool(world ^ command), bool(world))
            for semantic in range(2):
                for paraphrase in range(2):
                    seed = f"{rectangle}{world}{command}{semantic}{paraphrase}"
                    row = {
                        "schema": ROW_SCHEMA,
                        "row_id": "0" * 64,
                        "fold": fold,
                        "split": split,
                        "ontology": ontology,
                        "stratum": stratum,
                        "rectangle_id": _digest(rectangle),
                        "world_index": world,
                        "command_index": command,
                        "packet_id": packet,
                        "semantic_index": semantic,
                        "paraphrase_index": paraphrase,
                        "depth": depth,
                        "renderer": renderer,
                        "presentation": presentation,
                        "theory_index": theory,
                        "theory_sha256": _digest(
                            f"{split}:{ontology}:theory:{theory}:{rectangle}"
                        ),
                        "semantic_world_sha256": _digest(
                            f"{split}:{ontology}:world:{world}:{rectangle}"
                        ),
                        "command_sha256": _digest(
                            f"{split}:{ontology}:command:{command}:{rectangle}"
                        ),
                        "opaque_names": [f"n{seed}"],
                        "graph_sha256": _digest(f"{split}:graph:{seed}"),
                        "token_ids": [int(_digest(f"tokens:{seed}")[:8], 16)],
                        "candidate": _candidate(seed),
                        "terminal_sha256": terminal,
                        "target": targets[semantic],
                        "disposition": "answer",
                        "wrong_world_target_changed": True,
                        "wrong_command_target_changed": True,
                        "shuffled_state_target_changed": True,
                    }
                    material = {key: value for key, value in row.items() if key != "row_id"}
                    row["row_id"] = hashlib.sha256(
                        canonical_json_bytes(material)
                    ).hexdigest()
                    rows.append(row)
    return rows


def _rehash(row: dict) -> None:
    row["row_id"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in row.items() if key != "row_id"}
        )
    ).hexdigest()


def test_canonical_json_and_dataset_commitment_are_deterministic() -> None:
    left = canonical_json_bytes({"z": [3, 2], "a": {"q": True}})
    right = canonical_json_bytes({"a": {"q": True}, "z": [3, 2]})
    assert left == right == b'{"a":{"q":true},"z":[3,2]}\n'
    records = [
        dataset_file_record("b.jsonl", b"b\n", 1),
        dataset_file_record("a.jsonl", b"a\n", 1),
    ]
    assert dataset_sha256(records) == dataset_sha256(tuple(reversed(records)))


def test_repository_dry_run_is_deterministic_and_honestly_blocked() -> None:
    first = build_dry_run_report(ROOT)
    second = build_dry_run_report(ROOT)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["master_commitment"] == MASTER_COMMITMENT
    assert first["status"] == "blocked"
    assert first["materialization_authorized"] is False
    codes = {item["code"] for item in first["blocked_clauses"]}
    assert "split_spec_preimage_unavailable" in codes
    assert "renderer_grammar_unspecified" in codes
    assert "tokenizer_identity_unspecified" in codes
    assert "production_depth_support_insufficient" in codes
    assert "selection_key_encoding_unspecified" in codes
    assert first["production_capabilities"]["production_ontology_apis_available"]
    assert first["production_capabilities"]["resource_max_composition_depth"] == 3


def test_preloaded_production_module_substitute_cannot_spoof_audit() -> None:
    name = "cross_ontology_horn_board"
    previous = sys.modules.get(name)
    hostile = ModuleType(name)
    hostile.THEORIES = ()
    hostile.__file__ = "/tmp/hostile.py"
    sys.modules[name] = hostile
    try:
        report = build_dry_run_report(ROOT)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    assert report["production_capabilities"]["theory_pools"]["horn"]["total"] == 20


def test_prereg_hostile_mutation_is_detected(tmp_path: Path) -> None:
    source = (ROOT / PREREG_NAME).read_text("ascii")
    mutant = tmp_path / PREREG_NAME
    mutant.write_text(source.replace(MASTER_COMMITMENT, "f" * 64), "ascii")
    report = build_dry_run_report(ROOT, prereg_path=mutant)
    assert "prereg_anchor_missing" in {
        item["code"] for item in report["blocked_clauses"]
    }


def test_source_hostile_mutation_is_detected(tmp_path: Path) -> None:
    # A minimal shadow tree is sufficient: every absent source also fails closed.
    shadow = tmp_path / "repo"
    path = shadow / "train" / "endogenous_typed_theory_reactor.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        (ROOT / "train" / "endogenous_typed_theory_reactor.py").read_bytes()
        + b"\n# mutation\n"
    )
    report = build_dry_run_report(
        shadow,
        prereg_path=ROOT / PREREG_NAME,
    )
    details = [
        item["detail"]
        for item in report["blocked_clauses"]
        if item["code"] == "source_hash_mismatch"
    ]
    assert any("endogenous_typed_theory_reactor.py" in detail for detail in details)


def test_preflight_refuses_prohibited_optional_input_path(tmp_path: Path) -> None:
    path = tmp_path / "flagship_shard_split_spec.json"
    path.write_bytes(canonical_json_bytes({"hostile": True}))
    report = build_dry_run_report(ROOT, split_spec=path)
    assert report["input_commitments"]["split_spec_sha256"] is None
    assert any(
        item["code"] == "split_spec_preimage_unavailable"
        and "prohibited training asset" in item["detail"]
        for item in report["blocked_clauses"]
    )


def test_strict_row_audit_accepts_a_complete_tiny_rectangle() -> None:
    rows = _rectangle("r0")
    report = audit_rows(rows, require_full_counts=False)
    assert report["all_contracts_pass"]
    assert report["counts"] == {"rectangles": 1, "rows": 16}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("score_theory", "score-only theory"),
        ("score_renderer", "score-only renderer"),
        ("candidate_label", "leaks assessor words"),
        ("donor", "wrong_world_target_changed"),
        ("packet_collision", "terminal packets collide"),
        ("edge_collision", "WORLD edge changes no answer"),
    ],
)
def test_hostile_row_mutations_fail_closed(mutation: str, message: str) -> None:
    rows = _rectangle("r1")
    if mutation == "score_theory":
        rows[0]["theory_index"] = 0
        _rehash(rows[0])
    elif mutation == "score_renderer":
        rows[0]["renderer"] = 3
        _rehash(rows[0])
    elif mutation == "candidate_label":
        rows[0]["candidate"]["query_hex"] = b"oracle target".hex()
        _rehash(rows[0])
    elif mutation == "donor":
        rows[0]["wrong_world_target_changed"] = False
        _rehash(rows[0])
    elif mutation == "packet_collision":
        for row in rows:
            if row["world_index"] == 1 and row["command_index"] == 1:
                row["terminal_sha256"] = rows[0]["terminal_sha256"]
                _rehash(row)
    else:
        for row in rows:
            row["target"] = bool(row["command_index"] ^ row["semantic_index"])
            _rehash(row)
    with pytest.raises(FreezeError, match=message):
        audit_rows(rows, require_full_counts=False)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("semantic_world_sha256", "semantic_world_sha256 overlaps"),
        ("theory_sha256", "theory_sha256 overlaps"),
        ("command_sha256", "command_sha256 overlaps"),
        ("opaque_names", "opaque_name overlaps"),
        ("graph_sha256", "graph_sha256 overlaps"),
        ("token_ids", "token_sequence overlaps"),
        ("candidate", "normalized_13gram overlaps"),
    ],
)
def test_cross_split_overlap_mutations_fail_closed(field: str, message: str) -> None:
    train = _rectangle("train")
    development = _rectangle(
        "dev",
        split="development",
        stratum="seen_id",
        theory=3,
    )
    source = train[0]
    target = development[0]
    if field == "candidate":
        target[field]["world_hex"] = source[field]["world_hex"]
    else:
        target[field] = deepcopy(source[field])
    _rehash(target)
    with pytest.raises(FreezeError, match=message):
        audit_rows([*train, *development], require_full_counts=False)


def test_noncanonical_jsonl_is_rejected(tmp_path: Path) -> None:
    rows = _rectangle("jsonl")
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_bytes(canonical_jsonl_bytes(rows))
    assert read_canonical_jsonl(canonical) == rows

    mutant = tmp_path / "mutant.jsonl"
    mutant.write_text(
        "\n".join(json.dumps(row, sort_keys=False) for row in rows) + "\n",
        "ascii",
    )
    with pytest.raises(FreezeError, match="not canonical"):
        read_canonical_jsonl(mutant)


def test_jsonl_symlink_and_prohibited_asset_path_are_rejected(
    tmp_path: Path,
) -> None:
    rows = _rectangle("path")
    source = tmp_path / "rows.jsonl"
    source.write_bytes(canonical_jsonl_bytes(rows))
    link = tmp_path / "link.jsonl"
    link.symlink_to(source)
    with pytest.raises(FreezeError, match="non-symlink"):
        read_canonical_jsonl(link)

    prohibited = tmp_path / "checkpoint_rows.jsonl"
    prohibited.write_bytes(canonical_jsonl_bytes(rows))
    with pytest.raises(FreezeError, match="prohibited training asset"):
        read_canonical_jsonl(prohibited)


def test_nonfinite_json_is_rejected_as_noncanonical(tmp_path: Path) -> None:
    path = tmp_path / "nan.jsonl"
    path.write_bytes(b'{"value":NaN}\n')
    with pytest.raises(FreezeError, match="not canonical"):
        read_canonical_jsonl(path)


def test_materialize_cli_creates_no_output_when_blocked(tmp_path: Path) -> None:
    output = tmp_path / "frozen"
    script = ROOT / "pipeline" / "freeze_ettr_isolated_learnability.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "materialize",
            "--repo-root",
            str(ROOT),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "materialization blocked before output creation" in result.stderr
    assert not output.exists()


def test_full_count_geometry_rejects_partial_data() -> None:
    with pytest.raises(FreezeError, match="training rectangle count differs"):
        audit_rows(
            _rectangle("partial"),
            geometry=Geometry(
                train_rectangles_per_fit_ontology=1,
                scored_rectangles_per_cell=1,
            ),
            require_full_counts=True,
        )
