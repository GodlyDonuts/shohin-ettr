from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

import pytest

from ettr_il_v2_dataset import (
    CARDINALITY_RECEIPT_SCHEMA,
    DatasetBuildRequest,
    DatasetError,
    FIT_THEORIES,
    MaterializedBatchInput,
    SCORE_THEORIES,
    SelectedRectangle,
    SourceArtifact,
    audit_published_manifest,
    canonical_json_bytes,
    dry_run_dataset,
    publish_manifest_no_replace,
    _schedule_receipts,
    _validate_score_quotas,
    _validate_train_quotas,
)
from ettr_il_v2_materialize import (
    GenericInvariantPair,
    MaterializationRequest,
    materialize_ettr_il_v2,
)
from ettr_il_v2_schedule import InvariantPairRecord
from test_ettr_il_v2_materialize import TOKENIZER, _rectangle


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _selected(
    label: str,
    *,
    core: str,
    presentation: str,
    renderer: int,
    split: str = "train",
    world_ids: tuple[str, str] | None = None,
) -> SelectedRectangle:
    rectangle_id = _digest(f"rectangle|{label}")
    return SelectedRectangle(
        semantic_rectangle_id=rectangle_id,
        semantic_core_id=core,
        fold=0,
        split=split,
        ontology="rewrite",
        stratum="seen_id",
        depth=1,
        renderer=renderer,
        presentation=presentation,
        theory_pool_index=0,
        law_signature=_digest("law|rewrite|0"),
        world_ids=world_ids
        or (_digest(f"world|{split}|0"), _digest(f"world|{split}|1")),
        command_instance_ids=(
            _digest(f"instance|{split}|{core}|0"),
            _digest(f"instance|{split}|{core}|1"),
        ),
        command_template_ids=(_digest("template|0"), _digest("template|1")),
        opaque_name_ids=(
            _digest(f"opaque|{split}|{label}|0"),
            _digest(f"opaque|{split}|{label}|1"),
        ),
        raw_row_sha256=tuple(_digest(f"row|{split}|{label}|{i}") for i in range(16)),
        world_source_sha256=tuple(
            _digest(f"world-source|{split}|{label}|{i}") for i in range(4)
        ),
        command_source_sha256=tuple(
            _digest(f"command-source|{split}|{label}|{i}") for i in range(4)
        ),
        query_prefix_sha256=tuple(
            _digest(f"query-source|{split}|{label}|{i}") for i in range(4)
        ),
    )


def _write(path: Path, payload: bytes) -> SourceArtifact:
    path.write_bytes(payload)
    return SourceArtifact.from_path(
        role=path.stem,
        logical_name=path.name,
        path=path,
    )


def _sources(tmp_path: Path, rectangles: tuple[SelectedRectangle, ...]) -> tuple[SourceArtifact, ...]:
    report = _write(tmp_path / "cardinality_report", b"bounded-canary-report\n")
    validator = _write(tmp_path / "cardinality_validator", b"validator-source\n")
    core_keys = sorted(
        {
            (value.fold, value.split, value.semantic_core_id)
            for value in rectangles
        }
    )
    core_values = [
        {
            "fold": fold,
            "semantic_core_id": core_id,
            "split": split,
        }
        for fold, split, core_id in core_keys
    ]
    receipt_value = {
        "protocol": "R12-ETTR-IL-v2",
        "report_sha256": report.sha256,
        "result": "canary",
        "schema": CARDINALITY_RECEIPT_SCHEMA,
        "selected_semantic_core_count": len(core_values),
        "selected_semantic_core_ids_sha256": hashlib.sha256(
            canonical_json_bytes(core_values)
        ).hexdigest(),
        "validator_sha256": validator.sha256,
    }
    receipt_path = tmp_path / "cardinality_validation_receipt"
    receipt_path.write_bytes(canonical_json_bytes(receipt_value))
    receipt = SourceArtifact.from_path(
        role="cardinality_validation_receipt",
        logical_name=receipt_path.name,
        path=receipt_path,
    )
    return (
        replace(report, role="cardinality_report"),
        replace(validator, role="cardinality_validator"),
        receipt,
    )


def _request(tmp_path: Path) -> DatasetBuildRequest:
    core = _digest("core")
    left = _selected("left", core=core, presentation="base", renderer=0)
    right = _selected(
        "right",
        core=core,
        presentation="alpha_reorder",
        renderer=1,
    )
    generic_left = _rectangle(left.semantic_rectangle_id, source_suffix=b"-L")
    generic_right = _rectangle(right.semantic_rectangle_id, source_suffix=b"-R")
    generic_pair = GenericInvariantPair(left_rectangle=0, right_rectangle=1)
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256="a" * 64,
            dataset_sha256="b" * 64,
            vocab_size=512,
            rectangles=(generic_left, generic_right),
            invariant_pairs=(generic_pair,),
        ),
        TOKENIZER,
    )
    pair = InvariantPairRecord(
        pair_id=_digest("pair"),
        semantic_core_id=core,
        ontology="rewrite",
        depth=1,
        left_semantic_rectangle_id=left.semantic_rectangle_id,
        right_semantic_rectangle_id=right.semantic_rectangle_id,
    )
    rectangles = (left, right)
    return DatasetBuildRequest(
        mode="canary",
        vocab_size=512,
        rectangles=rectangles,
        invariant_pairs=(pair,),
        materialized_batches=(
            MaterializedBatchInput(
                fold=0,
                split="train",
                rectangle_ids=(
                    left.semantic_rectangle_id,
                    right.semantic_rectangle_id,
                ),
                batch=batch,
            ),
        ),
        sources=_sources(tmp_path, rectangles),
    )


def test_canary_dry_run_is_deterministic_source_bound_and_never_fits(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    first = dry_run_dataset(request)
    second = dry_run_dataset(
        replace(
            request,
            rectangles=tuple(reversed(request.rectangles)),
            sources=tuple(reversed(request.sources)),
        )
    )
    assert first.manifest_bytes == second.manifest_bytes
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.manifest["claim_status"] == (
        "canary_only_no_fit_no_dynamic_feasibility_claim"
    )
    assert first.manifest["schedule_status"] == "not_emitted_for_partial_canary"
    assert first.manifest["schedule_receipts"] == []
    assert first.manifest["cardinality"][
        "dynamic_counts_recomputed_by_dataset_builder"
    ] is False
    assert first.manifest["counts"] == {
        "cells": [
            {
                "count": 2,
                "fold": 0,
                "ontology": "rewrite",
                "split": "train",
                "stratum": "seen_id",
            }
        ],
        "causal_rectangles": 8,
        "query_rows": 32,
        "semantic_cores": 1,
        "semantic_rectangles": 2,
    }
    assert first.frozen_batches[0].manifest_sha256 == first.manifest_sha256
    assert first.frozen_batches[0].dataset_sha256 == first.dataset_sha256


def test_publication_is_canonical_read_only_and_no_replace(tmp_path: Path) -> None:
    result = dry_run_dataset(_request(tmp_path))
    destination = tmp_path / "manifest.json"
    assert publish_manifest_no_replace(result, destination) == result.manifest_sha256
    assert destination.read_bytes() == result.manifest_bytes
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    assert (
        audit_published_manifest(
            destination,
            expected_sha256=result.manifest_sha256,
        )
        == result.manifest
    )
    with pytest.raises(DatasetError, match="already exists"):
        publish_manifest_no_replace(result, destination)


def test_rejects_source_substitution_and_noncanonical_cardinality_receipt(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request.sources[0].path.write_bytes(b"substituted\n")
    with pytest.raises(DatasetError, match="source identity changed"):
        dry_run_dataset(request)

    fresh_dir = tmp_path / "partial"
    fresh_dir.mkdir()
    fresh = _request(fresh_dir)
    with pytest.raises(DatasetError, match="source bundle is incomplete"):
        dry_run_dataset(replace(fresh, sources=(fresh.sources[0],)))


def test_rejects_split_leakage_batch_mismatch_and_pair_geometry(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    development = _selected(
        "development",
        core=_digest("development-core"),
        presentation="base",
        renderer=0,
        split="development",
        world_ids=request.rectangles[0].world_ids,
    )
    altered_rectangles = request.rectangles + (development,)
    altered_source_dir = tmp_path / "altered-sources"
    altered_source_dir.mkdir()
    with pytest.raises(DatasetError, match="split-disjoint world_ids overlap"):
        dry_run_dataset(
            replace(
                request,
                rectangles=altered_rectangles,
                sources=_sources(altered_source_dir, altered_rectangles),
            )
        )

    with pytest.raises(DatasetError, match="do not partition"):
        dry_run_dataset(replace(request, materialized_batches=()))

    bad_pair = replace(
        request.invariant_pairs[0],
        right_semantic_rectangle_id=_digest("absent"),
    )
    with pytest.raises(DatasetError, match="non-training rectangle"):
        dry_run_dataset(replace(request, invariant_pairs=(bad_pair,)))


def test_rejects_noncanonical_json_and_production_shortcuts(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="non-canonical value"):
        canonical_json_bytes({"float": 1.5})
    with pytest.raises(DatasetError, match="non-ASCII"):
        canonical_json_bytes({"text": "é"})

    request = _request(tmp_path)
    with pytest.raises(DatasetError, match="production rectangle population"):
        dry_run_dataset(replace(request, mode="production"))


def test_cardinality_receipt_is_not_a_dynamic_count_claim(tmp_path: Path) -> None:
    request = _request(tmp_path)
    receipt = next(
        source
        for source in request.sources
        if source.role == "cardinality_validation_receipt"
    )
    value = json.loads(receipt.path.read_text("ascii"))
    value["selected_semantic_core_count"] = 999
    receipt.path.chmod(0o600)
    receipt.path.write_bytes(canonical_json_bytes(value))
    changed = SourceArtifact.from_path(
        role=receipt.role,
        logical_name=receipt.logical_name,
        path=receipt.path,
    )
    sources = tuple(changed if item is receipt else item for item in request.sources)
    with pytest.raises(DatasetError, match="receipt identity differs"):
        dry_run_dataset(replace(request, sources=sources))


@dataclass(frozen=True, slots=True)
class _QuotaView:
    fold: int
    split: str
    ontology: str
    stratum: str
    depth: int
    renderer: int
    presentation: str
    semantic_core_id: str
    theory_pool_index: int
    law_signature: str


def _train_quota_views() -> list[_QuotaView]:
    views: list[_QuotaView] = []
    bundles = (
        ("base", 0),
        ("alpha_reorder", 1),
        ("base", 1),
        ("alias_split", 0),
    )
    fit_ontologies = {
        0: ("rewrite", "resource"),
        1: ("horn", "resource"),
        2: ("horn", "rewrite"),
    }
    for fold in range(3):
        for ontology in fit_ontologies[fold]:
            for depth in (1, 2, 3):
                for core_index in range(96):
                    core = _digest(
                        f"quota-train|{fold}|{ontology}|{depth}|{core_index}"
                    )
                    theory = FIT_THEORIES[ontology][
                        core_index % len(FIT_THEORIES[ontology])
                    ]
                    views.extend(
                        _QuotaView(
                            fold=fold,
                            split="train",
                            ontology=ontology,
                            stratum="seen_id",
                            depth=depth,
                            renderer=renderer,
                            presentation=presentation,
                            semantic_core_id=core,
                            theory_pool_index=theory,
                            law_signature=_digest(f"law|{ontology}|{theory}"),
                        )
                        for presentation, renderer in bundles
                    )
    return views


def _score_quota_views() -> list[_QuotaView]:
    views: list[_QuotaView] = []
    strata = (
        "seen_id",
        "rule",
        "composition",
        "renderer",
        "rule_composition",
        "rule_renderer",
        "composition_renderer",
        "all_axes",
    )
    renderer_shift = {
        "renderer",
        "rule_renderer",
        "composition_renderer",
        "all_axes",
    }
    composition_shift = {
        "composition",
        "rule_composition",
        "composition_renderer",
        "all_axes",
    }
    for fold in range(3):
        for split in ("development", "confirmation"):
            for ontology in ("horn", "rewrite", "resource"):
                for stratum in strata:
                    all_axes = stratum == "all_axes"
                    core_count = 24 if all_axes else 32
                    presentations = (
                        (
                            "base",
                            "relation_reification",
                            "type_twin",
                            "execution_semantics_twin",
                        )
                        if all_axes
                        else ("base", "alpha_reorder", "alias_split")
                    )
                    depth_domain = (
                        (4, 5, 6)
                        if stratum in composition_shift
                        else (1, 2, 3)
                    )
                    depth_core_counts = (8, 8, 8) if all_axes else (11, 11, 10)
                    depths = tuple(
                        depth
                        for depth, count in zip(
                            depth_domain, depth_core_counts, strict=True
                        )
                        for _ in range(count)
                    )
                    theory_pool = (
                        SCORE_THEORIES[ontology]
                        if stratum
                        in {
                            "rule",
                            "rule_composition",
                            "rule_renderer",
                            "all_axes",
                        }
                        else FIT_THEORIES[ontology]
                    )
                    for core_index in range(core_count):
                        if stratum in renderer_shift:
                            renderer = (
                                3
                                if split == "confirmation"
                                else 2 + int(core_index >= core_count // 2)
                            )
                        else:
                            renderer = int(core_index >= core_count // 2)
                        core = _digest(
                            "quota-score|"
                            f"{fold}|{split}|{ontology}|{stratum}|{core_index}"
                        )
                        theory = theory_pool[core_index % len(theory_pool)]
                        views.extend(
                            _QuotaView(
                                fold=fold,
                                split=split,
                                ontology=ontology,
                                stratum=stratum,
                                depth=depths[core_index],
                                renderer=renderer,
                                presentation=presentation,
                                semantic_core_id=core,
                                theory_pool_index=theory,
                                law_signature=_digest(
                                    f"law|{ontology}|{theory}"
                                ),
                            )
                            for presentation in presentations
                        )
    return views


def _schedule_population(fold: int) -> tuple[InvariantPairRecord, ...]:
    fit_ontologies = {
        0: ("rewrite", "resource"),
        1: ("horn", "resource"),
        2: ("horn", "rewrite"),
    }
    values: list[InvariantPairRecord] = []
    for ontology in fit_ontologies[fold]:
        for depth in (1, 2, 3):
            for core_index in range(96):
                core = _digest(f"schedule-core|{fold}|{ontology}|{depth}|{core_index}")
                for pair_index in range(2):
                    values.append(
                        InvariantPairRecord(
                            pair_id=_digest(f"schedule-pair|{core}|{pair_index}"),
                            semantic_core_id=core,
                            ontology=ontology,
                            depth=depth,
                            left_semantic_rectangle_id=_digest(
                                f"schedule-left|{core}|{pair_index}"
                            ),
                            right_semantic_rectangle_id=_digest(
                                f"schedule-right|{core}|{pair_index}"
                            ),
                        )
                    )
    return tuple(values)


def test_exact_static_quota_and_schedule_identities_are_enforced() -> None:
    training = _train_quota_views()
    scoring = _score_quota_views()
    _validate_train_quotas(training, production=True)
    _validate_score_quotas(scoring, production=True)

    damaged_training = list(training)
    damaged_training[0] = replace(damaged_training[0], renderer=1)
    with pytest.raises(DatasetError, match="training view quotas differ"):
        _validate_train_quotas(damaged_training, production=True)

    damaged_scoring = list(scoring)
    damaged_scoring[0] = replace(damaged_scoring[0], depth=6)
    with pytest.raises(DatasetError, match="depth quota differs"):
        _validate_score_quotas(damaged_scoring, production=True)

    receipts = _schedule_receipts(
        {fold: _schedule_population(fold) for fold in range(3)},
        production=True,
    )
    assert len(receipts) == 15
    assert {receipt["updates"] for receipt in receipts} == {6000}
    assert {receipt["pair_population"] for receipt in receipts} == {1152}
    assert {receipt["pair_exposures"] for receipt in receipts} == {24000}
    assert len({receipt["schedule_sha256"] for receipt in receipts}) == 15
