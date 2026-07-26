"""Locked CPU-only result evaluator for R12-ETTR-IL-v2.

The evaluator consumes one immutable, canonical aggregate panel.  It never
loads a model, fits parameters, selects a checkpoint, or accepts row-level
predictions.  The panel contains integer sufficient statistics clustered by
semantic core for all 75 preregistered update-6000 runs.

Input panel schema ``r12-ettr-il-v2-evaluation-panel-v1`` has exactly:

```
schema, protocol, split, split_plaintext_sha256, dataset_root_sha256,
endpoint_root_sha256, runs
```

Each run has exactly:

```
arm, fold, seed, checkpoint_update, checkpoint_sha256,
endpoint_receipt_sha256, optimizer_receipt_sha256, schedule_sha256,
dataset_sha256, source_payload_sha256, objective_weights_sha256,
arm_config_sha256, control_receipt_sha256, budget_receipt_sha256,
static_flop_receipt_sha256, custody_receipt_sha256, trainable_parameters,
complete_system_parameters, optimizer_updates, encoded_tokens,
query_row_exposures, charged_flops, static_loss_path_flops,
fit_semantic_core_exact, cells, diagnostics
```

``cells`` contains all 24 fold-local score cells.  A cell contains
``ontology``, ``stratum``, and 24 or 32 semantic-core records.  Core records
contain only ``semantic_core_id``, ``view_count``, and
``exact_causal_rectangles``.  ``diagnostics`` is empty for controls and, for
the treatment, contains all named absolute-gate integer counts.

The public file evaluator always uses the frozen 100,000-replicate
hierarchical paired bootstrap implemented by :mod:`ettr_il_v2_statistics`.
Confirmation additionally requires the frozen authorization document and
atomically creates a no-replace consume-on-attempt claim before panel access.
The local claim is a mechanical guard; independent WORM/CAS and signature
verification remain responsibilities of the custody broker.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import ettr_il_v2_statistics as statistics
from ettr_il_v2_schedule import MODEL_SEEDS


PROTOCOL = "R12-ETTR-IL-v2"
PANEL_SCHEMA = "r12-ettr-il-v2-evaluation-panel-v1"
RESULT_SCHEMA = "r12-ettr-il-v2-evaluation-result-v1"
OPEN_AUTHORIZATION_SCHEMA = "r12-ettr-il-v2-open-authorization-v1"
OPENING_CLAIM_SCHEMA = "r12-ettr-il-v2-evaluator-opening-claim-v1"

FOLDS = (0, 1, 2)
ONTOLOGIES = ("horn", "rewrite", "resource")
STRATA = (
    "seen_id",
    "rule",
    "composition",
    "renderer",
    "rule_composition",
    "rule_renderer",
    "composition_renderer",
    "all_axes",
)
ARMS = (
    "treatment",
    "state_reset",
    "binding_deranged",
    "query_only",
    "dense_state",
)
CONTROLS = ARMS[1:]
FIT_ONTOLOGIES = {
    0: ("rewrite", "resource"),
    1: ("horn", "resource"),
    2: ("horn", "rewrite"),
}
WITHHELD_ONTOLOGY = {
    fold: next(
        ontology
        for ontology in ONTOLOGIES
        if ontology not in FIT_ONTOLOGIES[fold]
    )
    for fold in FOLDS
}

CLAIM_BEARING_UPDATE = 6_000
TRAINABLE_PARAMETERS = 67_697_771
COMPLETE_SYSTEM_PARAMETERS = 192_779_435
ENCODED_TOKENS = 405_504_000
QUERY_ROW_EXPOSURES = 768_000
CHARGED_FLOPS = 164_710_301_589_504_000
EXPECTED_RUNS = len(ARMS) * len(FOLDS) * len(MODEL_SEEDS)
EXPECTED_CELLS = len(FOLDS) * len(ONTOLOGIES) * len(STRATA)
EXPECTED_ENDPOINTS = len(CONTROLS) * (1 + EXPECTED_CELLS)

OPEN_CONFIRMATION = "OPEN_CONFIRMATION"
CLOSE_WITHOUT_CONFIRMATION = "CLOSE_V2_WITHOUT_CONFIRMATION"
POSITIVE_DECISION = (
    "ettr_isolated_synthetic_learnability_systematic_transfer_"
    "and_typed_state_advantage_confirmed"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX128 = re.compile(r"[0-9a-f]{128}\Z")

_PANEL_FIELDS = {
    "dataset_root_sha256",
    "endpoint_root_sha256",
    "protocol",
    "runs",
    "schema",
    "split",
    "split_plaintext_sha256",
}
_RUN_FIELDS = {
    "arm",
    "arm_config_sha256",
    "budget_receipt_sha256",
    "charged_flops",
    "checkpoint_sha256",
    "checkpoint_update",
    "complete_system_parameters",
    "control_receipt_sha256",
    "custody_receipt_sha256",
    "dataset_sha256",
    "diagnostics",
    "encoded_tokens",
    "endpoint_receipt_sha256",
    "fit_semantic_core_exact",
    "fold",
    "objective_weights_sha256",
    "optimizer_receipt_sha256",
    "optimizer_updates",
    "query_row_exposures",
    "schedule_sha256",
    "seed",
    "source_payload_sha256",
    "static_flop_receipt_sha256",
    "static_loss_path_flops",
    "trainable_parameters",
    "cells",
}
_CORE_FIELDS = {
    "exact_causal_rectangles",
    "semantic_core_id",
    "view_count",
}
_COUNT_FIELDS = {"denominator", "numerator"}
_TREATMENT_DIAGNOSTIC_FIELDS = {
    "ambiguity_disposition_accuracy",
    "changed_answer_semantic_twin_separation",
    "command_intervention_answer_exact",
    "command_intervention_replay_agreement",
    "contradiction_disposition_accuracy",
    "gold_initial",
    "gold_terminal",
    "invariant_agreement",
    "learned_initial_packet_exact",
    "order_noncommuting_twin_separation",
    "poisoning_invariance",
    "post_seal_deletion_invariance",
    "query_twin_joint_accuracy",
    "source_replacement_invariance",
    "world_intervention_answer_exact",
    "world_intervention_replay_agreement",
}
_SCALAR_DIAGNOSTICS = _TREATMENT_DIAGNOSTIC_FIELDS - {
    "learned_initial_packet_exact"
}
_AUTHORIZATION_FIELDS = {
    "authorization_nonce",
    "authorizer_signature",
    "checkpoint_selection_allowed",
    "confirmation_envelope_roots",
    "dataset_root_sha256",
    "development_result_sha256",
    "endpoint_root_sha256",
    "evaluator_root_sha256",
    "expires_at_utc",
    "panel_sha256",
    "protocol",
    "rescore_allowed",
    "retry_allowed",
    "schema",
}


class EvaluationError(ValueError):
    """The panel or custody request differs from the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    path: str
    byte_count: int
    sha256: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "byte_count": self.byte_count,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CoreScore:
    semantic_core_id: str
    view_count: int
    exact_causal_rectangles: int

    @property
    def rate(self) -> Fraction:
        return Fraction(self.exact_causal_rectangles, 4 * self.view_count)


@dataclass(frozen=True, order=True, slots=True)
class CellKey:
    fold: int
    ontology: str
    stratum: str

    @property
    def core_count(self) -> int:
        return 24 if self.stratum == "all_axes" else 32

    @property
    def bootstrap_cell(self) -> statistics.BootstrapCell:
        return statistics.BootstrapCell(
            self.fold,
            self.ontology,
            self.stratum,
            self.core_count,
        )

    @property
    def text(self) -> str:
        return f"{self.fold}|{self.ontology}|{self.stratum}"


@dataclass(frozen=True, slots=True)
class RunScore:
    arm: str
    fold: int
    seed: int
    cells: Mapping[CellKey, tuple[CoreScore, ...]]
    fit: Mapping[str, Fraction]
    diagnostics: Mapping[str, Any]
    schedule_sha256: str
    dataset_sha256: str
    source_payload_sha256: str
    objective_weights_sha256: str
    static_loss_path_flops: int


@dataclass(frozen=True, slots=True)
class ParsedPanel:
    split: str
    split_plaintext_sha256: str
    dataset_root_sha256: str
    endpoint_root_sha256: str
    runs: Mapping[tuple[str, int, int], RunScore]


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} is not an object")
    return value


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise EvaluationError(f"{label} is not an array")
    return value


def _require_fields(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise EvaluationError(f"{label} fields differ")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise EvaluationError(f"{label} SHA-256 differs")
    return value


def _strict_json_value(value: object, label: str = "canonical JSON") -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and not value.isascii():
            raise EvaluationError(f"{label} contains non-ASCII text")
        return
    if _plain_int(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _strict_json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise EvaluationError(f"{label} has a non-ASCII key")
            _strict_json_value(item, f"{label}.{key}")
        return
    raise EvaluationError(f"{label} contains a non-canonical value")


def canonical_json_bytes(value: object) -> bytes:
    """Render strict deterministic ASCII JSON with one final newline."""

    _strict_json_value(value)
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError("canonical JSON rendering failed") from exc
    return (rendered + "\n").encode("ascii")


def _strict_loads(payload: bytes, label: str) -> object:
    if not payload.endswith(b"\n"):
        raise EvaluationError(f"{label} is not canonical JSON")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationError(f"{label} has a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _: (_ for _ in ()).throw(
                EvaluationError(f"{label} contains a float")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                EvaluationError(f"{label} contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{label} is not strict ASCII JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise EvaluationError(f"{label} is not canonical JSON")
    return value


def _read_immutable_canonical(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[object, ArtifactReceipt]:
    expected = _require_sha256(expected_sha256, f"expected {label}")
    target = Path(path).absolute()
    try:
        before = target.lstat()
    except OSError as exc:
        raise EvaluationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o222
    ):
        raise EvaluationError(
            f"{label} is not an immutable single-link regular file"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise EvaluationError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(opened) != identity(before):
            raise EvaluationError(f"{label} changed before read")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if identity(after) != identity(opened):
            raise EvaluationError(f"{label} changed during read")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise EvaluationError(f"{label} content hash differs")
    value = _strict_loads(payload, label)
    return value, ArtifactReceipt(str(target), len(payload), digest)


def _write_no_replace(
    path: str | Path,
    value: Mapping[str, object],
    *,
    label: str,
) -> ArtifactReceipt:
    target = Path(path)
    if not target.parent.is_dir():
        raise EvaluationError(f"{label} parent is unavailable")
    payload = canonical_json_bytes(dict(value))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags, 0o400)
    except OSError as exc:
        raise EvaluationError(f"{label} no-replace publication failed") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise EvaluationError(f"{label} write failed")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return ArtifactReceipt(
        str(target.resolve()),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _count(value: object, label: str, *, denominator: int | None = None) -> Fraction:
    item = _require_mapping(value, label)
    _require_fields(item, _COUNT_FIELDS, label)
    numerator = item["numerator"]
    total = item["denominator"]
    if (
        not _plain_int(numerator)
        or not _plain_int(total)
        or total <= 0
        or not 0 <= numerator <= total
    ):
        raise EvaluationError(f"{label} count differs")
    if denominator is not None and total != denominator:
        raise EvaluationError(f"{label} denominator differs")
    return Fraction(numerator, total)


def _parse_fit(value: object, fold: int, label: str) -> dict[str, Fraction]:
    records = _require_sequence(value, label)
    result: dict[str, Fraction] = {}
    for index, raw in enumerate(records):
        item = _require_mapping(raw, f"{label}[{index}]")
        _require_fields(
            item,
            {"denominator", "numerator", "ontology"},
            f"{label}[{index}]",
        )
        ontology = item["ontology"]
        if ontology not in FIT_ONTOLOGIES[fold] or ontology in result:
            raise EvaluationError(f"{label} ontology coverage differs")
        result[str(ontology)] = _count(
            {
                "denominator": item["denominator"],
                "numerator": item["numerator"],
            },
            f"{label}[{index}]",
            denominator=288,
        )
    if set(result) != set(FIT_ONTOLOGIES[fold]):
        raise EvaluationError(f"{label} ontology coverage differs")
    return result


def _parse_cells(
    value: object,
    *,
    fold: int,
    label: str,
) -> dict[CellKey, tuple[CoreScore, ...]]:
    records = _require_sequence(value, label)
    if len(records) != len(ONTOLOGIES) * len(STRATA):
        raise EvaluationError(f"{label} count differs")
    result: dict[CellKey, tuple[CoreScore, ...]] = {}
    seen_core_ids: set[str] = set()
    for index, raw_cell in enumerate(records):
        cell = _require_mapping(raw_cell, f"{label}[{index}]")
        _require_fields(
            cell,
            {"cores", "ontology", "stratum"},
            f"{label}[{index}]",
        )
        ontology = cell["ontology"]
        stratum = cell["stratum"]
        if ontology not in ONTOLOGIES or stratum not in STRATA:
            raise EvaluationError(f"{label}[{index}] identity differs")
        key = CellKey(fold, str(ontology), str(stratum))
        if key in result:
            raise EvaluationError(f"{label} has a duplicate cell")
        raw_cores = _require_sequence(cell["cores"], f"{label}[{index}].cores")
        if len(raw_cores) != key.core_count:
            raise EvaluationError(f"{label}[{index}] core count differs")
        expected_views = 4 if stratum == "all_axes" else 3
        parsed: list[CoreScore] = []
        for core_index, raw_core in enumerate(raw_cores):
            core = _require_mapping(
                raw_core,
                f"{label}[{index}].cores[{core_index}]",
            )
            _require_fields(
                core,
                _CORE_FIELDS,
                f"{label}[{index}].cores[{core_index}]",
            )
            core_id = _require_sha256(
                core["semantic_core_id"],
                f"{label}[{index}] semantic core",
            )
            views = core["view_count"]
            exact = core["exact_causal_rectangles"]
            if (
                views != expected_views
                or not _plain_int(exact)
                or not 0 <= exact <= 4 * expected_views
            ):
                raise EvaluationError(
                    f"{label}[{index}] semantic-core score differs"
                )
            if core_id in seen_core_ids:
                raise EvaluationError(
                    f"{label} semantic core occurs in multiple cells"
                )
            seen_core_ids.add(core_id)
            parsed.append(CoreScore(core_id, views, exact))
        parsed.sort(key=lambda item: item.semantic_core_id)
        result[key] = tuple(parsed)
    expected = {
        CellKey(fold, ontology, stratum)
        for ontology in ONTOLOGIES
        for stratum in STRATA
    }
    if set(result) != expected:
        raise EvaluationError(f"{label} cell coverage differs")
    return result


def _parse_diagnostics(
    value: object,
    *,
    arm: str,
    label: str,
) -> dict[str, Any]:
    item = _require_mapping(value, label)
    if arm != "treatment":
        if item:
            raise EvaluationError(f"{label} must be empty for controls")
        return {}
    _require_fields(item, _TREATMENT_DIAGNOSTIC_FIELDS, label)
    result: dict[str, Any] = {}
    for name in sorted(_SCALAR_DIAGNOSTICS):
        result[name] = _count(item[name], f"{label}.{name}")
    raw_initial = _require_sequence(
        item["learned_initial_packet_exact"],
        f"{label}.learned_initial_packet_exact",
    )
    by_ontology: dict[str, Fraction] = {}
    for index, raw in enumerate(raw_initial):
        record = _require_mapping(
            raw,
            f"{label}.learned_initial_packet_exact[{index}]",
        )
        _require_fields(
            record,
            {"denominator", "numerator", "ontology"},
            f"{label}.learned_initial_packet_exact[{index}]",
        )
        ontology = record["ontology"]
        if ontology not in ONTOLOGIES or ontology in by_ontology:
            raise EvaluationError(f"{label} initial-packet coverage differs")
        by_ontology[str(ontology)] = _count(
            {
                "denominator": record["denominator"],
                "numerator": record["numerator"],
            },
            f"{label}.learned_initial_packet_exact[{index}]",
        )
    if set(by_ontology) != set(ONTOLOGIES):
        raise EvaluationError(f"{label} initial-packet coverage differs")
    result["learned_initial_packet_exact"] = by_ontology
    return result


def _parse_run(value: object, label: str) -> RunScore:
    item = _require_mapping(value, label)
    _require_fields(item, _RUN_FIELDS, label)
    arm = item["arm"]
    fold = item["fold"]
    seed = item["seed"]
    if arm not in ARMS:
        raise EvaluationError(f"{label} arm differs")
    if fold not in FOLDS:
        raise EvaluationError(f"{label} fold differs")
    if seed not in MODEL_SEEDS:
        raise EvaluationError(f"{label} seed differs")
    if (
        item["checkpoint_update"] != CLAIM_BEARING_UPDATE
        or item["optimizer_updates"] != CLAIM_BEARING_UPDATE
    ):
        raise EvaluationError(f"{label} is not update-6000-only")
    for name in (
        "checkpoint_sha256",
        "endpoint_receipt_sha256",
        "optimizer_receipt_sha256",
        "schedule_sha256",
        "dataset_sha256",
        "source_payload_sha256",
        "objective_weights_sha256",
        "arm_config_sha256",
        "control_receipt_sha256",
        "budget_receipt_sha256",
        "static_flop_receipt_sha256",
        "custody_receipt_sha256",
    ):
        _require_sha256(item[name], f"{label}.{name}")
    exact_integers = {
        "charged_flops": CHARGED_FLOPS,
        "complete_system_parameters": COMPLETE_SYSTEM_PARAMETERS,
        "encoded_tokens": ENCODED_TOKENS,
        "query_row_exposures": QUERY_ROW_EXPOSURES,
        "trainable_parameters": TRAINABLE_PARAMETERS,
    }
    for name, expected in exact_integers.items():
        if item[name] != expected:
            raise EvaluationError(f"{label}.{name} differs")
    static_flops = item["static_loss_path_flops"]
    if not _plain_int(static_flops) or static_flops <= 0:
        raise EvaluationError(f"{label}.static_loss_path_flops differs")
    return RunScore(
        arm=str(arm),
        fold=int(fold),
        seed=int(seed),
        cells=_parse_cells(item["cells"], fold=int(fold), label=f"{label}.cells"),
        fit=_parse_fit(
            item["fit_semantic_core_exact"],
            int(fold),
            f"{label}.fit_semantic_core_exact",
        ),
        diagnostics=_parse_diagnostics(
            item["diagnostics"],
            arm=str(arm),
            label=f"{label}.diagnostics",
        ),
        schedule_sha256=str(item["schedule_sha256"]),
        dataset_sha256=str(item["dataset_sha256"]),
        source_payload_sha256=str(item["source_payload_sha256"]),
        objective_weights_sha256=str(item["objective_weights_sha256"]),
        static_loss_path_flops=int(static_flops),
    )


def parse_panel(value: object) -> ParsedPanel:
    """Validate and normalize the complete locked 75-run panel."""

    panel = _require_mapping(value, "evaluation panel")
    _require_fields(panel, _PANEL_FIELDS, "evaluation panel")
    if panel["schema"] != PANEL_SCHEMA or panel["protocol"] != PROTOCOL:
        raise EvaluationError("evaluation panel identity differs")
    split = panel["split"]
    if split not in {"development", "confirmation"}:
        raise EvaluationError("evaluation panel split differs")
    split_sha = _require_sha256(
        panel["split_plaintext_sha256"],
        "split plaintext",
    )
    dataset_root = _require_sha256(
        panel["dataset_root_sha256"],
        "dataset root",
    )
    endpoint_root = _require_sha256(
        panel["endpoint_root_sha256"],
        "endpoint root",
    )
    raw_runs = _require_sequence(panel["runs"], "evaluation panel.runs")
    if len(raw_runs) != EXPECTED_RUNS:
        raise EvaluationError("evaluation run count differs")
    runs: dict[tuple[str, int, int], RunScore] = {}
    for index, raw in enumerate(raw_runs):
        run = _parse_run(raw, f"evaluation panel.runs[{index}]")
        key = (run.arm, run.fold, run.seed)
        if key in runs:
            raise EvaluationError("evaluation run identity is duplicated")
        runs[key] = run
    expected_runs = {
        (arm, fold, seed)
        for arm in ARMS
        for fold in FOLDS
        for seed in MODEL_SEEDS
    }
    if set(runs) != expected_runs:
        raise EvaluationError("evaluation run coverage differs")
    if len({run.static_loss_path_flops for run in runs.values()}) != 1:
        raise EvaluationError("global static loss-path FLOP count differs")

    for fold in FOLDS:
        for seed in MODEL_SEEDS:
            group = [runs[(arm, fold, seed)] for arm in ARMS]
            for name in (
                "schedule_sha256",
                "dataset_sha256",
                "source_payload_sha256",
                "objective_weights_sha256",
                "static_loss_path_flops",
            ):
                if len({getattr(run, name) for run in group}) != 1:
                    raise EvaluationError(
                        f"equal-budget {name} differs within fold/seed"
                    )
            reference = group[0].cells
            for run in group[1:]:
                for cell in reference:
                    left = reference[cell]
                    right = run.cells[cell]
                    if tuple(
                        (core.semantic_core_id, core.view_count) for core in left
                    ) != tuple(
                        (core.semantic_core_id, core.view_count) for core in right
                    ):
                        raise EvaluationError(
                            "paired semantic-core population differs across arms"
                        )

    for fold in FOLDS:
        reference = runs[("treatment", fold, MODEL_SEEDS[0])].cells
        for seed in MODEL_SEEDS[1:]:
            candidate = runs[("treatment", fold, seed)].cells
            for cell in reference:
                if tuple(
                    (core.semantic_core_id, core.view_count)
                    for core in reference[cell]
                ) != tuple(
                    (core.semantic_core_id, core.view_count)
                    for core in candidate[cell]
                ):
                    raise EvaluationError(
                        "paired semantic-core population differs across seeds"
                    )

    return ParsedPanel(
        split=str(split),
        split_plaintext_sha256=split_sha,
        dataset_root_sha256=dataset_root,
        endpoint_root_sha256=endpoint_root,
        runs=runs,
    )


def _mean(values: Iterable[Fraction]) -> Fraction:
    items = tuple(values)
    if not items:
        raise EvaluationError("cannot aggregate an empty population")
    return sum(items, Fraction()) / len(items)


def _cell_rate(run: RunScore, cell: CellKey) -> Fraction:
    return _mean(core.rate for core in run.cells[cell])


def _all_cells() -> tuple[CellKey, ...]:
    return tuple(
        CellKey(fold, ontology, stratum)
        for fold in FOLDS
        for ontology in ONTOLOGIES
        for stratum in STRATA
    )


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _decimal_text(value: float | Fraction) -> str:
    numeric = float(value)
    if numeric == 0.0:
        numeric = 0.0
    return format(numeric, ".17g")


def _gate(
    gates: list[dict[str, object]],
    name: str,
    passed: bool,
    *,
    observed: Fraction | None = None,
    threshold: Fraction | None = None,
) -> None:
    record: dict[str, object] = {"gate": name, "passed": passed}
    if observed is not None:
        record["observed"] = _fraction_text(observed)
    if threshold is not None:
        record["threshold"] = _fraction_text(threshold)
    gates.append(record)


def _absolute_gates(panel: ParsedPanel) -> tuple[list[dict[str, object]], dict[str, bool]]:
    gates: list[dict[str, object]] = []
    categories = {
        "a4_qualification": True,
        "causal_use": True,
        "fit": True,
        "gold_initial": True,
        "gold_terminal": True,
        "initial_packet": True,
        "systematic_transfer": True,
        "withheld_transfer": True,
    }
    single_axis = ("rule", "composition", "renderer")
    two_axis = ("rule_composition", "rule_renderer", "composition_renderer")
    diagnostic_thresholds = {
        "ambiguity_disposition_accuracy": Fraction(95, 100),
        "changed_answer_semantic_twin_separation": Fraction(95, 100),
        "command_intervention_answer_exact": Fraction(90, 100),
        "command_intervention_replay_agreement": Fraction(1, 1),
        "contradiction_disposition_accuracy": Fraction(95, 100),
        "invariant_agreement": Fraction(99, 100),
        "order_noncommuting_twin_separation": Fraction(95, 100),
        "poisoning_invariance": Fraction(1, 1),
        "post_seal_deletion_invariance": Fraction(1, 1),
        "query_twin_joint_accuracy": Fraction(90, 100),
        "source_replacement_invariance": Fraction(1, 1),
        "world_intervention_answer_exact": Fraction(90, 100),
        "world_intervention_replay_agreement": Fraction(1, 1),
    }
    causal_names = {
        "command_intervention_answer_exact",
        "command_intervention_replay_agreement",
        "query_twin_joint_accuracy",
        "world_intervention_answer_exact",
        "world_intervention_replay_agreement",
    }

    for fold in FOLDS:
        for seed in MODEL_SEEDS:
            treatment = panel.runs[("treatment", fold, seed)]
            dense = panel.runs[("dense_state", fold, seed)]
            prefix = f"fold={fold}|seed={seed}"
            treatment_fit = _mean(
                treatment.fit[ontology]
                for ontology in FIT_ONTOLOGIES[fold]
            )
            treatment_fit_pass = treatment_fit >= Fraction(99, 100)
            _gate(
                gates,
                f"{prefix}|fit_semantic_core_exact",
                treatment_fit_pass,
                observed=treatment_fit,
                threshold=Fraction(99, 100),
            )
            categories["fit"] &= treatment_fit_pass
            dense_fit = _mean(
                dense.fit[ontology] for ontology in FIT_ONTOLOGIES[fold]
            )
            dense_fit_pass = dense_fit >= Fraction(99, 100)
            _gate(
                gates,
                f"{prefix}|a4_fit",
                dense_fit_pass,
                observed=dense_fit,
                threshold=Fraction(99, 100),
            )
            categories["a4_qualification"] &= dense_fit_pass
            for ontology in FIT_ONTOLOGIES[fold]:
                seen_cell = CellKey(fold, ontology, "seen_id")
                seen = _cell_rate(treatment, seen_cell)
                seen_pass = seen >= Fraction(95, 100)
                _gate(
                    gates,
                    f"{prefix}|seen_id|{ontology}",
                    seen_pass,
                    observed=seen,
                    threshold=Fraction(95, 100),
                )
                categories["systematic_transfer"] &= seen_pass
                dense_seen = _cell_rate(dense, seen_cell)
                dense_seen_pass = dense_seen >= Fraction(90, 100)
                _gate(
                    gates,
                    f"{prefix}|a4_seen_id|{ontology}",
                    dense_seen_pass,
                    observed=dense_seen,
                    threshold=Fraction(90, 100),
                )
                categories["a4_qualification"] &= dense_seen_pass

                for stratum in single_axis:
                    rate = _cell_rate(treatment, CellKey(fold, ontology, stratum))
                    passed = rate >= Fraction(85, 100)
                    _gate(
                        gates,
                        f"{prefix}|{ontology}|{stratum}",
                        passed,
                        observed=rate,
                        threshold=Fraction(85, 100),
                    )
                    categories["systematic_transfer"] &= passed
                for stratum in two_axis:
                    rate = _cell_rate(treatment, CellKey(fold, ontology, stratum))
                    passed = rate >= Fraction(80, 100)
                    _gate(
                        gates,
                        f"{prefix}|{ontology}|{stratum}",
                        passed,
                        observed=rate,
                        threshold=Fraction(80, 100),
                    )
                    categories["systematic_transfer"] &= passed
                all_axes = _cell_rate(
                    treatment,
                    CellKey(fold, ontology, "all_axes"),
                )
                all_axes_pass = all_axes >= Fraction(75, 100)
                _gate(
                    gates,
                    f"{prefix}|{ontology}|all_axes",
                    all_axes_pass,
                    observed=all_axes,
                    threshold=Fraction(75, 100),
                )
                categories["systematic_transfer"] &= all_axes_pass

            withheld = WITHHELD_ONTOLOGY[fold]
            withheld_rates = tuple(
                _cell_rate(treatment, CellKey(fold, withheld, stratum))
                for stratum in STRATA
            )
            withheld_macro = _mean(withheld_rates)
            withheld_macro_pass = withheld_macro >= Fraction(80, 100)
            _gate(
                gates,
                f"{prefix}|withheld_macro|{withheld}",
                withheld_macro_pass,
                observed=withheld_macro,
                threshold=Fraction(80, 100),
            )
            categories["withheld_transfer"] &= withheld_macro_pass
            withheld_all_axes = _cell_rate(
                treatment,
                CellKey(fold, withheld, "all_axes"),
            )
            withheld_axes_pass = withheld_all_axes >= Fraction(70, 100)
            _gate(
                gates,
                f"{prefix}|withheld_all_axes|{withheld}",
                withheld_axes_pass,
                observed=withheld_all_axes,
                threshold=Fraction(70, 100),
            )
            categories["withheld_transfer"] &= withheld_axes_pass

            for name, threshold in diagnostic_thresholds.items():
                rate = treatment.diagnostics[name]
                passed = rate >= threshold
                _gate(
                    gates,
                    f"{prefix}|{name}",
                    passed,
                    observed=rate,
                    threshold=threshold,
                )
                if name in causal_names:
                    categories["causal_use"] &= passed
                else:
                    categories["systematic_transfer"] &= passed
            initial = treatment.diagnostics["learned_initial_packet_exact"]
            for ontology in ONTOLOGIES:
                rate = initial[ontology]
                passed = rate >= Fraction(95, 100)
                _gate(
                    gates,
                    f"{prefix}|learned_initial_packet_exact|{ontology}",
                    passed,
                    observed=rate,
                    threshold=Fraction(95, 100),
                )
                categories["initial_packet"] &= passed
            ceiling_thresholds = {
                "gold_initial": Fraction(99, 100),
                "gold_terminal": Fraction(995, 1000),
            }
            for name, threshold in ceiling_thresholds.items():
                rate = treatment.diagnostics[name]
                passed = rate >= threshold
                _gate(
                    gates,
                    f"{prefix}|{name}",
                    passed,
                    observed=rate,
                    threshold=threshold,
                )
                categories[name] &= passed
    return gates, categories


def _endpoint_order() -> tuple[tuple[str, CellKey | None], ...]:
    cells = _all_cells()
    return tuple(
        (control, None)
        for control in CONTROLS
    ) + tuple(
        (control, cell)
        for control in CONTROLS
        for cell in cells
    )


def _observed_effects(
    panel: ParsedPanel,
) -> tuple[
    tuple[tuple[str, CellKey | None], ...],
    tuple[Fraction, ...],
]:
    endpoint_order = _endpoint_order()
    cell_effects: dict[tuple[str, CellKey], Fraction] = {}
    for control in CONTROLS:
        for cell in _all_cells():
            differences = (
                core_t.rate - core_c.rate
                for seed in MODEL_SEEDS
                for core_t, core_c in zip(
                    panel.runs[("treatment", cell.fold, seed)].cells[cell],
                    panel.runs[(control, cell.fold, seed)].cells[cell],
                    strict=True,
                )
            )
            cell_effects[(control, cell)] = _mean(differences)
    effects: list[Fraction] = []
    for control, cell in endpoint_order:
        if cell is None:
            effects.append(
                _mean(
                    cell_effects[(control, candidate)]
                    for candidate in _all_cells()
                )
            )
        else:
            effects.append(cell_effects[(control, cell)])
    if len(effects) != EXPECTED_ENDPOINTS:
        raise AssertionError("confirmatory endpoint count differs")
    return endpoint_order, tuple(effects)


def _bootstrap_effects(
    panel: ParsedPanel,
) -> Iterable[tuple[float, ...]]:
    cells = _all_cells()
    endpoint_order = _endpoint_order()
    differences = {
        (control, cell): tuple(
            tuple(
                float(core_t.rate - core_c.rate)
                for core_t, core_c in zip(
                    panel.runs[
                        ("treatment", cell.fold, seed)
                    ].cells[cell],
                    panel.runs[(control, cell.fold, seed)].cells[cell],
                    strict=True,
                )
            )
            for seed in MODEL_SEEDS
        )
        for control in CONTROLS
        for cell in cells
    }
    for replicate_index in range(statistics.BOOTSTRAP_REPLICATES):
        plan = statistics.build_bootstrap_plan(
            panel.split_plaintext_sha256,
            replicate_index,
            (cell.bootstrap_cell for cell in cells),
        )
        sampled_by_cell = {
            CellKey(value.fold, value.ontology, value.stratum): indices
            for value, indices in plan.cell_indices
        }
        cell_effects: dict[tuple[str, CellKey], float] = {}
        for control in CONTROLS:
            for cell in cells:
                core_indices = sampled_by_cell[cell]
                matrix = differences[(control, cell)]
                total = 0.0
                count = 0
                for seed_index in plan.model_seed_indices:
                    row = matrix[seed_index]
                    for core_index in core_indices:
                        total += row[core_index]
                        count += 1
                cell_effects[(control, cell)] = total / count
        values: list[float] = []
        for control, cell in endpoint_order:
            if cell is None:
                values.append(
                    sum(
                        cell_effects[(control, candidate)]
                        for candidate in cells
                    )
                    / len(cells)
                )
            else:
                values.append(cell_effects[(control, cell)])
        yield tuple(values)


def _compute_simultaneous_lcbs(
    panel: ParsedPanel,
    observed: tuple[Fraction, ...],
) -> tuple[float, ...]:
    """Execute the exact locked 100,000-replicate familywise correction."""

    if statistics.BOOTSTRAP_REPLICATES != 100_000:
        raise EvaluationError("bootstrap replicate contract differs")
    if len(observed) != EXPECTED_ENDPOINTS:
        raise EvaluationError("confirmatory endpoint family differs")
    return statistics.simultaneous_lower_bounds(
        tuple(float(value) for value in observed),
        _bootstrap_effects(panel),
    )


def _point_margin_gates(
    panel: ParsedPanel,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    gates: list[dict[str, object]] = []
    by_control = {control: True for control in CONTROLS}
    cells = _all_cells()
    for control in CONTROLS:
        for seed in MODEL_SEEDS:
            cell_effects: list[Fraction] = []
            for cell in cells:
                effect = _mean(
                    core_t.rate - core_c.rate
                    for core_t, core_c in zip(
                        panel.runs[
                            ("treatment", cell.fold, seed)
                        ].cells[cell],
                        panel.runs[(control, cell.fold, seed)].cells[cell],
                        strict=True,
                    )
                )
                cell_effects.append(effect)
                passed = effect >= Fraction(5, 100)
                _gate(
                    gates,
                    f"seed={seed}|control={control}|cell={cell.text}",
                    passed,
                    observed=effect,
                    threshold=Fraction(5, 100),
                )
                by_control[control] &= passed
            overall = _mean(cell_effects)
            passed = overall >= Fraction(10, 100)
            _gate(
                gates,
                f"seed={seed}|control={control}|overall",
                passed,
                observed=overall,
                threshold=Fraction(10, 100),
            )
            by_control[control] &= passed
    return gates, by_control


def _statistical_endpoints(
    panel: ParsedPanel,
) -> tuple[list[dict[str, object]], dict[str, bool], str]:
    endpoint_order, observed = _observed_effects(panel)
    lcbs = _compute_simultaneous_lcbs(panel, observed)
    if len(lcbs) != EXPECTED_ENDPOINTS:
        raise EvaluationError("simultaneous lower-bound count differs")
    records: list[dict[str, object]] = []
    by_control = {control: True for control in CONTROLS}
    for (control, cell), effect, lcb in zip(
        endpoint_order,
        observed,
        lcbs,
        strict=True,
    ):
        boundary = 0.10 if cell is None else 0.05
        point_pass = float(effect) >= boundary
        lower_bound_pass = lcb > boundary
        by_control[control] &= point_pass and lower_bound_pass
        records.append(
            {
                "boundary": _decimal_text(boundary),
                "cell": "overall" if cell is None else cell.text,
                "control": control,
                "lower_bound": _decimal_text(lcb),
                "lower_bound_strict_pass": lower_bound_pass,
                "observed_effect": _fraction_text(effect),
                "point_pass": point_pass,
            }
        )
    family_quantile = float(observed[0]) - lcbs[0]
    return records, by_control, _decimal_text(family_quantile)


def _localize_failure(
    *,
    categories: Mapping[str, bool],
    point_controls: Mapping[str, bool],
    statistical_controls: Mapping[str, bool],
    split: str,
) -> tuple[tuple[str, ...], str]:
    labels: list[str] = []
    if not categories["a4_qualification"]:
        labels.append("control_qualification")
    if not categories["gold_terminal"]:
        labels.append("query_reader")
    if categories["gold_terminal"] and not categories["gold_initial"]:
        labels.append("reactor_transaction_interface")
    if (
        categories["gold_terminal"]
        and categories["gold_initial"]
        and not categories["initial_packet"]
    ):
        labels.append("world_compiler")
    control_labels = {
        "state_reset": "recurrent_state_carry",
        "binding_deranged": "world_command_binding",
        "query_only": "query_bypass",
        "dense_state": "typed_sparse_state_advantage",
    }
    for control, localization in control_labels.items():
        if not point_controls[control] or not statistical_controls[control]:
            labels.append(localization)
    if categories["fit"] and not categories["systematic_transfer"]:
        labels.append("systematic_rule_composition_renderer_transfer")
    if not categories["withheld_transfer"]:
        labels.append("withheld_ontology_transfer")
    if not categories["causal_use"]:
        labels.append("architecture_native_causal_use")
    if labels:
        labels.append("seed_stability")
    if split == "confirmation" and labels:
        labels.append("confirmation_nonreplication")
    precedence = (
        "control_qualification",
        "query_reader",
        "reactor_transaction_interface",
        "world_compiler",
        "recurrent_state_carry",
        "world_command_binding",
        "query_bypass",
        "typed_sparse_state_advantage",
        "systematic_rule_composition_renderer_transfer",
        "withheld_ontology_transfer",
        "architecture_native_causal_use",
        "seed_stability",
        "confirmation_nonreplication",
    )
    unique = tuple(name for name in precedence if name in set(labels))
    return unique, unique[0] if unique else "none"


def aggregate_panel(panel: ParsedPanel) -> dict[str, object]:
    """Aggregate a previously validated complete panel without file I/O."""

    absolute_gates, categories = _absolute_gates(panel)
    point_gates, point_controls = _point_margin_gates(panel)
    statistical_endpoints, statistical_controls, family_quantile = (
        _statistical_endpoints(panel)
    )
    all_pass = (
        all(bool(gate["passed"]) for gate in absolute_gates)
        and all(bool(gate["passed"]) for gate in point_gates)
        and all(
            bool(endpoint["lower_bound_strict_pass"])
            and bool(endpoint["point_pass"])
            for endpoint in statistical_endpoints
        )
    )
    localizations, primary = _localize_failure(
        categories=categories,
        point_controls=point_controls,
        statistical_controls=statistical_controls,
        split=panel.split,
    )
    if panel.split == "development":
        decision = OPEN_CONFIRMATION if all_pass else CLOSE_WITHOUT_CONFIRMATION
    elif all_pass:
        decision = POSITIVE_DECISION
    else:
        decision = f"ettr_isolated_learnability_v2_rejected_at_{primary}"
    return {
        "absolute_gate_count": len(absolute_gates),
        "absolute_gates": absolute_gates,
        "all_gates_pass": all_pass,
        "bootstrap": {
            "cluster": "semantic_core",
            "family_quantile_0_95": family_quantile,
            "familywise_alpha": "0.05",
            "method": "hierarchical_paired_single_step_max_deviation",
            "replicates": statistics.BOOTSTRAP_REPLICATES,
            "simultaneous_one_sided_endpoints": EXPECTED_ENDPOINTS,
        },
        "decision": decision,
        "localizations": list(localizations),
        "point_margin_gate_count": len(point_gates),
        "point_margin_gates": point_gates,
        "primary_localization": primary,
        "statistical_endpoints": statistical_endpoints,
    }


def _source_receipt(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "byte_count": len(payload),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_authorization(
    value: object,
    *,
    authorization_sha256: str,
    expected_panel_sha256: str,
    expected_evaluator_root_sha256: str,
) -> Mapping[str, object]:
    item = _require_mapping(value, "confirmation authorization")
    _require_fields(item, _AUTHORIZATION_FIELDS, "confirmation authorization")
    if (
        item["schema"] != OPEN_AUTHORIZATION_SCHEMA
        or item["protocol"] != PROTOCOL
    ):
        raise EvaluationError("confirmation authorization identity differs")
    for name in (
        "dataset_root_sha256",
        "development_result_sha256",
        "endpoint_root_sha256",
        "evaluator_root_sha256",
        "panel_sha256",
    ):
        _require_sha256(item[name], f"confirmation authorization {name}")
    roots = _require_sequence(
        item["confirmation_envelope_roots"],
        "confirmation authorization envelope roots",
    )
    if len(roots) != len(FOLDS) or any(
        _HEX64.fullmatch(root) is None for root in roots if isinstance(root, str)
    ) or any(not isinstance(root, str) for root in roots):
        raise EvaluationError("confirmation envelope roots differ")
    if (
        item["panel_sha256"] != expected_panel_sha256
        or item["evaluator_root_sha256"] != expected_evaluator_root_sha256
    ):
        raise EvaluationError("confirmation authorization binding differs")
    if (
        item["rescore_allowed"] is not False
        or item["retry_allowed"] is not False
        or item["checkpoint_selection_allowed"] is not False
    ):
        raise EvaluationError("confirmation authorization policy differs")
    if (
        not isinstance(item["authorization_nonce"], str)
        or _HEX64.fullmatch(item["authorization_nonce"]) is None
        or not isinstance(item["authorizer_signature"], str)
        or _HEX128.fullmatch(item["authorizer_signature"]) is None
        or not isinstance(item["expires_at_utc"], str)
        or not item["expires_at_utc"].isascii()
        or not item["expires_at_utc"]
    ):
        raise EvaluationError("confirmation authorization custody fields differ")
    try:
        expires_at = datetime.strptime(
            str(item["expires_at_utc"]),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise EvaluationError(
            "confirmation authorization expiry differs"
        ) from exc
    if expires_at <= datetime.now(timezone.utc):
        raise EvaluationError("confirmation authorization is expired")
    _require_sha256(authorization_sha256, "authorization")
    return item


def evaluate_immutable_panel(
    *,
    panel_path: str | Path,
    expected_panel_sha256: str,
    result_path: str | Path,
    split: str,
    authorization_path: str | Path | None = None,
    expected_authorization_sha256: str | None = None,
    expected_evaluator_root_sha256: str | None = None,
    opening_claim_path: str | Path | None = None,
) -> ArtifactReceipt:
    """Evaluate one immutable panel and publish one immutable aggregate result.

    For confirmation, authorization is validated first, then the opening claim
    is created with ``O_EXCL`` before the panel is read.  Any subsequent error
    consumes the local attempt because the claim is never removed.
    """

    if split not in {"development", "confirmation"}:
        raise EvaluationError("requested split differs")
    expected_panel = _require_sha256(expected_panel_sha256, "expected panel")
    evaluator_source = Path(__file__).resolve()
    evaluator_source_receipt = _source_receipt(evaluator_source)
    statistics_source_receipt = _source_receipt(
        Path(statistics.__file__).resolve()
    )
    authorization_receipt: ArtifactReceipt | None = None
    claim_receipt: ArtifactReceipt | None = None
    authorization: Mapping[str, object] | None = None

    if split == "confirmation":
        if (
            authorization_path is None
            or expected_authorization_sha256 is None
            or expected_evaluator_root_sha256 is None
            or opening_claim_path is None
        ):
            raise EvaluationError(
                "confirmation authorization and opening claim are required"
            )
        raw_authorization, authorization_receipt = _read_immutable_canonical(
            authorization_path,
            expected_sha256=expected_authorization_sha256,
            label="confirmation authorization",
        )
        authorization = _validate_authorization(
            raw_authorization,
            authorization_sha256=authorization_receipt.sha256,
            expected_panel_sha256=expected_panel,
            expected_evaluator_root_sha256=expected_evaluator_root_sha256,
        )
        claim_receipt = _write_no_replace(
            opening_claim_path,
            {
                "access_count": 1,
                "authorization_sha256": authorization_receipt.sha256,
                "evaluator_root_sha256": expected_evaluator_root_sha256,
                "panel_sha256": expected_panel,
                "protocol": PROTOCOL,
                "schema": OPENING_CLAIM_SCHEMA,
                "state": "RESERVED_CONSUMED_ON_ATTEMPT",
            },
            label="confirmation opening claim",
        )
    elif any(
        value is not None
        for value in (
            authorization_path,
            expected_authorization_sha256,
            expected_evaluator_root_sha256,
            opening_claim_path,
        )
    ):
        raise EvaluationError("development must not consume confirmation custody")

    raw_panel, panel_receipt = _read_immutable_canonical(
        panel_path,
        expected_sha256=expected_panel,
        label=f"{split} evaluation panel",
    )
    panel = parse_panel(raw_panel)
    if panel.split != split:
        raise EvaluationError("panel split differs from requested split")
    if authorization is not None:
        if (
            authorization["dataset_root_sha256"] != panel.dataset_root_sha256
            or authorization["endpoint_root_sha256"] != panel.endpoint_root_sha256
        ):
            raise EvaluationError("confirmation panel authorization roots differ")

    aggregate = aggregate_panel(panel)
    result: dict[str, object] = {
        **aggregate,
        "authorization_receipt": (
            None
            if authorization_receipt is None
            else authorization_receipt.as_dict()
        ),
        "confirmation_access_count": 1 if split == "confirmation" else 0,
        "dataset_root_sha256": panel.dataset_root_sha256,
        "endpoint_root_sha256": panel.endpoint_root_sha256,
        "evaluator_source": evaluator_source_receipt,
        "opening_claim_receipt": (
            None if claim_receipt is None else claim_receipt.as_dict()
        ),
        "panel_receipt": panel_receipt.as_dict(),
        "protocol": PROTOCOL,
        "run_count": len(panel.runs),
        "schema": RESULT_SCHEMA,
        "selection": {
            "checkpoint_update": CLAIM_BEARING_UPDATE,
            "checkpoint_selection": "update_6000_only",
            "early_stopping": False,
            "seed_exclusion": False,
        },
        "split": split,
        "split_plaintext_sha256": panel.split_plaintext_sha256,
        "statistics_source": statistics_source_receipt,
        "status": "valid_locked_aggregate",
        "weight_updates": 0,
    }
    return _write_no_replace(
        result_path,
        result,
        label=f"{split} aggregate result",
    )


__all__ = [
    "ARMS",
    "ArtifactReceipt",
    "CLAIM_BEARING_UPDATE",
    "CLOSE_WITHOUT_CONFIRMATION",
    "CONTROLS",
    "EXPECTED_ENDPOINTS",
    "EXPECTED_RUNS",
    "EvaluationError",
    "FOLDS",
    "ONTOLOGIES",
    "OPEN_CONFIRMATION",
    "PANEL_SCHEMA",
    "POSITIVE_DECISION",
    "PROTOCOL",
    "RESULT_SCHEMA",
    "STRATA",
    "aggregate_panel",
    "canonical_json_bytes",
    "evaluate_immutable_panel",
    "parse_panel",
]
