"""Counterfactually complete role supervision for DIVERGE-RRG1."""

from __future__ import annotations

from collections import Counter
import hashlib
import random
from typing import Any, Literal, Mapping, Sequence

from diverge_iem1_data import canonical_sha256, _symbol_role_ids
from diverge_nve1_data import MAX_EVIDENCE_BYTES, symbol_occurrence_groups


SCHEMA = "shohin-diverge-rrg1-role-training-v1"
DATA_SEED = 2026080624
SOURCE_ROWS_PER_STAGE = 50_000
ROWS_PER_STAGE = 100_000
LEXICAL_FAMILIES = 10
CLAUSE_FORMS = 2
Stage = Literal["EVIDENCE", "QUERY"]


class RRG1DataError(RuntimeError):
    """An RRG1 training record violates the frozen role contract."""


_EVIDENCE_TARGET = (
    "verified register {target} owns value {value} after instruction {step}",
    "the certified value {value} belongs to register {target} at step {step}",
    "instruction {step} leaves approved register {target} with value {value}",
    "checked register {target} contains value {value} once step {step} ends",
    "use register {target} as the value {value} holder after instruction {step}",
    "value {value} is attached to the verified register {target} at step {step}",
    "the trusted register {target} reads value {value} following step {step}",
    "after instruction {step}, report value {value} from register {target}",
    "register {target} is the accepted location for value {value} at step {step}",
    "the valid step-{step} value {value} comes from register {target}",
)

_EVIDENCE_DISTRACTOR = (
    "ignore decoy register {distractor}",
    "reject the decoy register {distractor}",
    "do not use distractor register {distractor}",
    "exclude the irrelevant register {distractor}",
    "register {distractor} is only the decoy",
    "discard the false register {distractor}",
    "the rejected alternative is register {distractor}",
    "avoid reading decoy register {distractor}",
    "register {distractor} must not supply the value",
    "treat register {distractor} as irrelevant",
)

_QUERY_TARGET = (
    "report the requested value from register {target}",
    "return the answer held by register {target}",
    "choose register {target} for the final response",
    "read the result from the requested register {target}",
    "use register {target} as the answer source",
    "the answer must come from register {target}",
    "select register {target} for this request",
    "take the final value from register {target}",
    "register {target} supplies the requested result",
    "answer using the value in register {target}",
)

_QUERY_DISTRACTOR = (
    "ignore decoy register {distractor}",
    "reject the alternative register {distractor}",
    "do not answer from register {distractor}",
    "exclude the decoy register {distractor}",
    "register {distractor} is irrelevant to the answer",
    "discard register {distractor} as a distractor",
    "the rejected choice is register {distractor}",
    "avoid the decoy value in register {distractor}",
    "register {distractor} must not be reported",
    "treat register {distractor} only as a decoy",
)


def _nonce(identity: str) -> str:
    digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:10]
    return "".join(chr(ord("a") + int(value, 16)) for value in digest)


def render_training_text(
    stage: Stage,
    *,
    family: int,
    clause_form: int,
    role_order: int,
    nonce: str,
    target: str,
    distractor: str,
    step: int | None = None,
    value: str | None = None,
) -> str:
    if family not in range(LEXICAL_FAMILIES):
        raise RRG1DataError("RRG1 lexical family differs")
    if clause_form not in range(CLAUSE_FORMS) or role_order not in (0, 1):
        raise RRG1DataError("RRG1 clause geometry differs")
    fields: dict[str, object] = {
        "target": target,
        "distractor": distractor,
        "step": step,
        "value": value,
    }
    if stage == "EVIDENCE":
        if step is None or value is None:
            raise RRG1DataError("RRG1 evidence semantics are absent")
        target_clause = _EVIDENCE_TARGET[family].format(**fields)
        distractor_clause = _EVIDENCE_DISTRACTOR[family].format(**fields)
    elif stage == "QUERY":
        target_clause = _QUERY_TARGET[family].format(**fields)
        distractor_clause = _QUERY_DISTRACTOR[family].format(**fields)
    else:
        raise RRG1DataError(f"unknown RRG1 stage: {stage}")
    first, second = (
        (target_clause, distractor_clause)
        if role_order == 0
        else (distractor_clause, target_clause)
    )
    if clause_form == 0:
        return f"Case {nonce}: {first}; {second}."
    return f"{first}; {second}, for case {nonce}."


def derive_training_records(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    stage: Stage,
    seed: int = DATA_SEED,
) -> list[dict[str, Any]]:
    if len(source_rows) != SOURCE_ROWS_PER_STAGE or seed != DATA_SEED:
        raise RRG1DataError("RRG1 source geometry differs")
    indices = list(range(len(source_rows)))
    stage_tag = 0x45564944 if stage == "EVIDENCE" else 0x51554552
    random.Random(seed ^ stage_tag).shuffle(indices)
    assignment = {
        source_index: (slot % LEXICAL_FAMILIES, (slot // LEXICAL_FAMILIES) % 2)
        for slot, source_index in enumerate(indices)
    }
    output: list[dict[str, Any]] = []
    for source_index, source in enumerate(source_rows):
        family, clause_form = assignment[source_index]
        original_identity = str(source["identity_sha256"])
        pair_identity = canonical_sha256(
            {
                "schema": SCHEMA,
                "stage": stage,
                "original_identity_sha256": original_identity,
                "family": family,
                "clause_form": clause_form,
            }
        )
        nonce = _nonce(pair_identity)
        target = str(source["target"])
        distractor = str(source["distractor"])
        symbols = [str(value) for value in source["symbols"]]
        for role_order in (0, 1):
            text = render_training_text(
                stage,
                family=family,
                clause_form=clause_form,
                role_order=role_order,
                nonce=nonce,
                target=target,
                distractor=distractor,
                step=(int(source["step_ordinal"]) if stage == "EVIDENCE" else None),
                value=(str(source["value"]) if stage == "EVIDENCE" else None),
            )
            record: dict[str, Any] = {
                "schema": SCHEMA,
                "split": "train",
                "stage": stage,
                "family": family,
                "clause_form": clause_form,
                "role_order": role_order,
                "pair_identity_sha256": pair_identity,
                "original_identity_sha256": original_identity,
                "original_source_sha256": str(source["source_sha256"]),
                "source_text": text,
                "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
                "symbols": symbols,
                "target": target,
                "distractor": distractor,
                "symbol_role_ids": _symbol_role_ids(
                    text,
                    symbols,
                    target=target,
                    distractor=distractor,
                ),
            }
            if stage == "EVIDENCE":
                record.update(
                    {
                        "step_ordinal": int(source["step_ordinal"]),
                        "value": str(source["value"]),
                    }
                )
            record["identity_sha256"] = canonical_sha256(record)
            validate_training_record(record)
            output.append(record)
    if len(output) != ROWS_PER_STAGE:
        raise RRG1DataError("RRG1 derived row count differs")
    validate_counterfactual_completeness(output, stage=stage)
    return output


def validate_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != SCHEMA or record.get("split") != "train":
        raise RRG1DataError("RRG1 training schema differs")
    stage = str(record.get("stage"))
    if stage not in ("EVIDENCE", "QUERY"):
        raise RRG1DataError("RRG1 training stage differs")
    text = str(record["source_text"])
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise RRG1DataError("RRG1 source is not ASCII") from error
    if not encoded or len(encoded) + 1 > MAX_EVIDENCE_BYTES:
        raise RRG1DataError("RRG1 source width differs")
    if hashlib.sha256(encoded).hexdigest() != record["source_sha256"]:
        raise RRG1DataError("RRG1 source commitment differs")
    family = int(record["family"])
    clause_form = int(record["clause_form"])
    role_order = int(record["role_order"])
    target = str(record["target"])
    distractor = str(record["distractor"])
    symbols = tuple(str(value) for value in record["symbols"])
    if target == distractor or target not in symbols or distractor not in symbols:
        raise RRG1DataError("RRG1 referent symbols differ")
    pair_identity = canonical_sha256(
        {
            "schema": SCHEMA,
            "stage": stage,
            "original_identity_sha256": str(record["original_identity_sha256"]),
            "family": family,
            "clause_form": clause_form,
        }
    )
    if pair_identity != record["pair_identity_sha256"]:
        raise RRG1DataError("RRG1 pair identity differs")
    expected = render_training_text(
        stage,  # type: ignore[arg-type]
        family=family,
        clause_form=clause_form,
        role_order=role_order,
        nonce=_nonce(pair_identity),
        target=target,
        distractor=distractor,
        step=(int(record["step_ordinal"]) if stage == "EVIDENCE" else None),
        value=(str(record["value"]) if stage == "EVIDENCE" else None),
    )
    if text != expected:
        raise RRG1DataError("RRG1 rendered source differs")
    groups = symbol_occurrence_groups(text, symbols)
    if len(groups) != 2 or any(len(spans) != 1 for _, spans in groups):
        raise RRG1DataError("RRG1 mention geometry differs")
    expected_roles = _symbol_role_ids(
        text,
        symbols,
        target=target,
        distractor=distractor,
    )
    if list(record["symbol_role_ids"]) != expected_roles:
        raise RRG1DataError("RRG1 role assignment differs")
    if expected_roles != ([0, 1] if role_order == 0 else [1, 0]):
        raise RRG1DataError("RRG1 counterfactual order differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise RRG1DataError("RRG1 row identity differs")


def validate_counterfactual_completeness(
    rows: Sequence[Mapping[str, Any]], *, stage: Stage
) -> None:
    if len(rows) != ROWS_PER_STAGE:
        raise RRG1DataError("RRG1 stage row count differs")
    pair_orders: dict[str, set[int]] = {}
    family_order = Counter()
    family_form_order = Counter()
    identities = set()
    for row in rows:
        validate_training_record(row)
        if row["stage"] != stage:
            raise RRG1DataError("RRG1 mixed-stage corpus")
        identity = str(row["identity_sha256"])
        if identity in identities:
            raise RRG1DataError("RRG1 duplicate row identity")
        identities.add(identity)
        pair = str(row["pair_identity_sha256"])
        order = int(row["role_order"])
        pair_orders.setdefault(pair, set()).add(order)
        family = int(row["family"])
        form = int(row["clause_form"])
        family_order[(family, order)] += 1
        family_form_order[(family, form, order)] += 1
    if len(pair_orders) != SOURCE_ROWS_PER_STAGE or any(
        orders != {0, 1} for orders in pair_orders.values()
    ):
        raise RRG1DataError("RRG1 counterfactual pairs are incomplete")
    expected_family = SOURCE_ROWS_PER_STAGE // LEXICAL_FAMILIES
    expected_family_form = expected_family // CLAUSE_FORMS
    if any(
        family_order[(family, order)] != expected_family
        for family in range(LEXICAL_FAMILIES)
        for order in (0, 1)
    ):
        raise RRG1DataError("RRG1 family role balance differs")
    if any(
        family_form_order[(family, form, order)] != expected_family_form
        for family in range(LEXICAL_FAMILIES)
        for form in range(CLAUSE_FORMS)
        for order in (0, 1)
    ):
        raise RRG1DataError("RRG1 family/form role balance differs")


__all__ = [
    "CLAUSE_FORMS",
    "DATA_SEED",
    "LEXICAL_FAMILIES",
    "ROWS_PER_STAGE",
    "RRG1DataError",
    "SCHEMA",
    "derive_training_records",
    "render_training_text",
    "validate_counterfactual_completeness",
    "validate_training_record",
]
