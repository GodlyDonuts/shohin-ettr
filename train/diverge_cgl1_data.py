"""Outcome-only intervention orbits for DIVERGE-CGL1."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
from typing import Any, Mapping, Sequence

from diverge_iem1_data import canonical_sha256
from diverge_nve1_data import symbol_occurrence_groups
from diverge_rrg1_data import ROWS_PER_STAGE, validate_training_record


PUBLIC_SCHEMA = "shohin-diverge-cgl1-public-training-v1"
SUPERVISOR_SCHEMA = "shohin-diverge-cgl1-outcome-supervisor-v1"
REPORT_SCHEMA = "shohin-diverge-cgl1-data-report-v2"
DATA_SEED = 2026080701
STATE_ORBITS = 3


class CGL1DataError(RuntimeError):
    """A CGL1 outcome orbit violates the frozen contract."""


def _state_values(pair_identity: str, orbit: int) -> tuple[int, int]:
    if orbit not in range(STATE_ORBITS):
        raise CGL1DataError("CGL1 state orbit differs")
    seed = int(
        hashlib.sha256(f"{DATA_SEED}\0{pair_identity}".encode("ascii")).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)
    first = rng.randint(-127, 127)
    second = rng.randint(-127, 127)
    while second == first:
        second = rng.randint(-127, 127)
    if orbit == 0:
        return first, second
    if orbit == 1:
        return second, first
    equal = rng.randint(-127, 127)
    return equal, equal


def derive_outcome_orbits(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DATA_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != ROWS_PER_STAGE or seed != DATA_SEED:
        raise CGL1DataError("CGL1 source geometry differs")
    public_rows: list[dict[str, Any]] = []
    supervisors: list[dict[str, Any]] = []
    pair_members: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    value_counts = Counter()
    for source in rows:
        validate_training_record(source)
        if source["stage"] != "QUERY":
            raise CGL1DataError("CGL1 source stage differs")
        text = str(source["source_text"])
        symbols = tuple(str(value) for value in source["symbols"])
        target = str(source["target"])
        distractor = str(source["distractor"])
        groups = symbol_occurrence_groups(text, symbols)
        if len(groups) != 2 or {group[0] for group in groups} != {target, distractor}:
            raise CGL1DataError("CGL1 source mentions differ")
        pair_identity = str(source["pair_identity_sha256"])
        for orbit in range(STATE_ORBITS):
            target_value, distractor_value = _state_values(pair_identity, orbit)
            physical_values = {target: target_value, distractor: distractor_value}
            candidate_values = [physical_values[group[0]] for group in groups]
            public: dict[str, Any] = {
                "schema": PUBLIC_SCHEMA,
                "pair_identity_sha256": pair_identity,
                "source_identity_sha256": str(source["identity_sha256"]),
                "state_orbit": orbit,
                "source_text": text,
                "source_sha256": str(source["source_sha256"]),
                "symbols": list(symbols),
                "candidate_values": candidate_values,
            }
            public["identity_sha256"] = canonical_sha256(public)
            supervisor: dict[str, Any] = {
                "schema": SUPERVISOR_SCHEMA,
                "public_identity_sha256": public["identity_sha256"],
                "terminal_answer": target_value,
            }
            supervisor["identity_sha256"] = canonical_sha256(supervisor)
            validate_public_record(public)
            validate_supervisor_record(supervisor, public)
            public_rows.append(public)
            supervisors.append(supervisor)
            pair_members[pair_identity].append((public, supervisor))
            value_counts[tuple(candidate_values)] += 1

    if len(public_rows) != ROWS_PER_STAGE * STATE_ORBITS:
        raise CGL1DataError("CGL1 output count differs")
    if len({row["identity_sha256"] for row in public_rows}) != len(public_rows):
        raise CGL1DataError("CGL1 public identities collide")
    if len({row["identity_sha256"] for row in supervisors}) != len(supervisors):
        raise CGL1DataError("CGL1 supervisor identities collide")
    for pair_identity, members in pair_members.items():
        if len(members) != 2 * STATE_ORBITS:
            raise CGL1DataError(f"CGL1 pair orbit incomplete: {pair_identity}")
        by_orbit: defaultdict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for public, supervisor in members:
            by_orbit[int(public["state_orbit"])].append((public, supervisor))
        if set(by_orbit) != set(range(STATE_ORBITS)) or any(
            len(values) != 2 for values in by_orbit.values()
        ):
            raise CGL1DataError("CGL1 pair state coverage differs")
        for orbit, values in by_orbit.items():
            answers = {item[1]["terminal_answer"] for item in values}
            if len(answers) != 1:
                raise CGL1DataError("CGL1 clause-order answer is not invariant")
            candidate_sets = {tuple(sorted(item[0]["candidate_values"])) for item in values}
            if len(candidate_sets) != 1:
                raise CGL1DataError("CGL1 clause-order state differs")
            if orbit == 1:
                first_answer = by_orbit[0][0][1]["terminal_answer"]
                if next(iter(answers)) == first_answer:
                    raise CGL1DataError("CGL1 state swap did not change the answer")
            if orbit == 2 and any(
                item[0]["candidate_values"][0] != item[0]["candidate_values"][1]
                for item in values
            ):
                raise CGL1DataError("CGL1 equal-outcome orbit differs")

    forbidden = {"target", "distractor", "symbol_role_ids", "role_order", "gold_transaction"}
    if any(forbidden & set(row) for row in public_rows):
        raise CGL1DataError("CGL1 public record leaks a semantic role")
    report = {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "source_rows": len(rows),
        "public_rows": len(public_rows),
        "supervisor_rows": len(supervisors),
        "pairs": len(pair_members),
        "state_orbits": STATE_ORBITS,
        "clause_orders_per_orbit": 2,
        "distinct_candidate_value_pairs": len(value_counts),
        "public_forbidden_fields_absent": True,
        "distinct_outcome_rows": sum(
            count for (left, right), count in value_counts.items() if left != right
        ),
        "equal_outcome_rows": sum(
            count for (left, right), count in value_counts.items() if left == right
        ),
        "all_pair_orbits_complete": True,
    }
    return public_rows, supervisors, report


def validate_public_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != PUBLIC_SCHEMA:
        raise CGL1DataError("CGL1 public schema differs")
    text = str(record["source_text"])
    if hashlib.sha256(text.encode("ascii")).hexdigest() != record["source_sha256"]:
        raise CGL1DataError("CGL1 public source commitment differs")
    symbols = tuple(str(value) for value in record["symbols"])
    groups = symbol_occurrence_groups(text, symbols)
    if len(groups) != 2:
        raise CGL1DataError("CGL1 public mention geometry differs")
    values = tuple(int(value) for value in record["candidate_values"])
    if len(values) != 2:
        raise CGL1DataError("CGL1 candidate state differs")
    orbit = int(record["state_orbit"])
    if orbit not in range(STATE_ORBITS):
        raise CGL1DataError("CGL1 public orbit differs")
    if (orbit == 2) != (values[0] == values[1]):
        raise CGL1DataError("CGL1 equal-outcome orbit contract differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise CGL1DataError("CGL1 public identity differs")


def validate_supervisor_record(
    record: Mapping[str, Any], public: Mapping[str, Any]
) -> None:
    if record.get("schema") != SUPERVISOR_SCHEMA:
        raise CGL1DataError("CGL1 supervisor schema differs")
    if record.get("public_identity_sha256") != public.get("identity_sha256"):
        raise CGL1DataError("CGL1 supervisor/public identity differs")
    answer = int(record["terminal_answer"])
    if answer not in tuple(int(value) for value in public["candidate_values"]):
        raise CGL1DataError("CGL1 terminal answer is outside candidate outcomes")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise CGL1DataError("CGL1 supervisor identity differs")


__all__ = [
    "CGL1DataError",
    "DATA_SEED",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "STATE_ORBITS",
    "SUPERVISOR_SCHEMA",
    "derive_outcome_orbits",
    "validate_public_record",
    "validate_supervisor_record",
]
