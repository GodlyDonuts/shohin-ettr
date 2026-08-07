"""Natural, source-disjoint surfaces for conditional DIVERGE-NPL1."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from diverge_iem1_data import (
    _evidence_confirmation_text,
    _numeric_role_ids,
    _symbol_role_ids,
)
from diverge_pl1_data import Episode, Program, commitment
from diverge_srp1_data import query_text


SCHEMA = "shohin-diverge-npl1-development-v1"
DEVELOPMENT_SEED = 2026080811
DEVELOPMENT_COUNT = 256

_PROGRAM_PATTERN = re.compile(
    r"^Begin with (?P<x_name>[a-z]+) = (?P<x>[0-9]+) and "
    r"(?P<y_name>[a-z]+) = (?P<y>[0-9]+)\. "
    r"Execute aliases in order: (?P<aliases>[a-z]+(?: \| [a-z]+)*)\.$"
)


class NPL1DataError(RuntimeError):
    """An NPL1 natural surface or split violates the frozen contract."""


def _name(domain: str, episode_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{domain}:{episode_id}:{index}".encode("ascii")).digest()
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    chunks = []
    for offset in range(5):
        chunks.append(
            consonants[digest[2 * offset] % len(consonants)]
            + vowels[digest[2 * offset + 1] % len(vowels)]
        )
    return "".join(chunks)


def episode_names(episode: Episode) -> tuple[tuple[str, ...], tuple[str, str]]:
    branches = tuple(_name("branch", episode.episode_id, index) for index in range(8))
    registers = tuple(_name("register", episode.episode_id, index) for index in range(2))
    if len(set((*episode.aliases, *branches, *registers))) != 18:
        raise NPL1DataError("NPL1 episode-local names collide")
    return branches, (registers[0], registers[1])


def operation_aliases(episode: Episode) -> tuple[str, ...]:
    aliases = tuple(
        "nup" + _name("operation", episode.episode_id, index) for index in range(8)
    )
    if len(set(aliases)) != 8:
        raise NPL1DataError("NPL1 operation aliases collide")
    return aliases


def program_surface(
    program: Program,
    aliases: Sequence[str],
    registers: tuple[str, str],
) -> dict[str, object]:
    sequence = " | ".join(aliases[symbol] for symbol in program.symbols)
    text = (
        f"Begin with {registers[0]} = {program.initial_state[0]} and "
        f"{registers[1]} = {program.initial_state[1]}. "
        f"Execute aliases in order: {sequence}."
    )
    return {
        "program_id": program.program_id,
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "depth": len(program.symbols),
    }


def parse_program_surface(
    record: Mapping[str, object],
    aliases: Sequence[str],
    registers: tuple[str, str],
) -> tuple[tuple[int, int], tuple[int, ...]]:
    text = str(record["source_text"])
    match = _PROGRAM_PATTERN.fullmatch(text)
    if match is None or (match["x_name"], match["y_name"]) != registers:
        raise NPL1DataError("NPL1 WORLD structural surface differs")
    alias_to_index = {alias: index for index, alias in enumerate(aliases)}
    try:
        symbols = tuple(
            alias_to_index[value] for value in match["aliases"].split(" | ")
        )
    except KeyError as error:
        raise NPL1DataError("NPL1 WORLD exposes an undeclared alias") from error
    if int(record["depth"]) != len(symbols):
        raise NPL1DataError("NPL1 WORLD depth differs")
    if hashlib.sha256(text.encode("ascii")).hexdigest() != record["source_sha256"]:
        raise NPL1DataError("NPL1 WORLD source commitment differs")
    return (int(match["x"]), int(match["y"])), symbols


def render_feedback(plan: Mapping[str, object], certificate_code: int) -> str:
    if certificate_code < 0:
        raise NPL1DataError("NPL1 certificate code is negative")
    return _evidence_confirmation_text(
        int(plan["renderer"]),
        step=int(plan["attempt"]) + 1,
        value=str(certificate_code),
        target=str(plan["target_branch"]),
        distractor=str(plan["distractor_branch"]),
    )


def natural_public_record(episode: Episode) -> dict[str, object]:
    branches, registers = episode_names(episode)
    aliases = operation_aliases(episode)
    symbols = (*aliases, *branches, *registers)
    offset = int(episode.episode_id[:8], 16)
    acquisition = [
        program_surface(program, aliases, registers)
        for program in episode.acquisition
    ]
    transfer = [
        program_surface(program, aliases, registers)
        for program in episode.transfer
    ]
    feedback_plan = []
    for attempt in range(12):
        for branch in range(8):
            target = branches[branch]
            distractor = branches[(branch + (attempt % 7) + 1) % len(branches)]
            renderer = (offset + 8 * attempt + branch) % 3
            example = render_feedback(
                {
                    "renderer": renderer,
                    "attempt": attempt,
                    "target_branch": target,
                    "distractor_branch": distractor,
                },
                certificate_code=0,
            )
            feedback_plan.append(
                {
                    "attempt": attempt,
                    "branch": branch,
                    "renderer": renderer,
                    "target_branch": target,
                    "distractor_branch": distractor,
                    "numeric_role_ids": _numeric_role_ids(
                        example, renderer=renderer
                    ),
                    "symbol_role_ids": _symbol_role_ids(
                        example,
                        symbols,
                        target=target,
                        distractor=distractor,
                    ),
                }
            )
    queries = []
    for program_index, program in enumerate(episode.transfer):
        for register_index, target in enumerate(registers):
            distractor = registers[1 - register_index]
            renderer = (offset + 2 * program_index + register_index) % 6
            text = query_text(renderer, target=target, distractor=distractor)
            queries.append(
                {
                    "program_id": program.program_id,
                    "register_index": register_index,
                    "renderer": renderer,
                    "source_text": text,
                    "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
                    "target": target,
                    "distractor": distractor,
                    "symbol_role_ids": _symbol_role_ids(
                        text,
                        symbols,
                        target=target,
                        distractor=distractor,
                    ),
                }
            )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "episode_id": episode.episode_id,
        "split": episode.split,
        "aliases": list(aliases),
        "branch_names": list(branches),
        "register_names": list(registers),
        "symbol_table": list(symbols),
        "acquisition": acquisition,
        "transfer": transfer,
        "feedback_plan": feedback_plan,
        "queries": queries,
    }
    payload["identity_sha256"] = commitment("diverge-npl1-public", payload)
    validate_natural_public_record(payload)
    return payload


def natural_program_identities(episode: Episode) -> tuple[str, ...]:
    aliases = operation_aliases(episode)
    return tuple(
        commitment(
            "diverge-npl1-program-identity",
            [program.initial_state, [aliases[symbol] for symbol in program.symbols]],
        )
        for program in (*episode.acquisition, *episode.transfer)
    )


def natural_assessor_record(episode: Episode) -> dict[str, object]:
    public = natural_public_record(episode)
    payload = {
        "schema": SCHEMA,
        "public": public,
        "oracle": episode.assessor_record(),
    }
    payload["identity_sha256"] = commitment("diverge-npl1-assessor", payload)
    return payload


def validate_natural_public_record(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if commitment("diverge-npl1-public", payload) != identity:
        raise NPL1DataError("NPL1 public identity differs")
    if record.get("schema") != SCHEMA:
        raise NPL1DataError("NPL1 public schema differs")
    aliases = tuple(str(value) for value in record["aliases"])
    branches = tuple(str(value) for value in record["branch_names"])
    registers_raw = tuple(str(value) for value in record["register_names"])
    if len(aliases) != 8 or len(branches) != 8 or len(registers_raw) != 2:
        raise NPL1DataError("NPL1 symbol geometry differs")
    registers = (registers_raw[0], registers_raw[1])
    expected_table = (*aliases, *branches, *registers)
    if tuple(record["symbol_table"]) != expected_table or len(set(expected_table)) != 18:
        raise NPL1DataError("NPL1 symbol table differs")
    programs = (*record["acquisition"], *record["transfer"])
    if len(record["acquisition"]) != 12 or len(record["transfer"]) != 16:
        raise NPL1DataError("NPL1 program geometry differs")
    for program in programs:
        parse_program_surface(program, aliases, registers)
        if any(key in program for key in ("symbols", "trace", "terminal_state")):
            raise NPL1DataError("NPL1 public program leaks typed supervision")
    plans = record["feedback_plan"]
    if len(plans) != 96:
        raise NPL1DataError("NPL1 feedback plan geometry differs")
    if {int(plan["renderer"]) for plan in plans} != {0, 1, 2}:
        raise NPL1DataError("NPL1 feedback renderers are incomplete")
    if len(record["queries"]) != 32 or {
        int(query["renderer"]) for query in record["queries"]
    } != set(range(6)):
        raise NPL1DataError("NPL1 query renderers are incomplete")


__all__ = [
    "DEVELOPMENT_COUNT",
    "DEVELOPMENT_SEED",
    "NPL1DataError",
    "SCHEMA",
    "episode_names",
    "natural_assessor_record",
    "natural_program_identities",
    "natural_public_record",
    "operation_aliases",
    "parse_program_surface",
    "render_feedback",
    "validate_natural_public_record",
]
