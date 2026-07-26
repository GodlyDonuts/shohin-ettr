"""Variable-topology source-deleted machine reasoning board.

This successor board removes the fixed-incidence shortcut from the first
multi-family qualification. State cardinality and action count vary, and
dedicated collision cells satisfy ``cardinality == 2 * action_count``. In a
complete permutation table that makes every state and action key occur equally
often, so frequency cannot reveal semantic type.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
import re
from collections.abc import Mapping, Sequence


FAMILIES = ("affine_modular", "bitwise_rotate_xor", "permutation")
RENDERERS = (
    "fields",
    "prose",
    "tuple",
    "right_arrow",
    "left_arrow",
    "passive",
)
TRAIN_RENDERERS = (0, 1, 2, 3, 4)
HELD_OUT_RENDERER = 5
FIT_TOPOLOGIES = ((4, 3), (8, 3), (8, 5), (16, 3))
DEVELOPMENT_CELLS = (
    "law",
    "composition",
    "topology",
    "collision",
    "renderer",
    "joint",
)


class VariableTopologyBoardError(ValueError):
    """Raised when an episode leaves the variable-topology contract."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: object) -> str:
    return sha256(canonical_json(value).encode("ascii")).hexdigest()


def _opaque(rng: random.Random) -> str:
    return f"h{rng.getrandbits(80):020x}"


def _rotate_left(value: int, shift: int, width: int) -> int:
    mask = (1 << width) - 1
    shift %= width
    return ((value << shift) | (value >> (width - shift))) & mask


def _transitions(
    *,
    family: str,
    cardinality: int,
    action_count: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    actions: list[tuple[int, ...]] = []
    if family == "affine_modular":
        used: set[tuple[int, int]] = set()
        while len(actions) < action_count:
            multiplier = rng.randrange(1, cardinality)
            offset = rng.randrange(cardinality)
            if (
                math.gcd(multiplier, cardinality) != 1
                or (multiplier, offset) in used
            ):
                continue
            used.add((multiplier, offset))
            actions.append(
                tuple(
                    (multiplier * state + offset) % cardinality
                    for state in range(cardinality)
                )
            )
    elif family == "bitwise_rotate_xor":
        width = int(math.log2(cardinality))
        if 1 << width != cardinality:
            raise VariableTopologyBoardError(
                "bitwise cardinality is not a power of two"
            )
        used_bitwise: set[tuple[int, int]] = set()
        while len(actions) < action_count:
            shift = rng.randrange(width)
            xor_mask = rng.randrange(1, cardinality)
            if (shift, xor_mask) in used_bitwise:
                continue
            used_bitwise.add((shift, xor_mask))
            actions.append(
                tuple(
                    _rotate_left(state ^ xor_mask, shift, width)
                    for state in range(cardinality)
                )
            )
    elif family == "permutation":
        while len(actions) < action_count:
            values = list(range(cardinality))
            rng.shuffle(values)
            candidate = tuple(values)
            if candidate not in actions:
                actions.append(candidate)
    else:
        raise VariableTopologyBoardError("family leaves the contract")
    return tuple(actions)


def _render_record(
    renderer: int,
    source: str,
    action: str,
    target: str,
) -> str:
    if renderer == 0:
        return f"origin={source}; operation={action}; destination={target}"
    if renderer == 1:
        return f"Applying {action} to {source} produces {target}."
    if renderer == 2:
        return f"({action}|{source}|{target})"
    if renderer == 3:
        return f"{source} -[{action}]-> {target}"
    if renderer == 4:
        return f"{target} <-{{{action}}}- ({source})"
    if renderer == 5:
        return f"{target} is reached from {source} using {action}"
    raise VariableTopologyBoardError("renderer leaves the contract")


def _render_query(
    renderer: int,
    start: str,
    actions: Sequence[str],
) -> str:
    word = ",".join(actions)
    if renderer == 0:
        return f"origin={start}; program={word}"
    if renderer == 1:
        return f"Begin at {start}. Apply: {word}."
    if renderer == 2:
        return f"?({start}|{word})"
    if renderer == 3:
        return f"{start} -[{word}]-> ?"
    if renderer == 4:
        return f"then {word} beginning-from {start}"
    if renderer == 5:
        return f"After {word}, the initial state was {start}; result?"
    raise VariableTopologyBoardError("renderer leaves the contract")


_RECORD_PATTERNS = (
    re.compile(
        r"origin=(?P<s>\S+); operation=(?P<a>\S+); "
        r"destination=(?P<t>\S+)\Z"
    ),
    re.compile(
        r"Applying (?P<a>\S+) to (?P<s>\S+) produces (?P<t>\S+)\.\Z"
    ),
    re.compile(r"\((?P<a>[^|]+)\|(?P<s>[^|]+)\|(?P<t>[^)]+)\)\Z"),
    re.compile(r"(?P<s>\S+) -\[(?P<a>[^\]]+)\]-> (?P<t>\S+)\Z"),
    re.compile(r"(?P<t>\S+) <-\{(?P<a>[^}]+)\}- \((?P<s>[^)]+)\)\Z"),
    re.compile(
        r"(?P<t>\S+) is reached from (?P<s>\S+) using (?P<a>\S+)\Z"
    ),
)
_QUERY_PATTERNS = (
    re.compile(r"origin=(?P<s>\S+); program=(?P<w>\S+)\Z"),
    re.compile(r"Begin at (?P<s>\S+)\. Apply: (?P<w>\S+)\.\Z"),
    re.compile(r"\?\((?P<s>[^|]+)\|(?P<w>[^)]+)\)\Z"),
    re.compile(r"(?P<s>\S+) -\[(?P<w>[^\]]+)\]-> \?\Z"),
    re.compile(r"then (?P<w>\S+) beginning-from (?P<s>\S+)\Z"),
    re.compile(
        r"After (?P<w>\S+), the initial state was (?P<s>\S+); result\?\Z"
    ),
)


def _match(
    line: str,
    patterns: Sequence[re.Pattern[str]],
) -> Mapping[str, str]:
    matches = [
        match for pattern in patterns if (match := pattern.fullmatch(line))
    ]
    if len(matches) != 1:
        raise VariableTopologyBoardError("rendered line is invalid or ambiguous")
    return matches[0].groupdict()


@dataclass(frozen=True, slots=True)
class CandidateEpisode:
    source: str
    query: str


@dataclass(frozen=True, slots=True)
class SupervisorEpisode:
    family: str
    split: str
    cell: str
    renderer: int
    cardinality: int
    action_count: int
    incidence_collision: bool
    composition_length: int
    law_sha256: str
    answer: str
    episode_seed: int


@dataclass(frozen=True, slots=True)
class GeneratedEpisode:
    candidate: CandidateEpisode
    supervisor: SupervisorEpisode


@dataclass(frozen=True, slots=True)
class SealedVariableMachine:
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    transition: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        cardinality = len(self.state_keys)
        action_count = len(self.action_keys)
        if (
            cardinality not in {4, 8, 16}
            or not 2 <= action_count <= 5
            or len(set(self.state_keys)) != cardinality
            or len(set(self.action_keys)) != action_count
            or set(self.state_keys) & set(self.action_keys)
            or len(self.transition) != action_count
        ):
            raise VariableTopologyBoardError("sealed geometry differs")
        expected = set(range(cardinality))
        if any(
            len(row) != cardinality or set(row) != expected
            for row in self.transition
        ):
            raise VariableTopologyBoardError(
                "sealed transition is not a permutation"
            )

    @property
    def packet_sha256(self) -> str:
        return sha256_json(
            {
                "action_keys": self.action_keys,
                "state_keys": self.state_keys,
                "transition": self.transition,
            }
        )


def generate_episode(
    *,
    seed: int,
    split: str,
    family: str,
    renderer: int,
    cell: str,
    cardinality: int,
    action_count: int,
) -> GeneratedEpisode:
    if family not in FAMILIES:
        raise VariableTopologyBoardError("family leaves the contract")
    if split not in {"train", "development"}:
        raise VariableTopologyBoardError("split leaves the contract")
    if renderer not in range(len(RENDERERS)):
        raise VariableTopologyBoardError("renderer leaves the contract")
    if split == "train" and (
        cell != "fit" or renderer not in TRAIN_RENDERERS
    ):
        raise VariableTopologyBoardError("train row leaves fit support")
    if split == "development" and cell not in DEVELOPMENT_CELLS:
        raise VariableTopologyBoardError("development cell differs")
    if (
        cardinality not in {4, 8, 16}
        or not 2 <= action_count <= 5
    ):
        raise VariableTopologyBoardError("topology leaves the contract")
    rng = random.Random(
        int.from_bytes(
            sha256(
                (
                    f"VTM-V1|{seed}|{split}|{family}|{cell}|"
                    f"{cardinality}|{action_count}"
                ).encode("ascii")
            ).digest()[:8],
            "big",
        )
    )
    transition = _transitions(
        family=family,
        cardinality=cardinality,
        action_count=action_count,
        rng=rng,
    )
    state_keys = tuple(_opaque(rng) for _ in range(cardinality))
    action_keys = tuple(_opaque(rng) for _ in range(action_count))
    records = [
        _render_record(
            renderer,
            state_keys[state],
            action_keys[action],
            state_keys[transition[action][state]],
        )
        for action in range(action_count)
        for state in range(cardinality)
    ]
    rng.shuffle(records)
    source = "\n".join(records)
    composition_length = (
        rng.randint(5, 8)
        if cell in {"composition", "joint"}
        else rng.randint(1, 4)
    )
    start = rng.randrange(cardinality)
    word = tuple(rng.randrange(action_count) for _ in range(composition_length))
    state = start
    for action in word:
        state = transition[action][state]
    query = _render_query(
        renderer,
        state_keys[start],
        tuple(action_keys[action] for action in word),
    )
    law_sha256 = sha256_json(
        {
            "action_count": action_count,
            "cardinality": cardinality,
            "transition": transition,
        }
    )
    return GeneratedEpisode(
        candidate=CandidateEpisode(source=source, query=query),
        supervisor=SupervisorEpisode(
            family=family,
            split=split,
            cell=cell,
            renderer=renderer,
            cardinality=cardinality,
            action_count=action_count,
            incidence_collision=cardinality == 2 * action_count,
            composition_length=composition_length,
            law_sha256=law_sha256,
            answer=state_keys[state],
            episode_seed=seed,
        ),
    )


def compile_source(source: str) -> SealedVariableMachine:
    if not source:
        raise VariableTopologyBoardError("source is empty")
    triples = [
        (
            fields["s"],
            fields["a"],
            fields["t"],
        )
        for line in source.splitlines()
        for fields in (_match(line, _RECORD_PATTERNS),)
    ]
    # Incidence collisions are intentionally ambiguous. Exact parsing uses
    # renderer grammar; candidates may not call this function.
    grammar_actions = {action for _, action, _ in triples}
    action_keys = tuple(sorted(grammar_actions))
    state_keys = tuple(
        sorted({key for source_key, _, target in triples for key in (source_key, target)})
    )
    state_to_index = {key: index for index, key in enumerate(state_keys)}
    action_to_index = {key: index for index, key in enumerate(action_keys)}
    rows = [[-1] * len(state_keys) for _ in action_keys]
    for source_key, action, target in triples:
        location = (
            action_to_index[action],
            state_to_index[source_key],
        )
        target_index = state_to_index[target]
        if rows[location[0]][location[1]] not in {-1, target_index}:
            raise VariableTopologyBoardError("transition conflict")
        rows[location[0]][location[1]] = target_index
    if any(target < 0 for row in rows for target in row):
        raise VariableTopologyBoardError("transition table is incomplete")
    return SealedVariableMachine(
        state_keys=state_keys,
        action_keys=action_keys,
        transition=tuple(tuple(row) for row in rows),
    )


def decode_query(
    machine: SealedVariableMachine,
    query: str,
) -> tuple[int, tuple[int, ...]]:
    fields = _match(query, _QUERY_PATTERNS)
    try:
        start = machine.state_keys.index(fields["s"])
        actions = tuple(
            machine.action_keys.index(action)
            for action in fields["w"].split(",")
        )
    except ValueError as exc:
        raise VariableTopologyBoardError("query key leaves machine") from exc
    if not actions:
        raise VariableTopologyBoardError("query word is empty")
    return start, actions


def execute_query(machine: SealedVariableMachine, query: str) -> str:
    state, actions = decode_query(machine, query)
    for action in actions:
        state = machine.transition[action][state]
    return machine.state_keys[state]


def _cell_topology(cell: str, index: int) -> tuple[int, int]:
    if cell in {"law", "composition", "renderer"}:
        return (8, 3)
    if cell == "topology":
        return (16, 5)
    if cell == "collision":
        return ((4, 2), (8, 4))[index % 2]
    if cell == "joint":
        return (8, 4)
    raise VariableTopologyBoardError("cell topology differs")


def build_frozen_board(
    *,
    seed: int,
    train_per_renderer: int = 4,
    development_per_cell: int = 4,
) -> tuple[GeneratedEpisode, ...]:
    if train_per_renderer < 1 or development_per_cell < 1:
        raise VariableTopologyBoardError("board count is not positive")
    rows: list[GeneratedEpisode] = []
    laws: set[str] = set()
    cursor = 0

    def append_unique(
        *,
        split: str,
        family: str,
        renderer: int,
        cell: str,
        topology: tuple[int, int],
    ) -> None:
        nonlocal cursor
        for _attempt in range(100_000):
            row = generate_episode(
                seed=seed + cursor,
                split=split,
                family=family,
                renderer=renderer,
                cell=cell,
                cardinality=topology[0],
                action_count=topology[1],
            )
            cursor += 1
            if row.supervisor.law_sha256 in laws:
                continue
            laws.add(row.supervisor.law_sha256)
            rows.append(row)
            return
        raise VariableTopologyBoardError("unique-law generation exhausted")

    for family in FAMILIES:
        for renderer in TRAIN_RENDERERS:
            for index in range(train_per_renderer):
                append_unique(
                    split="train",
                    family=family,
                    renderer=renderer,
                    cell="fit",
                    topology=FIT_TOPOLOGIES[index % len(FIT_TOPOLOGIES)],
                )
        for cell in DEVELOPMENT_CELLS:
            renderer = (
                HELD_OUT_RENDERER
                if cell in {"renderer", "joint"}
                else 0
            )
            for index in range(development_per_cell):
                append_unique(
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell=cell,
                    topology=_cell_topology(cell, index),
                )
    return tuple(rows)


def family_holdout_folds() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "fit_families": [
                other for other in FAMILIES if other != family
            ],
            "held_out_family": family,
        }
        for family in FAMILIES
    )


__all__ = [
    "DEVELOPMENT_CELLS",
    "FAMILIES",
    "FIT_TOPOLOGIES",
    "GeneratedEpisode",
    "HELD_OUT_RENDERER",
    "RENDERERS",
    "SealedVariableMachine",
    "TRAIN_RENDERERS",
    "VariableTopologyBoardError",
    "build_frozen_board",
    "compile_source",
    "decode_query",
    "execute_query",
    "family_holdout_folds",
    "generate_episode",
]
