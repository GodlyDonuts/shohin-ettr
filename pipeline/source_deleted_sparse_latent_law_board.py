"""Source-deleted sparse latent-law induction board.

Unlike the complete-table variable-topology board, this board reveals only an
inclusion-minimal set of transition demonstrations for each opaque action.
The demonstrations uniquely identify one operator in a finite union of three
structured law families. Every query step uses a transition that was absent
from the source, so exact execution requires latent-law completion rather than
record lookup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import random
import re


FAMILIES = (
    "affine_modular",
    "bitwise_rotate_xor",
    "gray_conjugate_affine",
)
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
FIT_TOPOLOGIES = ((8, 2), (8, 3), (16, 2), (16, 3))
DEVELOPMENT_CELLS = (
    "law",
    "composition",
    "topology",
    "renderer",
    "joint",
)


class SparseLatentLawBoardError(ValueError):
    """Raised when an episode leaves the sparse-law contract."""


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


def _gray(value: int) -> int:
    return value ^ (value >> 1)


def _inverse_gray(value: int) -> int:
    result = value
    while value:
        value >>= 1
        result ^= value
    return result


@lru_cache(maxsize=None)
def family_hypotheses(
    family: str,
    cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate distinct lawful permutations for one public geometry."""

    if family not in FAMILIES or cardinality not in {8, 16}:
        raise SparseLatentLawBoardError("hypothesis geometry differs")
    maps: set[tuple[int, ...]] = set()
    if family == "affine_modular":
        maps.update(
            tuple(
                (multiplier * state + offset) % cardinality
                for state in range(cardinality)
            )
            for multiplier in range(1, cardinality)
            if math.gcd(multiplier, cardinality) == 1
            for offset in range(cardinality)
        )
    elif family == "bitwise_rotate_xor":
        width = int(math.log2(cardinality))
        maps.update(
            tuple(
                _rotate_left(state ^ xor_mask, shift, width)
                for state in range(cardinality)
            )
            for shift in range(width)
            for xor_mask in range(1, cardinality)
        )
    else:
        maps.update(
            tuple(
                _inverse_gray(
                    (
                        multiplier * _gray(state)
                        + offset
                    )
                    % cardinality
                )
                for state in range(cardinality)
            )
            for multiplier in range(1, cardinality)
            if math.gcd(multiplier, cardinality) == 1
            for offset in range(cardinality)
        )
    return tuple(sorted(maps))


@lru_cache(maxsize=None)
def union_hypotheses(
    cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    if cardinality not in {8, 16}:
        raise SparseLatentLawBoardError("hypothesis cardinality differs")
    return tuple(
        sorted(
            {
                transition
                for family in FAMILIES
                for transition in family_hypotheses(family, cardinality)
            }
        )
    )


def _consistent_hypotheses(
    *,
    cardinality: int,
    observations: Mapping[int, int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        transition
        for transition in union_hypotheses(cardinality)
        if all(
            transition[source] == target
            for source, target in observations.items()
        )
    )


def hypothesis_split(transition: tuple[int, ...]) -> str:
    """Globally partition individual maps, including cross-family overlaps."""

    digest = sha256(bytes(transition)).digest()
    return "development" if digest[0] % 3 == 0 else "train"


def _minimal_identifying_inputs(
    transition: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    """Return an inclusion-minimal witness for one union-hypothesis map."""

    cardinality = len(transition)
    visible = set(range(cardinality))
    order = list(range(cardinality))
    rng.shuffle(order)
    for source in order:
        trial = visible - {source}
        candidates = _consistent_hypotheses(
            cardinality=cardinality,
            observations={
                state: transition[state]
                for state in trial
            },
        )
        if len(candidates) == 1:
            visible = trial
    if (
        not visible
        or len(visible) > cardinality // 2
        or _consistent_hypotheses(
            cardinality=cardinality,
            observations={
                state: transition[state]
                for state in visible
            },
        )
        != (transition,)
    ):
        raise SparseLatentLawBoardError(
            "minimal identifying witness differs"
        )
    for source in tuple(visible):
        candidates = _consistent_hypotheses(
            cardinality=cardinality,
            observations={
                state: transition[state]
                for state in visible - {source}
            },
        )
        if len(candidates) <= 1:
            raise SparseLatentLawBoardError("witness is not minimal")
    return tuple(sorted(visible))


def _render_header(renderer: int, cardinality: int) -> str:
    if renderer not in range(len(RENDERERS)):
        raise SparseLatentLawBoardError("renderer leaves the contract")
    return f"domain-size={cardinality}"


def _render_record(
    renderer: int,
    source: int,
    action: str,
    target: int,
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
    raise SparseLatentLawBoardError("renderer leaves the contract")


def _render_query(
    renderer: int,
    start: int,
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
    raise SparseLatentLawBoardError("renderer leaves the contract")


_HEADER_PATTERNS = (
    re.compile(r"domain-size=(?P<n>\d+)\Z"),
)
_RECORD_PATTERNS = (
    re.compile(
        r"origin=(?P<s>\d+); operation=(?P<a>\S+); "
        r"destination=(?P<t>\d+)\Z"
    ),
    re.compile(
        r"Applying (?P<a>\S+) to (?P<s>\d+) produces (?P<t>\d+)\.\Z"
    ),
    re.compile(r"\((?P<a>[^|]+)\|(?P<s>\d+)\|(?P<t>\d+)\)\Z"),
    re.compile(r"(?P<s>\d+) -\[(?P<a>[^\]]+)\]-> (?P<t>\d+)\Z"),
    re.compile(r"(?P<t>\d+) <-\{(?P<a>[^}]+)\}- \((?P<s>\d+)\)\Z"),
    re.compile(
        r"(?P<t>\d+) is reached from (?P<s>\d+) using (?P<a>\S+)\Z"
    ),
)
_QUERY_PATTERNS = (
    re.compile(r"origin=(?P<s>\d+); program=(?P<w>\S+)\Z"),
    re.compile(r"Begin at (?P<s>\d+)\. Apply: (?P<w>\S+)\.\Z"),
    re.compile(r"\?\((?P<s>\d+)\|(?P<w>[^)]+)\)\Z"),
    re.compile(r"(?P<s>\d+) -\[(?P<w>[^\]]+)\]-> \?\Z"),
    re.compile(r"then (?P<w>\S+) beginning-from (?P<s>\d+)\Z"),
    re.compile(
        r"After (?P<w>\S+), the initial state was (?P<s>\d+); "
        r"result\?\Z"
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
        raise SparseLatentLawBoardError(
            "rendered line is invalid or ambiguous"
        )
    return matches[0].groupdict()


def _parse_cardinality(line: str) -> int:
    fields = _match(line, _HEADER_PATTERNS)
    cardinality = int(fields["n"]) if fields.get("n") else int(fields["m"]) + 1
    if fields.get("m") is not None and int(fields["m"]) != cardinality - 1:
        raise SparseLatentLawBoardError("header range differs")
    if cardinality not in {8, 16}:
        raise SparseLatentLawBoardError("header cardinality differs")
    return cardinality


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
    composition_length: int
    visible_records: int
    complete_records: int
    action_law_sha256: tuple[str, ...]
    law_sha256: str
    answer: int
    episode_seed: int


@dataclass(frozen=True, slots=True)
class GeneratedEpisode:
    candidate: CandidateEpisode
    supervisor: SupervisorEpisode


@dataclass(frozen=True, slots=True)
class SealedSparseLawMachine:
    cardinality: int
    action_keys: tuple[str, ...]
    transition: tuple[tuple[int, ...], ...]
    visible_inputs: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        expected = set(range(self.cardinality))
        if (
            self.cardinality not in {8, 16}
            or not 2 <= len(self.action_keys) <= 4
            or len(set(self.action_keys)) != len(self.action_keys)
            or len(self.transition) != len(self.action_keys)
            or len(self.visible_inputs) != len(self.action_keys)
            or any(
                len(row) != self.cardinality or set(row) != expected
                for row in self.transition
            )
            or any(
                not visible
                or len(visible) > self.cardinality // 2
                or len(set(visible)) != len(visible)
                for visible in self.visible_inputs
            )
        ):
            raise SparseLatentLawBoardError("sealed sparse machine differs")

    @property
    def packet_sha256(self) -> str:
        return sha256_json(
            {
                "action_keys": self.action_keys,
                "cardinality": self.cardinality,
                "transition": self.transition,
                "visible_inputs": self.visible_inputs,
            }
        )

    def deployed_wire(self) -> bytes:
        return canonical_json(
            {
                "action_keys": self.action_keys,
                "cardinality": self.cardinality,
                "transition": self.transition,
                "visible_inputs": self.visible_inputs,
            }
        ).encode("ascii")

    @classmethod
    def from_deployed_wire(
        cls,
        payload: bytes,
    ) -> SealedSparseLawMachine:
        try:
            value = json.loads(payload)
            return cls(
                cardinality=int(value["cardinality"]),
                action_keys=tuple(value["action_keys"]),
                transition=tuple(
                    tuple(int(target) for target in row)
                    for row in value["transition"]
                ),
                visible_inputs=tuple(
                    tuple(int(source) for source in row)
                    for row in value["visible_inputs"]
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SparseLatentLawBoardError(
                "deployed sparse packet differs"
            ) from exc


def _choose_hidden_query(
    *,
    transitions: Sequence[Sequence[int]],
    visible_inputs: Sequence[set[int]],
    length: int,
    rng: random.Random,
) -> tuple[int, tuple[int, ...], int]:
    cardinality = len(transitions[0])
    action_count = len(transitions)
    for _attempt in range(100_000):
        start = rng.randrange(cardinality)
        actions = tuple(
            rng.randrange(action_count)
            for _ in range(length)
        )
        state = start
        hidden = True
        for action in actions:
            hidden &= state not in visible_inputs[action]
            state = transitions[action][state]
        if hidden:
            return start, actions, state
    raise SparseLatentLawBoardError("hidden query generation exhausted")


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
        raise SparseLatentLawBoardError("family leaves the contract")
    if split not in {"train", "development"}:
        raise SparseLatentLawBoardError("split leaves the contract")
    if renderer not in range(len(RENDERERS)):
        raise SparseLatentLawBoardError("renderer leaves the contract")
    if split == "train" and (
        cell != "fit" or renderer not in TRAIN_RENDERERS
    ):
        raise SparseLatentLawBoardError("train row leaves fit support")
    if split == "development" and cell not in DEVELOPMENT_CELLS:
        raise SparseLatentLawBoardError("development cell differs")
    if cardinality not in {8, 16} or not 2 <= action_count <= 4:
        raise SparseLatentLawBoardError("topology leaves the contract")
    rng = random.Random(
        int.from_bytes(
            sha256(
                (
                    f"SLL-V1|{seed}|{split}|{family}|{cell}|"
                    f"{cardinality}|{action_count}"
                ).encode("ascii")
            ).digest()[:8],
            "big",
        )
    )
    wanted_map_split = "train" if split == "train" else "development"
    family_maps = [
        transition
        for transition in family_hypotheses(family, cardinality)
        if hypothesis_split(transition) == wanted_map_split
    ]
    rng.shuffle(family_maps)
    if len(family_maps) < action_count:
        raise SparseLatentLawBoardError(
            "map split does not contain enough actions"
        )
    transitions = tuple(family_maps[:action_count])
    if len(set(transitions)) != action_count:
        raise SparseLatentLawBoardError("action laws are not distinct")
    action_keys = tuple(_opaque(rng) for _ in range(action_count))
    visible_inputs = tuple(
        _minimal_identifying_inputs(transition, rng)
        for transition in transitions
    )
    records = [
        _render_record(
            renderer,
            source,
            action_keys[action],
            transitions[action][source],
        )
        for action in range(action_count)
        for source in visible_inputs[action]
    ]
    rng.shuffle(records)
    source = "\n".join([_render_header(renderer, cardinality), *records])
    composition_length = (
        rng.randint(5, 8)
        if cell in {"composition", "joint"}
        else rng.randint(1, 4)
    )
    start, actions, answer = _choose_hidden_query(
        transitions=transitions,
        visible_inputs=tuple(set(row) for row in visible_inputs),
        length=composition_length,
        rng=rng,
    )
    return GeneratedEpisode(
        candidate=CandidateEpisode(
            source=source,
            query=_render_query(
                renderer,
                start,
                tuple(action_keys[action] for action in actions),
            ),
        ),
        supervisor=SupervisorEpisode(
            family=family,
            split=split,
            cell=cell,
            renderer=renderer,
            cardinality=cardinality,
            action_count=action_count,
            composition_length=composition_length,
            visible_records=len(records),
            complete_records=cardinality * action_count,
            action_law_sha256=tuple(
                sha256(bytes(transition)).hexdigest()
                for transition in transitions
            ),
            law_sha256=sha256_json(
                {
                    "action_count": action_count,
                    "cardinality": cardinality,
                    "transition": transitions,
                }
            ),
            answer=answer,
            episode_seed=seed,
        ),
    )


def compile_source(source: str) -> SealedSparseLawMachine:
    lines = source.splitlines()
    if len(lines) < 2:
        raise SparseLatentLawBoardError("sparse source is empty")
    cardinality = _parse_cardinality(lines[0])
    observations: dict[str, dict[int, int]] = {}
    for line in lines[1:]:
        fields = _match(line, _RECORD_PATTERNS)
        source_state = int(fields["s"])
        target_state = int(fields["t"])
        action = fields["a"]
        if (
            not 0 <= source_state < cardinality
            or not 0 <= target_state < cardinality
        ):
            raise SparseLatentLawBoardError("record state leaves domain")
        previous = observations.setdefault(action, {}).get(source_state)
        if previous not in {None, target_state}:
            raise SparseLatentLawBoardError("sparse transition conflict")
        observations[action][source_state] = target_state
    action_keys = tuple(sorted(observations))
    if not 2 <= len(action_keys) <= 4:
        raise SparseLatentLawBoardError("sparse action count differs")
    transitions: list[tuple[int, ...]] = []
    visible_inputs: list[tuple[int, ...]] = []
    for action in action_keys:
        candidates = _consistent_hypotheses(
            cardinality=cardinality,
            observations=observations[action],
        )
        if len(candidates) != 1:
            raise SparseLatentLawBoardError(
                "sparse law is not uniquely identifiable"
            )
        transitions.append(candidates[0])
        visible_inputs.append(tuple(sorted(observations[action])))
    return SealedSparseLawMachine(
        cardinality=cardinality,
        action_keys=action_keys,
        transition=tuple(transitions),
        visible_inputs=tuple(visible_inputs),
    )


def decode_query(
    machine: SealedSparseLawMachine,
    query: str,
) -> tuple[int, tuple[int, ...]]:
    fields = _match(query, _QUERY_PATTERNS)
    start = int(fields["s"])
    try:
        actions = tuple(
            machine.action_keys.index(action)
            for action in fields["w"].split(",")
        )
    except ValueError as exc:
        raise SparseLatentLawBoardError(
            "query action leaves machine"
        ) from exc
    if not 0 <= start < machine.cardinality or not actions:
        raise SparseLatentLawBoardError("query leaves sparse machine")
    return start, actions


def execute_query(
    machine: SealedSparseLawMachine,
    query: str,
) -> int:
    state, actions = decode_query(machine, query)
    for action in actions:
        state = machine.transition[action][state]
    return state


def _cell_topology(cell: str) -> tuple[int, int]:
    if cell in {"law", "composition", "renderer"}:
        return (8, 3)
    if cell in {"topology", "joint"}:
        return (16, 4)
    raise SparseLatentLawBoardError("cell topology differs")


def build_frozen_board(
    *,
    seed: int,
    train_per_renderer: int = 4,
    development_per_cell: int = 4,
) -> tuple[GeneratedEpisode, ...]:
    if train_per_renderer < 1 or development_per_cell < 1:
        raise SparseLatentLawBoardError("board count is not positive")
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
        raise SparseLatentLawBoardError("unique-law generation exhausted")

    for family in FAMILIES:
        for renderer in TRAIN_RENDERERS:
            for index in range(train_per_renderer):
                append_unique(
                    split="train",
                    family=family,
                    renderer=renderer,
                    cell="fit",
                    topology=FIT_TOPOLOGIES[
                        index % len(FIT_TOPOLOGIES)
                    ],
                )
        for cell in DEVELOPMENT_CELLS:
            renderer = (
                HELD_OUT_RENDERER
                if cell in {"renderer", "joint"}
                else 0
            )
            for _index in range(development_per_cell):
                append_unique(
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell=cell,
                    topology=_cell_topology(cell),
                )
    return tuple(rows)


__all__ = [
    "DEVELOPMENT_CELLS",
    "FAMILIES",
    "FIT_TOPOLOGIES",
    "GeneratedEpisode",
    "HELD_OUT_RENDERER",
    "RENDERERS",
    "SealedSparseLawMachine",
    "SparseLatentLawBoardError",
    "TRAIN_RENDERERS",
    "build_frozen_board",
    "compile_source",
    "decode_query",
    "execute_query",
    "family_hypotheses",
    "generate_episode",
    "hypothesis_split",
    "union_hypotheses",
]
