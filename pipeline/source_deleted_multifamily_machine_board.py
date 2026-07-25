"""Frozen multi-family board for source-deleted machine compilation.

Every episode presents a complete finite transition law using opaque local
state and action names.  A candidate must compile those records into one
anonymous machine before a late program is disclosed.  The same packet and
executor contract covers three genuinely different law generators:

* affine maps over a finite modular ring;
* bitwise rotate/xor maps; and
* unconstrained permutations.

This is a bounded systematic-transfer qualification, not a general-reasoning
claim by itself.  Its purpose is to prevent another single-family proxy from
being promoted as evidence of a reusable reasoning mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence


FAMILIES = ("affine_modular", "bitwise_rotate_xor", "permutation")
RENDERERS = ("fields", "prose", "tuple", "reverse")
TRAIN_RENDERERS = (0, 1, 2)
HELD_OUT_RENDERER = 3
ACTION_COUNT = 3


class MultiFamilyBoardError(ValueError):
    """Raised when an episode or sealed machine violates the board contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: object) -> str:
    return sha256(canonical_json(value).encode("ascii")).hexdigest()


def _opaque(prefix: str, rng: random.Random) -> str:
    return f"{prefix}_{rng.getrandbits(80):020x}"


def _rotate_left(value: int, shift: int, width: int) -> int:
    mask = (1 << width) - 1
    shift %= width
    return ((value << shift) | (value >> (width - shift))) & mask


def _affine_transitions(
    rng: random.Random,
    *,
    cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    actions: list[tuple[int, ...]] = []
    used: set[tuple[int, int]] = set()
    while len(actions) < ACTION_COUNT:
        multiplier = rng.randrange(1, cardinality)
        offset = rng.randrange(cardinality)
        if math.gcd(multiplier, cardinality) != 1:
            continue
        if (multiplier, offset) in used:
            continue
        used.add((multiplier, offset))
        actions.append(
            tuple(
                (multiplier * state + offset) % cardinality
                for state in range(cardinality)
            )
        )
    return tuple(actions)


def _bitwise_transitions(
    rng: random.Random,
    *,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    cardinality = 1 << width
    actions: list[tuple[int, ...]] = []
    used: set[tuple[int, int]] = set()
    while len(actions) < ACTION_COUNT:
        shift = rng.randrange(width)
        xor_mask = rng.randrange(1, cardinality)
        if (shift, xor_mask) in used:
            continue
        used.add((shift, xor_mask))
        actions.append(
            tuple(
                _rotate_left(state ^ xor_mask, shift, width)
                for state in range(cardinality)
            )
        )
    return tuple(actions)


def _permutation_transitions(
    rng: random.Random,
    *,
    cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    actions: list[tuple[int, ...]] = []
    while len(actions) < ACTION_COUNT:
        values = list(range(cardinality))
        rng.shuffle(values)
        candidate = tuple(values)
        if candidate not in actions:
            actions.append(candidate)
    return tuple(actions)


def _family_geometry(family: str, split: str) -> tuple[int, int | None]:
    if split not in {"train", "development"}:
        raise MultiFamilyBoardError("only train and development are available")
    if family == "affine_modular":
        cardinality = 8 if split == "train" else 16
        return cardinality, None
    if family == "bitwise_rotate_xor":
        width = 3 if split == "train" else 4
        return 1 << width, width
    if family == "permutation":
        cardinality = 8 if split == "train" else 16
        return cardinality, None
    raise MultiFamilyBoardError("unknown law family")


def _render_record(renderer: int, state: str, action: str, target: str) -> str:
    if renderer == 0:
        return f"from={state}; action={action}; to={target}"
    if renderer == 1:
        return f"When {action} acts on {state}, the result is {target}."
    if renderer == 2:
        return f"({action}|{state}|{target})"
    if renderer == 3:
        return f"{target} <= [{action}] {state}"
    raise MultiFamilyBoardError("renderer leaves the public orbit")


def _render_query(renderer: int, start: str, actions: Sequence[str]) -> str:
    word = ",".join(actions)
    if renderer == 0:
        return f"start={start}; program={word}"
    if renderer == 1:
        return f"Begin at {start}. Apply in order: {word}."
    if renderer == 2:
        return f"?({start}|{word})"
    if renderer == 3:
        return f"{word} @ {start} => ?"
    raise MultiFamilyBoardError("query renderer leaves the public orbit")


_RECORD_PATTERNS = (
    re.compile(r"from=(?P<s>\S+); action=(?P<a>\S+); to=(?P<t>\S+)\Z"),
    re.compile(
        r"When (?P<a>\S+) acts on (?P<s>\S+), the result is (?P<t>\S+)\.\Z"
    ),
    re.compile(r"\((?P<a>[^|]+)\|(?P<s>[^|]+)\|(?P<t>[^)]+)\)\Z"),
    re.compile(r"(?P<t>\S+) <= \[(?P<a>[^\]]+)\] (?P<s>\S+)\Z"),
)
_QUERY_PATTERNS = (
    re.compile(r"start=(?P<s>\S+); program=(?P<w>\S+)\Z"),
    re.compile(r"Begin at (?P<s>\S+)\. Apply in order: (?P<w>\S+)\.\Z"),
    re.compile(r"\?\((?P<s>[^|]+)\|(?P<w>[^)]+)\)\Z"),
    re.compile(r"(?P<w>\S+) @ (?P<s>\S+) => \?\Z"),
)


def _match_line(
    line: str,
    patterns: Sequence[re.Pattern[str]],
) -> Mapping[str, str]:
    matches = [match for pattern in patterns if (match := pattern.fullmatch(line))]
    if len(matches) != 1:
        raise MultiFamilyBoardError("rendered line is ambiguous or invalid")
    return matches[0].groupdict()


@dataclass(frozen=True, slots=True)
class CandidateEpisode:
    """Only bytes available to a candidate at compile and late-query time."""

    source: str
    query: str


@dataclass(frozen=True, slots=True)
class SupervisorEpisode:
    """Private labels and split metadata excluded from candidate execution."""

    family: str
    split: str
    cell: str
    renderer: int
    law_sha256: str
    source_sha256: str
    query_sha256: str
    answer: str
    composition_length: int
    episode_seed: int


@dataclass(frozen=True, slots=True)
class GeneratedEpisode:
    candidate: CandidateEpisode
    supervisor: SupervisorEpisode


@dataclass(frozen=True, slots=True)
class SealedTransitionMachine:
    """Source-free anonymous transition packet used by the late executor."""

    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    transition: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        cardinality = len(self.state_keys)
        if (
            cardinality < 2
            or len(set(self.state_keys)) != cardinality
            or len(self.action_keys) != ACTION_COUNT
            or len(set(self.action_keys)) != ACTION_COUNT
            or len(self.transition) != ACTION_COUNT
        ):
            raise MultiFamilyBoardError("sealed machine geometry differs")
        expected = set(range(cardinality))
        for row in self.transition:
            if len(row) != cardinality or set(row) != expected:
                raise MultiFamilyBoardError(
                    "sealed action is not a complete permutation"
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
) -> GeneratedEpisode:
    """Generate one deterministic episode without confirmation access."""

    if family not in FAMILIES:
        raise MultiFamilyBoardError("family leaves the frozen contract")
    if split not in {"train", "development"}:
        raise MultiFamilyBoardError("confirmation generation is unavailable")
    if renderer not in range(len(RENDERERS)):
        raise MultiFamilyBoardError("renderer leaves the frozen contract")
    if cell not in {"fit", "law", "composition", "renderer", "joint"}:
        raise MultiFamilyBoardError("development cell differs")
    if split == "train" and cell != "fit":
        raise MultiFamilyBoardError("train rows must belong to fit")
    if split == "train" and renderer not in TRAIN_RENDERERS:
        raise MultiFamilyBoardError("held-out renderer entered training")
    rng = random.Random(
        int.from_bytes(
            sha256(
                f"MFM-V1|{seed}|{split}|{family}|{cell}".encode("ascii")
            ).digest()[:8],
            "big",
        )
    )
    geometry_split = (
        "development" if split == "development" and cell in {"law", "joint"} else "train"
    )
    cardinality, width = _family_geometry(family, geometry_split)
    if family == "affine_modular":
        transition = _affine_transitions(rng, cardinality=cardinality)
    elif family == "bitwise_rotate_xor":
        if width is None:
            raise MultiFamilyBoardError("bitwise width is missing")
        transition = _bitwise_transitions(rng, width=width)
    else:
        transition = _permutation_transitions(rng, cardinality=cardinality)

    state_keys = tuple(_opaque("s", rng) for _ in range(cardinality))
    action_keys = tuple(_opaque("a", rng) for _ in range(ACTION_COUNT))
    records = [
        _render_record(
            renderer,
            state_keys[state],
            action_keys[action],
            state_keys[transition[action][state]],
        )
        for action in range(ACTION_COUNT)
        for state in range(cardinality)
    ]
    rng.shuffle(records)
    source = "\n".join(records)

    if split == "train":
        composition_length = rng.randint(1, 4)
    elif cell in {"composition", "joint"}:
        composition_length = rng.randint(5, 8)
    else:
        composition_length = rng.randint(1, 4)
    start_index = rng.randrange(cardinality)
    action_word = tuple(rng.randrange(ACTION_COUNT) for _ in range(composition_length))
    state = start_index
    for action in action_word:
        state = transition[action][state]
    query = _render_query(
        renderer,
        state_keys[start_index],
        tuple(action_keys[action] for action in action_word),
    )
    law_sha256 = sha256_json(
        {
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
            law_sha256=law_sha256,
            source_sha256=sha256(source.encode("ascii")).hexdigest(),
            query_sha256=sha256(query.encode("ascii")).hexdigest(),
            answer=state_keys[state],
            composition_length=composition_length,
            episode_seed=seed,
        ),
    )


def compile_source(source: str) -> SealedTransitionMachine:
    """Compile complete rendered records into an anonymous finite machine."""

    if not isinstance(source, str) or not source:
        raise MultiFamilyBoardError("candidate source is empty")
    triples: list[tuple[str, str, str]] = []
    for line in source.splitlines():
        fields = _match_line(line, _RECORD_PATTERNS)
        triples.append((fields["s"], fields["a"], fields["t"]))
    state_keys = tuple(sorted({state for state, _, _ in triples}))
    action_keys = tuple(sorted({action for _, action, _ in triples}))
    if len(action_keys) != ACTION_COUNT:
        raise MultiFamilyBoardError("source action count differs")
    state_to_index = {key: index for index, key in enumerate(state_keys)}
    action_to_index = {key: index for index, key in enumerate(action_keys)}
    rows = [[-1 for _ in state_keys] for _ in action_keys]
    for state, action, target in triples:
        if target not in state_to_index:
            raise MultiFamilyBoardError("transition target is undeclared")
        action_index = action_to_index[action]
        state_index = state_to_index[state]
        target_index = state_to_index[target]
        previous = rows[action_index][state_index]
        if previous not in {-1, target_index}:
            raise MultiFamilyBoardError("source contains a transition conflict")
        rows[action_index][state_index] = target_index
    if any(target < 0 for row in rows for target in row):
        raise MultiFamilyBoardError("source transition table is incomplete")
    return SealedTransitionMachine(
        state_keys=state_keys,
        action_keys=action_keys,
        transition=tuple(tuple(row) for row in rows),
    )


def decode_late_query(
    machine: SealedTransitionMachine,
    query: str,
) -> tuple[int, tuple[int, ...]]:
    """Decode a rendered late query into anonymous machine indices."""

    fields = _match_line(query, _QUERY_PATTERNS)
    try:
        state = machine.state_keys.index(fields["s"])
        actions = tuple(
            machine.action_keys.index(action)
            for action in fields["w"].split(",")
        )
    except ValueError as exc:
        raise MultiFamilyBoardError("late query references an unknown key") from exc
    if not actions:
        raise MultiFamilyBoardError("late query action word is empty")
    return state, actions


def execute_action_indices(
    machine: SealedTransitionMachine,
    start: int,
    actions: Sequence[int],
) -> str:
    """Execute anonymous action indices without source or renderer access."""

    if (
        not 0 <= start < len(machine.state_keys)
        or not actions
        or any(not 0 <= action < len(machine.action_keys) for action in actions)
    ):
        raise MultiFamilyBoardError("anonymous late program leaves machine support")
    state = start
    for action in actions:
        state = machine.transition[action][state]
    return machine.state_keys[state]


def execute_late_query(machine: SealedTransitionMachine, query: str) -> str:
    """Execute a rendered late program using only a sealed machine."""

    start, actions = decode_late_query(machine, query)
    return execute_action_indices(machine, start, actions)


def build_frozen_board(
    *,
    seed: int,
    train_per_renderer: int = 4,
    development_per_cell: int = 4,
) -> tuple[GeneratedEpisode, ...]:
    """Build the complete train/development board in canonical order."""

    if train_per_renderer < 1 or development_per_cell < 1:
        raise MultiFamilyBoardError("board cell count must be positive")
    rows: list[GeneratedEpisode] = []
    used_laws: set[str] = set()
    cursor = 0

    def append_unique(
        *,
        split: str,
        family: str,
        renderer: int,
        cell: str,
    ) -> None:
        nonlocal cursor
        for _attempt in range(100_000):
            row = generate_episode(
                seed=seed + cursor,
                split=split,
                family=family,
                renderer=renderer,
                cell=cell,
            )
            cursor += 1
            if row.supervisor.law_sha256 in used_laws:
                continue
            used_laws.add(row.supervisor.law_sha256)
            rows.append(row)
            return
        raise MultiFamilyBoardError("unique-law board construction exhausted")

    for family in FAMILIES:
        for renderer in TRAIN_RENDERERS:
            for _ in range(train_per_renderer):
                append_unique(
                    split="train",
                    family=family,
                    renderer=renderer,
                    cell="fit",
                )
        for cell in ("law", "composition", "renderer", "joint"):
            renderer = HELD_OUT_RENDERER if cell in {"renderer", "joint"} else 0
            for _ in range(development_per_cell):
                append_unique(
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell=cell,
                )
    return tuple(rows)


def family_holdout_folds() -> tuple[Mapping[str, object], ...]:
    """Return the preregistered leave-one-family-out folds."""

    return tuple(
        {
            "held_out_family": family,
            "fit_families": [other for other in FAMILIES if other != family],
        }
        for family in FAMILIES
    )


def iter_candidate_bytes(episode: CandidateEpisode) -> Iterable[bytes]:
    """Expose the complete candidate-visible byte boundary for audits."""

    yield episode.source.encode("ascii")
    yield episode.query.encode("ascii")


__all__ = [
    "ACTION_COUNT",
    "CandidateEpisode",
    "FAMILIES",
    "GeneratedEpisode",
    "HELD_OUT_RENDERER",
    "MultiFamilyBoardError",
    "RENDERERS",
    "SealedTransitionMachine",
    "SupervisorEpisode",
    "TRAIN_RENDERERS",
    "build_frozen_board",
    "canonical_json",
    "compile_source",
    "decode_late_query",
    "execute_action_indices",
    "execute_late_query",
    "family_holdout_folds",
    "generate_episode",
    "iter_candidate_bytes",
    "sha256_json",
]
