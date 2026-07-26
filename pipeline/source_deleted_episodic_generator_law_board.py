"""Source-deleted episodic generator-law reasoning board.

Each episode exposes two complete transition tables for opaque support
generators and inclusion-minimal observations for two opaque target actions.
The targets are compositions in the support-generated closure at depth at
most six. The exact compiler infers the support tables from completeness,
constructs only that episode-local closure, seals uniquely identified target
maps, and discards the source and support machinery before query execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
import re


TRAIN_FAMILIES = ("cyclic", "dihedral", "bitwise")
HELD_OUT_FAMILY = "random_permutation"
FAMILIES = (*TRAIN_FAMILIES, HELD_OUT_FAMILY)
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
DEVELOPMENT_CELLS = (
    "law",
    "composition",
    "renderer",
    "topology",
    "joint",
)
MAX_CLOSURE_DEPTH = 6


class EpisodicGeneratorLawBoardError(ValueError):
    """Raised when an episode leaves the generator-law contract."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: object) -> str:
    return sha256(canonical_json(value).encode("ascii")).hexdigest()


def _map_sha256(transition: Sequence[int]) -> str:
    return sha256(
        bytes((len(transition), *transition))
    ).hexdigest()


def _opaque(rng: random.Random) -> str:
    return f"h{rng.getrandbits(80):020x}"


def _validate_permutation(transition: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(target) for target in transition)
    if (
        len(result) not in {8, 16}
        or set(result) != set(range(len(result)))
    ):
        raise EpisodicGeneratorLawBoardError(
            "transition is not a supported permutation"
        )
    return result


def _apply_word(
    supports: Sequence[Sequence[int]],
    word: Sequence[int],
    state: int,
) -> int:
    for generator in word:
        state = supports[generator][state]
    return state


def compose_support_word(
    supports: Sequence[Sequence[int]],
    word: Sequence[int],
) -> tuple[int, ...]:
    """Return the permutation induced by a support-generator word."""

    normalized = tuple(_validate_permutation(row) for row in supports)
    if len(normalized) != 2 or any(
        generator not in {0, 1}
        for generator in word
    ):
        raise EpisodicGeneratorLawBoardError(
            "support word leaves the contract"
        )
    return tuple(
        _apply_word(normalized, word, state)
        for state in range(len(normalized[0]))
    )


@dataclass(frozen=True, slots=True)
class ClosureEntry:
    transition: tuple[int, ...]
    word: tuple[int, ...]


def build_episode_closure(
    supports: Sequence[Sequence[int]],
    *,
    max_depth: int = MAX_CLOSURE_DEPTH,
) -> tuple[ClosureEntry, ...]:
    """Enumerate unique maps and a shortest word in the local closure."""

    normalized = tuple(_validate_permutation(row) for row in supports)
    if (
        len(normalized) != 2
        or len({len(row) for row in normalized}) != 1
        or normalized[0] == normalized[1]
        or max_depth < 1
    ):
        raise EpisodicGeneratorLawBoardError(
            "support closure geometry differs"
        )
    cardinality = len(normalized[0])
    identity = tuple(range(cardinality))
    words: dict[tuple[int, ...], tuple[int, ...]] = {identity: ()}
    frontier = [identity]
    for _depth in range(1, max_depth + 1):
        next_frontier: list[tuple[int, ...]] = []
        for current in frontier:
            prefix = words[current]
            for generator, support in enumerate(normalized):
                candidate = tuple(
                    support[current[state]]
                    for state in range(cardinality)
                )
                if candidate in words:
                    continue
                words[candidate] = (*prefix, generator)
                next_frontier.append(candidate)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(
        ClosureEntry(transition=transition, word=word)
        for transition, word in sorted(
            words.items(),
            key=lambda item: (
                len(item[1]),
                item[0],
                item[1],
            ),
        )
    )


def _conjugate(
    transition: Sequence[int],
    labels: Sequence[int],
) -> tuple[int, ...]:
    cardinality = len(transition)
    result = [-1] * cardinality
    for abstract_source, abstract_target in enumerate(transition):
        result[labels[abstract_source]] = labels[abstract_target]
    return tuple(result)


def _rotate_left(value: int, shift: int, width: int) -> int:
    mask = (1 << width) - 1
    shift %= width
    return (
        (value << shift)
        | (value >> (width - shift))
    ) & mask


def _generate_supports(
    *,
    family: str,
    cardinality: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    labels = list(range(cardinality))
    rng.shuffle(labels)
    if family == "cyclic":
        first = tuple(
            (state + 1) % cardinality
            for state in range(cardinality)
        )
        second = tuple(
            (state - 1) % cardinality
            for state in range(cardinality)
        )
    elif family == "dihedral":
        first = tuple(
            (state + 1) % cardinality
            for state in range(cardinality)
        )
        reflection_offset = rng.randrange(cardinality)
        second = tuple(
            (reflection_offset - state) % cardinality
            for state in range(cardinality)
        )
    elif family == "bitwise":
        width = int(math.log2(cardinality))
        bit = 1 << rng.randrange(width)
        first = tuple(
            _rotate_left(state, 1, width)
            for state in range(cardinality)
        )
        second = tuple(
            state ^ bit
            for state in range(cardinality)
        )
    elif family == HELD_OUT_FAMILY:
        first_values = list(range(cardinality))
        second_values = list(range(cardinality))
        rng.shuffle(first_values)
        for _attempt in range(1_000):
            rng.shuffle(second_values)
            if second_values != first_values:
                break
        first = tuple(first_values)
        second = tuple(second_values)
    else:
        raise EpisodicGeneratorLawBoardError(
            "generator family leaves the contract"
        )
    return (
        _conjugate(first, labels),
        _conjugate(second, labels),
    )


def _consistent_maps(
    closure: Sequence[ClosureEntry],
    observations: Mapping[int, int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        entry.transition
        for entry in closure
        if all(
            entry.transition[source] == target
            for source, target in observations.items()
        )
    )


def _minimal_identifying_inputs(
    *,
    target: tuple[int, ...],
    closure: Sequence[ClosureEntry],
    rng: random.Random,
) -> tuple[int, ...]:
    cardinality = len(target)
    visible = set(range(cardinality))
    removal_order = list(range(cardinality))
    rng.shuffle(removal_order)
    for source in removal_order:
        trial = visible - {source}
        candidates = _consistent_maps(
            closure,
            {
                state: target[state]
                for state in trial
            },
        )
        if candidates == (target,):
            visible = trial
    observations = {
        state: target[state]
        for state in visible
    }
    if (
        not visible
        or len(visible) >= cardinality
        or _consistent_maps(closure, observations) != (target,)
    ):
        raise EpisodicGeneratorLawBoardError(
            "sparse target witness differs"
        )
    for source in visible:
        candidates = _consistent_maps(
            closure,
            {
                state: target[state]
                for state in visible - {source}
            },
        )
        if len(candidates) <= 1:
            raise EpisodicGeneratorLawBoardError(
                "target witness is not inclusion-minimal"
            )
    return tuple(sorted(visible))


def _maps_commute(
    first: Sequence[int],
    second: Sequence[int],
) -> bool:
    return all(
        first[second[state]] == second[first[state]]
        for state in range(len(first))
    )


def _execute_indices(
    transitions: Sequence[Sequence[int]],
    start: int,
    actions: Sequence[int],
) -> int:
    state = start
    for action in actions:
        state = transitions[action][state]
    return state


def _choose_hidden_query(
    *,
    transitions: tuple[tuple[int, ...], tuple[int, ...]],
    visible_inputs: tuple[tuple[int, ...], tuple[int, ...]],
    length_bounds: tuple[int, int],
    rng: random.Random,
) -> tuple[int, tuple[int, ...], int, bool]:
    cardinality = len(transitions[0])
    visible = tuple(set(row) for row in visible_inputs)
    order_sensitive = not _maps_commute(*transitions)
    for _attempt in range(100_000):
        length = rng.randint(*length_bounds)
        actions = tuple(
            rng.randrange(2)
            for _ in range(length)
        )
        if len(set(actions)) != 2:
            continue
        state = rng.randrange(cardinality)
        start = state
        if any(
            (
                state in visible[action],
                state := transitions[action][state],
            )[0]
            for action in actions
        ):
            continue
        answer = state
        swapped_answer = _execute_indices(
            tuple(reversed(transitions)),
            start,
            actions,
        )
        if swapped_answer == answer:
            continue
        if order_sensitive and (
            _execute_indices(
                transitions,
                start,
                tuple(reversed(actions)),
            )
            == answer
        ):
            continue
        return start, actions, answer, order_sensitive
    raise EpisodicGeneratorLawBoardError(
        "hidden causal query generation exhausted"
    )


def _render_header(renderer: int, cardinality: int) -> str:
    if renderer == 0:
        return f"domain-size={cardinality}"
    if renderer == 1:
        return f"There are {cardinality} states."
    if renderer == 2:
        return f"(domain|{cardinality})"
    if renderer == 3:
        return f"states=0..{cardinality - 1}"
    if renderer == 4:
        return f"maximum-state={cardinality - 1}"
    if renderer == 5:
        return f"The state domain contains {cardinality} values."
    raise EpisodicGeneratorLawBoardError(
        "renderer leaves the contract"
    )


def _render_record(
    renderer: int,
    source: int,
    action: str,
    target: int,
) -> str:
    if renderer == 0:
        return (
            f"origin={source}; operation={action}; "
            f"destination={target}"
        )
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
    raise EpisodicGeneratorLawBoardError(
        "renderer leaves the contract"
    )


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
        return (
            f"After {word}, the initial state was {start}; result?"
        )
    raise EpisodicGeneratorLawBoardError(
        "renderer leaves the contract"
    )


_HEADER_PATTERNS = (
    re.compile(r"domain-size=(?P<n>\d+)\Z"),
    re.compile(r"There are (?P<n>\d+) states\.\Z"),
    re.compile(r"\(domain\|(?P<n>\d+)\)\Z"),
    re.compile(r"states=0\.\.(?P<m>\d+)\Z"),
    re.compile(r"maximum-state=(?P<m>\d+)\Z"),
    re.compile(
        r"The state domain contains (?P<n>\d+) values\.\Z"
    ),
)
_RECORD_PATTERNS = (
    re.compile(
        r"origin=(?P<s>\d+); operation=(?P<a>\S+); "
        r"destination=(?P<t>\d+)\Z"
    ),
    re.compile(
        r"Applying (?P<a>\S+) to (?P<s>\d+) "
        r"produces (?P<t>\d+)\.\Z"
    ),
    re.compile(
        r"\((?P<a>[^|]+)\|(?P<s>\d+)\|(?P<t>\d+)\)\Z"
    ),
    re.compile(
        r"(?P<s>\d+) -\[(?P<a>[^\]]+)\]-> (?P<t>\d+)\Z"
    ),
    re.compile(
        r"(?P<t>\d+) <-\{(?P<a>[^}]+)\}- "
        r"\((?P<s>\d+)\)\Z"
    ),
    re.compile(
        r"(?P<t>\d+) is reached from (?P<s>\d+) "
        r"using (?P<a>\S+)\Z"
    ),
)
_QUERY_PATTERNS = (
    re.compile(r"origin=(?P<s>\d+); program=(?P<w>\S+)\Z"),
    re.compile(
        r"Begin at (?P<s>\d+)\. Apply: (?P<w>\S+)\.\Z"
    ),
    re.compile(r"\?\((?P<s>\d+)\|(?P<w>[^)]+)\)\Z"),
    re.compile(r"(?P<s>\d+) -\[(?P<w>[^\]]+)\]-> \?\Z"),
    re.compile(
        r"then (?P<w>\S+) beginning-from (?P<s>\d+)\Z"
    ),
    re.compile(
        r"After (?P<w>\S+), the initial state was "
        r"(?P<s>\d+); result\?\Z"
    ),
)


def _match(
    line: str,
    patterns: Sequence[re.Pattern[str]],
) -> Mapping[str, str]:
    matches = [
        match
        for pattern in patterns
        if (match := pattern.fullmatch(line))
    ]
    if len(matches) != 1:
        raise EpisodicGeneratorLawBoardError(
            "rendered line is invalid or ambiguous"
        )
    return matches[0].groupdict()


def _parse_cardinality(line: str) -> int:
    fields = _match(line, _HEADER_PATTERNS)
    cardinality = (
        int(fields["n"])
        if fields.get("n") is not None
        else int(fields["m"]) + 1
    )
    if cardinality not in {8, 16}:
        raise EpisodicGeneratorLawBoardError(
            "header cardinality differs"
        )
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
    closure_size: int
    max_closure_depth: int
    target_composition_lengths: tuple[int, int]
    target_composition_words: tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]
    query_composition_length: int
    target_visible_records: int
    target_complete_records: int
    support_complete_records: int
    support_transition: tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]
    target_transition: tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]
    target_visible_inputs: tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]
    support_law_sha256: tuple[str, str]
    target_map_sha256: tuple[str, str]
    target_law_sha256: tuple[str, str]
    law_sha256: str
    order_sensitive: bool
    answer: int
    episode_seed: int


@dataclass(frozen=True, slots=True)
class GeneratedEpisode:
    candidate: CandidateEpisode
    supervisor: SupervisorEpisode


@dataclass(frozen=True, slots=True)
class SealedEpisodicGeneratorMachine:
    cardinality: int
    target_keys: tuple[str, str]
    transition: tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]
    visible_inputs: tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]

    def __post_init__(self) -> None:
        expected = set(range(self.cardinality))
        if (
            self.cardinality not in {8, 16}
            or len(self.target_keys) != 2
            or len(set(self.target_keys)) != 2
            or len(self.transition) != 2
            or len(self.visible_inputs) != 2
            or any(
                len(row) != self.cardinality
                or set(row) != expected
                for row in self.transition
            )
            or any(
                not row
                or len(row) >= self.cardinality
                or len(set(row)) != len(row)
                or any(
                    not 0 <= source < self.cardinality
                    for source in row
                )
                for row in self.visible_inputs
            )
        ):
            raise EpisodicGeneratorLawBoardError(
                "sealed episodic machine differs"
            )

    @property
    def packet_sha256(self) -> str:
        return sha256(self.deployed_wire()).hexdigest()

    def deployed_wire(self) -> bytes:
        return canonical_json(
            {
                "cardinality": self.cardinality,
                "target_keys": self.target_keys,
                "transition": self.transition,
                "visible_inputs": self.visible_inputs,
            }
        ).encode("ascii")

    @classmethod
    def from_deployed_wire(
        cls,
        payload: bytes,
    ) -> SealedEpisodicGeneratorMachine:
        try:
            value = json.loads(payload)
            return cls(
                cardinality=int(value["cardinality"]),
                target_keys=tuple(value["target_keys"]),
                transition=tuple(
                    tuple(int(target) for target in row)
                    for row in value["transition"]
                ),
                visible_inputs=tuple(
                    tuple(int(source) for source in row)
                    for row in value["visible_inputs"]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise EpisodicGeneratorLawBoardError(
                "deployed episodic packet differs"
            ) from exc


def _target_depth_bounds(cell: str) -> tuple[int, int]:
    return (3, MAX_CLOSURE_DEPTH) if cell in {
        "composition",
        "joint",
    } else (2, 2)


def _query_length_bounds(cell: str) -> tuple[int, int]:
    return (5, 6) if cell in {
        "composition",
        "joint",
    } else (3, 4)


def generate_episode(
    *,
    seed: int,
    split: str,
    family: str,
    renderer: int,
    cell: str,
    cardinality: int,
) -> GeneratedEpisode:
    if family not in FAMILIES:
        raise EpisodicGeneratorLawBoardError(
            "family leaves the contract"
        )
    if split not in {"train", "development"}:
        raise EpisodicGeneratorLawBoardError(
            "split leaves the contract"
        )
    if renderer not in range(len(RENDERERS)):
        raise EpisodicGeneratorLawBoardError(
            "renderer leaves the contract"
        )
    if split == "train" and (
        family not in TRAIN_FAMILIES
        or cell != "fit"
        or renderer not in TRAIN_RENDERERS
        or cardinality != 8
    ):
        raise EpisodicGeneratorLawBoardError(
            "train row leaves fit support"
        )
    if (
        split == "development"
        and cell not in DEVELOPMENT_CELLS
    ):
        raise EpisodicGeneratorLawBoardError(
            "development cell differs"
        )
    if cardinality not in {8, 16}:
        raise EpisodicGeneratorLawBoardError(
            "topology leaves the contract"
        )
    rng = random.Random(
        int.from_bytes(
            sha256(
                (
                    f"SEGL-V1|{seed}|{split}|{family}|{cell}|"
                    f"{cardinality}"
                ).encode("ascii")
            ).digest()[:8],
            "big",
        )
    )
    depth_min, depth_max = _target_depth_bounds(cell)
    for _support_attempt in range(10_000):
        raw_supports = _generate_supports(
            family=family,
            cardinality=cardinality,
            rng=rng,
        )
        raw_support_keys = (_opaque(rng), _opaque(rng))
        support_pairs = sorted(
            zip(
                raw_support_keys,
                raw_supports,
                strict=True,
            ),
            key=lambda item: item[0],
        )
        support_keys = tuple(
            key for key, _transition in support_pairs
        )
        supports = tuple(
            transition for _key, transition in support_pairs
        )
        closure = build_episode_closure(supports)
        identity = tuple(range(cardinality))
        candidates = [
            entry
            for entry in closure
            if (
                depth_min <= len(entry.word) <= depth_max
                and entry.transition != identity
                and entry.transition not in supports
            )
        ]
        rng.shuffle(candidates)
        prepared: list[
            tuple[
                ClosureEntry,
                tuple[int, ...],
            ]
        ] = []
        for entry in candidates:
            try:
                visible = _minimal_identifying_inputs(
                    target=entry.transition,
                    closure=closure,
                    rng=rng,
                )
            except EpisodicGeneratorLawBoardError:
                continue
            prepared.append((entry, visible))
            if len(prepared) >= 12:
                break
        found: tuple[
            tuple[ClosureEntry, tuple[int, ...]],
            tuple[ClosureEntry, tuple[int, ...]],
        ] | None = None
        for first_index, first in enumerate(prepared):
            for second in prepared[first_index + 1:]:
                if first[0].transition == second[0].transition:
                    continue
                found = (first, second)
                break
            if found is not None:
                break
        if found is None:
            continue
        raw_target_keys = (_opaque(rng), _opaque(rng))
        target_pairs = sorted(
            zip(
                raw_target_keys,
                found,
                strict=True,
            ),
            key=lambda item: item[0],
        )
        target_keys = tuple(
            key for key, _item in target_pairs
        )
        target_entries = tuple(
            item[0] for _key, item in target_pairs
        )
        visible_inputs = tuple(
            item[1] for _key, item in target_pairs
        )
        transitions = tuple(
            entry.transition for entry in target_entries
        )
        try:
            (
                start,
                query_actions,
                answer,
                order_sensitive,
            ) = _choose_hidden_query(
                transitions=transitions,
                visible_inputs=visible_inputs,
                length_bounds=_query_length_bounds(cell),
                rng=rng,
            )
        except EpisodicGeneratorLawBoardError:
            continue
        break
    else:
        raise EpisodicGeneratorLawBoardError(
            "episode generator exhausted"
        )

    records = [
        _render_record(
            renderer,
            source,
            support_keys[action],
            supports[action][source],
        )
        for action in range(2)
        for source in range(cardinality)
    ]
    records.extend(
        _render_record(
            renderer,
            source,
            target_keys[action],
            transitions[action][source],
        )
        for action in range(2)
        for source in visible_inputs[action]
    )
    rng.shuffle(records)
    source = "\n".join(
        [_render_header(renderer, cardinality), *records]
    )
    target_law_sha256 = tuple(
        sha256_json(
            {
                "cardinality": cardinality,
                "supports": supports,
                "target": transition,
            }
        )
        for transition in transitions
    )
    return GeneratedEpisode(
        candidate=CandidateEpisode(
            source=source,
            query=_render_query(
                renderer,
                start,
                tuple(
                    target_keys[action]
                    for action in query_actions
                ),
            ),
        ),
        supervisor=SupervisorEpisode(
            family=family,
            split=split,
            cell=cell,
            renderer=renderer,
            cardinality=cardinality,
            closure_size=len(closure),
            max_closure_depth=MAX_CLOSURE_DEPTH,
            target_composition_lengths=tuple(
                len(entry.word)
                for entry in target_entries
            ),
            target_composition_words=tuple(
                entry.word
                for entry in target_entries
            ),
            query_composition_length=len(query_actions),
            target_visible_records=sum(
                len(row) for row in visible_inputs
            ),
            target_complete_records=2 * cardinality,
            support_complete_records=2 * cardinality,
            support_transition=supports,
            target_transition=transitions,
            target_visible_inputs=visible_inputs,
            support_law_sha256=tuple(
                _map_sha256(transition)
                for transition in supports
            ),
            target_map_sha256=tuple(
                _map_sha256(transition)
                for transition in transitions
            ),
            target_law_sha256=target_law_sha256,
            law_sha256=sha256_json(
                {
                    "cardinality": cardinality,
                    "supports": supports,
                    "targets": transitions,
                }
            ),
            order_sensitive=order_sensitive,
            answer=answer,
            episode_seed=seed,
        ),
    )


def compile_source(
    source: str,
) -> SealedEpisodicGeneratorMachine:
    lines = source.splitlines()
    if len(lines) < 2:
        raise EpisodicGeneratorLawBoardError(
            "episodic source is empty"
        )
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
            raise EpisodicGeneratorLawBoardError(
                "record state leaves domain"
            )
        previous = observations.setdefault(action, {}).get(
            source_state
        )
        if previous not in {None, target_state}:
            raise EpisodicGeneratorLawBoardError(
                "episodic transition conflict"
            )
        observations[action][source_state] = target_state
    if len(observations) != 4:
        raise EpisodicGeneratorLawBoardError(
            "episodic action count differs"
        )
    support_keys = tuple(
        sorted(
            action
            for action, row in observations.items()
            if len(row) == cardinality
        )
    )
    target_keys = tuple(
        sorted(
            action
            for action, row in observations.items()
            if len(row) < cardinality
        )
    )
    if len(support_keys) != 2 or len(target_keys) != 2:
        raise EpisodicGeneratorLawBoardError(
            "support or target role is not identifiable"
        )
    supports = tuple(
        _validate_permutation(
            tuple(
                observations[action][source]
                for source in range(cardinality)
            )
        )
        for action in support_keys
    )
    closure = build_episode_closure(supports)
    transitions: list[tuple[int, ...]] = []
    visible_inputs: list[tuple[int, ...]] = []
    for action in target_keys:
        if not observations[action]:
            raise EpisodicGeneratorLawBoardError(
                "target observation set is empty"
            )
        candidates = _consistent_maps(
            closure,
            observations[action],
        )
        if len(candidates) != 1:
            raise EpisodicGeneratorLawBoardError(
                "target law is not uniquely identifiable"
            )
        transitions.append(candidates[0])
        visible_inputs.append(
            tuple(sorted(observations[action]))
        )
    return SealedEpisodicGeneratorMachine(
        cardinality=cardinality,
        target_keys=target_keys,
        transition=tuple(transitions),
        visible_inputs=tuple(visible_inputs),
    )


def decode_query(
    machine: SealedEpisodicGeneratorMachine,
    query: str,
) -> tuple[int, tuple[int, ...]]:
    fields = _match(query, _QUERY_PATTERNS)
    start = int(fields["s"])
    try:
        actions = tuple(
            machine.target_keys.index(action)
            for action in fields["w"].split(",")
        )
    except ValueError as exc:
        raise EpisodicGeneratorLawBoardError(
            "query action leaves sealed targets"
        ) from exc
    if (
        not 0 <= start < machine.cardinality
        or not actions
    ):
        raise EpisodicGeneratorLawBoardError(
            "query leaves episodic machine"
        )
    return start, actions


def execute_query(
    machine: SealedEpisodicGeneratorMachine,
    query: str,
) -> int:
    state, actions = decode_query(machine, query)
    return _execute_indices(machine.transition, state, actions)


def _development_specs(
    *,
    cell: str,
) -> tuple[tuple[str, int, int], ...]:
    if cell == "law":
        return ((HELD_OUT_FAMILY, 0, 8),)
    if cell == "composition":
        return tuple(
            (family, 0, 8)
            for family in TRAIN_FAMILIES
        )
    if cell == "renderer":
        return tuple(
            (family, HELD_OUT_RENDERER, 8)
            for family in TRAIN_FAMILIES
        )
    if cell == "topology":
        return tuple(
            (family, 0, 16)
            for family in TRAIN_FAMILIES
        )
    if cell == "joint":
        return (
            (
                HELD_OUT_FAMILY,
                HELD_OUT_RENDERER,
                16,
            ),
        )
    raise EpisodicGeneratorLawBoardError(
        "development cell differs"
    )


def build_frozen_board(
    *,
    seed: int,
    train_per_renderer: int = 1,
    development_per_cell: int = 1,
) -> tuple[GeneratedEpisode, ...]:
    if (
        train_per_renderer < 1
        or development_per_cell < 1
    ):
        raise EpisodicGeneratorLawBoardError(
            "board count is not positive"
        )
    rows: list[GeneratedEpisode] = []
    episode_laws: set[str] = set()
    target_laws: set[str] = set()
    target_maps: set[str] = set()
    cursor = 0

    def append_unique(
        *,
        split: str,
        family: str,
        renderer: int,
        cell: str,
        cardinality: int,
    ) -> None:
        nonlocal cursor
        for _attempt in range(100_000):
            row = generate_episode(
                seed=seed + cursor,
                split=split,
                family=family,
                renderer=renderer,
                cell=cell,
                cardinality=cardinality,
            )
            cursor += 1
            row_target_laws = set(
                row.supervisor.target_law_sha256
            )
            row_target_maps = set(
                row.supervisor.target_map_sha256
            )
            if (
                row.supervisor.law_sha256 in episode_laws
                or len(row_target_laws) != 2
                or row_target_laws & target_laws
                or len(row_target_maps) != 2
                or row_target_maps & target_maps
            ):
                continue
            episode_laws.add(row.supervisor.law_sha256)
            target_laws.update(row_target_laws)
            target_maps.update(row_target_maps)
            rows.append(row)
            return
        raise EpisodicGeneratorLawBoardError(
            "unique target-law generation exhausted"
        )

    for family in TRAIN_FAMILIES:
        for renderer in TRAIN_RENDERERS:
            for _index in range(train_per_renderer):
                append_unique(
                    split="train",
                    family=family,
                    renderer=renderer,
                    cell="fit",
                    cardinality=8,
                )
    for cell in DEVELOPMENT_CELLS:
        for family, renderer, cardinality in _development_specs(
            cell=cell
        ):
            for _index in range(development_per_cell):
                append_unique(
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell=cell,
                    cardinality=cardinality,
                )
    return tuple(rows)


__all__ = [
    "ClosureEntry",
    "DEVELOPMENT_CELLS",
    "EpisodicGeneratorLawBoardError",
    "FAMILIES",
    "GeneratedEpisode",
    "HELD_OUT_FAMILY",
    "HELD_OUT_RENDERER",
    "MAX_CLOSURE_DEPTH",
    "RENDERERS",
    "SealedEpisodicGeneratorMachine",
    "TRAIN_FAMILIES",
    "TRAIN_RENDERERS",
    "build_episode_closure",
    "build_frozen_board",
    "compile_source",
    "compose_support_word",
    "decode_query",
    "execute_query",
    "generate_episode",
]
