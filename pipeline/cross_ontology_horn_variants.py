"""Exact structural and semantic variants for the Horn qualification board.

This module is an offline generator and auditor.  Candidate processes receive
only ``HornVariantMaterialization.source`` during compilation and one
``HornVariantChallenge.source`` after source deletion.  Canonical alignments,
theory indices, and expected terminals are assessor-only records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
import random

from cross_ontology_horn_board import (
    Demonstration,
    GroundAtom,
    OBJECT_TYPES,
    PREDICATES,
    THEORIES,
    behavior_signature,
    challenge_initials,
    consistent_theories,
    execute_closure,
    identifying_evidence,
)


class HornVariantKind(StrEnum):
    """The seven preregistered Horn qualification variants, in order."""

    BASE = "base"
    ALPHA_REORDER = "alpha/reorder"
    ALIAS_SPLIT = "alias split"
    RELATION_REIFICATION = "relation reification"
    TYPE_TWIN = "type twin"
    EXECUTION_SEMANTICS_TWIN = "execution-semantics twin"
    AMBIGUITY_DELETED_TWIN = "ambiguity-deleted twin"


class HornExecutionSemantics(StrEnum):
    """Exact terminal-state semantics used by a materialized world."""

    PERSISTENT_CLOSURE = "persistent least-fixed-point closure"
    DERIVED_ONLY_CLOSURE = "derived-facts-only least-fixed-point closure"


class HornExpectationRelation(StrEnum):
    """Required relationship between a variant and its base world."""

    CANONICALLY_INVARIANT = "canonically invariant"
    EXECUTION_SEPARATE = "execution-semantics separate"
    IDENTIFIABILITY_SEPARATE = "identifiability separate"


@dataclass(frozen=True, slots=True)
class HornVariantExpectation:
    """Machine-checkable preregistered expectation for one variant."""

    relation_to_base: HornExpectationRelation
    evidence_invariant_after_alignment: bool
    challenge_outputs_invariant_after_alignment: bool
    requires_execution_separation: bool
    requires_version_space_separation: bool


VARIANT_ORDER = tuple(HornVariantKind)

VARIANT_EXPECTATIONS = {
    kind: HornVariantExpectation(
        relation_to_base=HornExpectationRelation.CANONICALLY_INVARIANT,
        evidence_invariant_after_alignment=True,
        challenge_outputs_invariant_after_alignment=True,
        requires_execution_separation=False,
        requires_version_space_separation=False,
    )
    for kind in VARIANT_ORDER[:5]
}
VARIANT_EXPECTATIONS.update(
    {
        HornVariantKind.EXECUTION_SEMANTICS_TWIN: HornVariantExpectation(
            relation_to_base=HornExpectationRelation.EXECUTION_SEPARATE,
            evidence_invariant_after_alignment=False,
            challenge_outputs_invariant_after_alignment=False,
            requires_execution_separation=True,
            requires_version_space_separation=False,
        ),
        HornVariantKind.AMBIGUITY_DELETED_TWIN: HornVariantExpectation(
            relation_to_base=HornExpectationRelation.IDENTIFIABILITY_SEPARATE,
            evidence_invariant_after_alignment=False,
            challenge_outputs_invariant_after_alignment=False,
            requires_execution_separation=False,
            requires_version_space_separation=True,
        ),
    }
)


@dataclass(frozen=True, order=True, slots=True)
class HornSurfaceFact:
    """One inert source-level relation over opaque symbols."""

    relation: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HornSurfaceDemonstration:
    """One source-level before/after pair."""

    initial: tuple[HornSurfaceFact, ...]
    terminal: tuple[HornSurfaceFact, ...]


@dataclass(frozen=True, slots=True)
class HornSurfacePresentation:
    """Public source presentation, excluding every assessor alignment."""

    schema: str
    declarations: tuple[HornSurfaceFact, ...]
    demonstrations: tuple[HornSurfaceDemonstration, ...]


@dataclass(frozen=True, slots=True)
class HornCanonicalAlignment:
    """Assessor-only quotient from opaque source symbols to board indices."""

    predicate_symbols: tuple[tuple[str, int], ...]
    object_symbols: tuple[tuple[str, int], ...]
    type_symbols: tuple[tuple[str, int], ...]
    role_symbols: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class HornVariantChallenge:
    """One late challenge plus assessor-only exact possible outcomes."""

    challenge_index: int
    canonical_initial: tuple[GroundAtom, ...]
    presented_initial: tuple[HornSurfaceFact, ...]
    source: str
    expected_terminals: tuple[tuple[GroundAtom, ...], ...]
    requires_abstention: bool


@dataclass(frozen=True, slots=True)
class HornVariantMaterialization:
    """One complete offline world, presentation, and assessor receipt."""

    kind: HornVariantKind
    seed: int
    target_theory_index: int
    execution_semantics: HornExecutionSemantics
    expectation: HornVariantExpectation
    canonical_evidence: tuple[Demonstration, ...]
    presentation: HornSurfacePresentation
    source: str
    assessor_only_alignment: HornCanonicalAlignment
    consistent_theory_indices: tuple[int, ...]
    behavioral_class_count: int
    challenges: tuple[HornVariantChallenge, ...]

    def compiler_source(self) -> str:
        """Return the only world artifact admitted to a compiler process."""

        return self.source

    def late_challenge_source(self, index: int) -> str:
        """Return one post-seal challenge without assessor-side metadata."""

        return self.challenges[index].source


@dataclass(frozen=True, slots=True)
class HornVariantAuditReport:
    """Exact receipt emitted only after every variant contract passes."""

    target_theory_index: int
    variant_count: int
    challenge_count_per_variant: int
    invariant_variant_count: int
    execution_separation_count: int
    ambiguity_separating_challenge_count: int


@dataclass(frozen=True, slots=True)
class _Codebook:
    predicates: tuple[str, ...]
    objects: tuple[tuple[str, ...], ...]
    types: tuple[tuple[str, ...], ...]
    roles: tuple[str, ...]


def _opaque(
    seed: int,
    namespace: str,
    category: str,
    index: int,
    copy_index: int = 0,
) -> str:
    payload = (
        f"horn-variant-v1|{seed}|{namespace}|{category}|"
        f"{index}|{copy_index}"
    ).encode("ascii")
    return f"z{sha256(payload).hexdigest()[:15]}"


def _lexical_namespace(kind: HornVariantKind) -> str:
    if kind in {
        HornVariantKind.EXECUTION_SEMANTICS_TWIN,
        HornVariantKind.AMBIGUITY_DELETED_TWIN,
    }:
        return HornVariantKind.BASE.value
    return kind.value


def _make_codebook(
    seed: int,
    kind: HornVariantKind,
) -> _Codebook:
    namespace = _lexical_namespace(kind)
    object_copies = 2 if kind == HornVariantKind.ALIAS_SPLIT else 1
    type_copies = 2 if kind == HornVariantKind.TYPE_TWIN else 1
    return _Codebook(
        predicates=tuple(
            _opaque(seed, namespace, "predicate", index)
            for index in range(len(PREDICATES))
        ),
        objects=tuple(
            tuple(
                _opaque(seed, namespace, "object", index, copy_index)
                for copy_index in range(object_copies)
            )
            for index in range(len(OBJECT_TYPES))
        ),
        types=tuple(
            tuple(
                _opaque(seed, namespace, "type", index, copy_index)
                for copy_index in range(type_copies)
            )
            for index in range(len(set(OBJECT_TYPES)))
        ),
        roles=tuple(
            _opaque(seed, namespace, "role", index)
            for index in range(2)
        ),
    )


def _alignment(codebook: _Codebook) -> HornCanonicalAlignment:
    return HornCanonicalAlignment(
        predicate_symbols=tuple(
            (symbol, index)
            for index, symbol in enumerate(codebook.predicates)
        ),
        object_symbols=tuple(
            (symbol, index)
            for index, symbols in enumerate(codebook.objects)
            for symbol in symbols
        ),
        type_symbols=tuple(
            (symbol, index)
            for index, symbols in enumerate(codebook.types)
            for symbol in symbols
        ),
        role_symbols=tuple(
            (symbol, index)
            for index, symbol in enumerate(codebook.roles)
        ),
    )


def _declarations(
    kind: HornVariantKind,
    codebook: _Codebook,
) -> tuple[HornSurfaceFact, ...]:
    facts: list[HornSurfaceFact] = []
    for object_index, object_symbols in enumerate(codebook.objects):
        type_index = OBJECT_TYPES[object_index]
        if kind == HornVariantKind.TYPE_TWIN:
            type_symbol = codebook.types[type_index][object_index % 2]
        else:
            type_symbol = codebook.types[type_index][0]
        for object_symbol in object_symbols:
            facts.append(
                HornSurfaceFact(
                    "@entity_type",
                    (object_symbol, type_symbol),
                )
            )
        if len(object_symbols) == 2:
            facts.append(
                HornSurfaceFact("@alias", object_symbols)
            )
    for type_symbols in codebook.types:
        if len(type_symbols) == 2:
            facts.append(
                HornSurfaceFact("@type_twin", type_symbols)
            )
    for predicate in PREDICATES:
        predicate_symbol = codebook.predicates[predicate.index]
        for role_index, type_index in enumerate(
            predicate.argument_types
        ):
            admitted_types = codebook.types[type_index]
            for type_symbol in admitted_types:
                facts.append(
                    HornSurfaceFact(
                        "@predicate_role",
                        (
                            predicate_symbol,
                            codebook.roles[role_index],
                            type_symbol,
                        ),
                    )
                )
    if kind == HornVariantKind.ALPHA_REORDER:
        return tuple(reversed(sorted(facts)))
    return tuple(sorted(facts))


def _mention(
    codebook: _Codebook,
    object_index: int,
    occurrence_key: str,
    argument_index: int,
) -> str:
    symbols = codebook.objects[object_index]
    if len(symbols) == 1:
        return symbols[0]
    digest = sha256(
        f"{occurrence_key}|{argument_index}".encode("ascii")
    ).digest()
    return symbols[digest[0] % len(symbols)]


def _ordered_atoms(
    atoms: tuple[GroundAtom, ...],
    kind: HornVariantKind,
) -> tuple[GroundAtom, ...]:
    ordered = tuple(sorted(atoms))
    if kind == HornVariantKind.ALPHA_REORDER:
        return tuple(reversed(ordered))
    return ordered


def _present_atoms(
    atoms: tuple[GroundAtom, ...],
    *,
    kind: HornVariantKind,
    codebook: _Codebook,
    seed: int,
    occurrence_key: str,
) -> tuple[HornSurfaceFact, ...]:
    ordered_atoms = _ordered_atoms(atoms, kind)
    if kind != HornVariantKind.RELATION_REIFICATION:
        return tuple(
            HornSurfaceFact(
                codebook.predicates[atom.predicate],
                tuple(
                    _mention(
                        codebook,
                        object_index,
                        f"{occurrence_key}:{atom_index}",
                        argument_index,
                    )
                    for argument_index, object_index in enumerate(
                        atom.arguments
                    )
                ),
            )
            for atom_index, atom in enumerate(ordered_atoms)
        )

    facts: list[HornSurfaceFact] = []
    for atom_index, atom in enumerate(ordered_atoms):
        fact_node = _opaque(
            seed,
            HornVariantKind.RELATION_REIFICATION.value,
            occurrence_key,
            atom_index,
        )
        facts.extend(
            (
                HornSurfaceFact("@fact", (fact_node,)),
                HornSurfaceFact(
                    "@predicate",
                    (fact_node, codebook.predicates[atom.predicate]),
                ),
            )
        )
        for argument_index, object_index in enumerate(atom.arguments):
            facts.append(
                HornSurfaceFact(
                    "@argument",
                    (
                        fact_node,
                        codebook.roles[argument_index],
                        _mention(
                            codebook,
                            object_index,
                            f"{occurrence_key}:{atom_index}",
                            argument_index,
                        ),
                    ),
                )
            )
    return tuple(facts)


def _present_evidence(
    evidence: tuple[Demonstration, ...],
    *,
    kind: HornVariantKind,
    codebook: _Codebook,
    seed: int,
) -> HornSurfacePresentation:
    indexed = tuple(enumerate(evidence))
    if kind == HornVariantKind.ALPHA_REORDER:
        indexed = tuple(reversed(indexed))
    demonstrations = tuple(
        HornSurfaceDemonstration(
            initial=_present_atoms(
                demo.initial,
                kind=kind,
                codebook=codebook,
                seed=seed,
                occurrence_key=f"demo:{demo_index}:initial",
            ),
            terminal=_present_atoms(
                demo.terminal,
                kind=kind,
                codebook=codebook,
                seed=seed,
                occurrence_key=f"demo:{demo_index}:terminal",
            ),
        )
        for demo_index, demo in indexed
    )
    schema = (
        "reified-incidence-v1"
        if kind == HornVariantKind.RELATION_REIFICATION
        else "direct-relations-v1"
    )
    return HornSurfacePresentation(
        schema=schema,
        declarations=_declarations(kind, codebook),
        demonstrations=demonstrations,
    )


def _render_fact(fact: HornSurfaceFact) -> str:
    return f"{fact.relation}({','.join(fact.arguments)})"


def render_horn_presentation(
    presentation: HornSurfacePresentation,
) -> str:
    """Render an exact source without serializing its canonical alignment."""

    lines = [f"schema {presentation.schema}", "declarations"]
    lines.extend(_render_fact(fact) for fact in presentation.declarations)
    lines.append("evidence")
    for index, demo in enumerate(presentation.demonstrations):
        lines.append(f"demo {index}")
        lines.append(
            "before " + " ".join(
                _render_fact(fact) for fact in demo.initial
            )
        )
        lines.append(
            "after " + " ".join(
                _render_fact(fact) for fact in demo.terminal
            )
        )
    return "\n".join(lines) + "\n"


def _render_challenge(
    schema: str,
    facts: tuple[HornSurfaceFact, ...],
) -> str:
    return (
        f"schema {schema}\nchallenge\n"
        + " ".join(_render_fact(fact) for fact in facts)
        + "\n"
    )


def _decode_direct(
    facts: tuple[HornSurfaceFact, ...],
    alignment: HornCanonicalAlignment,
) -> tuple[GroundAtom, ...]:
    predicates = dict(alignment.predicate_symbols)
    objects = dict(alignment.object_symbols)
    result: list[GroundAtom] = []
    for fact in facts:
        if fact.relation not in predicates:
            raise ValueError("direct atom relation is not aligned")
        try:
            arguments = tuple(objects[value] for value in fact.arguments)
        except KeyError as error:
            raise ValueError("direct atom object is not aligned") from error
        atom = GroundAtom(predicates[fact.relation], arguments)
        if (
            len(arguments)
            != len(PREDICATES[atom.predicate].argument_types)
        ):
            raise ValueError("direct atom arity differs")
        result.append(atom)
    return tuple(sorted(result))


def _decode_reified(
    facts: tuple[HornSurfaceFact, ...],
    alignment: HornCanonicalAlignment,
) -> tuple[GroundAtom, ...]:
    predicates = dict(alignment.predicate_symbols)
    objects = dict(alignment.object_symbols)
    roles = dict(alignment.role_symbols)
    nodes: set[str] = set()
    node_predicates: dict[str, int] = {}
    node_arguments: dict[str, dict[int, int]] = {}
    for fact in facts:
        if fact.relation == "@fact" and len(fact.arguments) == 1:
            if fact.arguments[0] in nodes:
                raise ValueError("reified fact node is duplicated")
            nodes.add(fact.arguments[0])
        elif fact.relation == "@predicate" and len(fact.arguments) == 2:
            node, predicate_symbol = fact.arguments
            if node in node_predicates or predicate_symbol not in predicates:
                raise ValueError("reified predicate assignment differs")
            node_predicates[node] = predicates[predicate_symbol]
        elif fact.relation == "@argument" and len(fact.arguments) == 3:
            node, role_symbol, object_symbol = fact.arguments
            if role_symbol not in roles or object_symbol not in objects:
                raise ValueError("reified argument alignment differs")
            role = roles[role_symbol]
            arguments = node_arguments.setdefault(node, {})
            if role in arguments:
                raise ValueError("reified argument role is duplicated")
            arguments[role] = objects[object_symbol]
        else:
            raise ValueError("unexpected reified surface fact")
    if nodes != set(node_predicates) or nodes != set(node_arguments):
        raise ValueError("reified incidence graph is incomplete")
    result: list[GroundAtom] = []
    for node in sorted(nodes):
        predicate = node_predicates[node]
        arity = len(PREDICATES[predicate].argument_types)
        arguments = node_arguments[node]
        if set(arguments) != set(range(arity)):
            raise ValueError("reified incidence roles differ")
        result.append(
            GroundAtom(
                predicate,
                tuple(arguments[index] for index in range(arity)),
            )
        )
    return tuple(sorted(result))


def canonicalize_surface_facts(
    facts: tuple[HornSurfaceFact, ...],
    *,
    schema: str,
    alignment: HornCanonicalAlignment,
) -> tuple[GroundAtom, ...]:
    """Apply the assessor-only quotient to one presented fact collection."""

    if schema == "direct-relations-v1":
        return _decode_direct(facts, alignment)
    if schema == "reified-incidence-v1":
        return _decode_reified(facts, alignment)
    raise ValueError("surface schema differs")


def canonicalize_presentation(
    presentation: HornSurfacePresentation,
    alignment: HornCanonicalAlignment,
) -> tuple[Demonstration, ...]:
    """Recover canonical evidence for an offline exactness audit."""

    return tuple(
        Demonstration(
            initial=canonicalize_surface_facts(
                demo.initial,
                schema=presentation.schema,
                alignment=alignment,
            ),
            terminal=canonicalize_surface_facts(
                demo.terminal,
                schema=presentation.schema,
                alignment=alignment,
            ),
        )
        for demo in presentation.demonstrations
    )


def normalized_evidence(
    evidence: tuple[Demonstration, ...],
) -> tuple[Demonstration, ...]:
    """Remove only demonstration and fact ordering, not semantic identity."""

    normalized = tuple(
        Demonstration(
            tuple(sorted(demo.initial)),
            tuple(sorted(demo.terminal)),
        )
        for demo in evidence
    )
    return tuple(
        sorted(normalized, key=lambda demo: (demo.initial, demo.terminal))
    )


def execute_horn_semantics(
    theory_index: int,
    initial: tuple[GroundAtom, ...],
    semantics: HornExecutionSemantics,
) -> tuple[GroundAtom, ...]:
    """Execute one of the two explicit Horn terminal-state policies."""

    closure = execute_closure(THEORIES[theory_index], initial)
    if semantics == HornExecutionSemantics.PERSISTENT_CLOSURE:
        return closure
    if semantics == HornExecutionSemantics.DERIVED_ONLY_CLOSURE:
        initial_set = set(initial)
        return tuple(atom for atom in closure if atom not in initial_set)
    raise ValueError("Horn execution semantics differs")


@lru_cache(maxsize=None)
def _behavior_signature_under(
    theory_index: int,
    semantics: HornExecutionSemantics,
) -> tuple[tuple[GroundAtom, ...], ...]:
    if semantics == HornExecutionSemantics.PERSISTENT_CLOSURE:
        return behavior_signature(theory_index)
    return tuple(
        execute_horn_semantics(theory_index, initial, semantics)
        for initial in challenge_initials()
    )


def _consistent_theories_under(
    evidence: tuple[Demonstration, ...],
    semantics: HornExecutionSemantics,
) -> tuple[int, ...]:
    if semantics == HornExecutionSemantics.PERSISTENT_CLOSURE:
        return consistent_theories(evidence)
    return tuple(
        theory_index
        for theory_index in range(len(THEORIES))
        if all(
            execute_horn_semantics(
                theory_index,
                demo.initial,
                semantics,
            )
            == demo.terminal
            for demo in evidence
        )
    )


def _behavioral_class_count_under(
    theory_indices: tuple[int, ...],
    semantics: HornExecutionSemantics,
) -> int:
    return len(
        {
            _behavior_signature_under(theory_index, semantics)
            for theory_index in theory_indices
        }
    )


@lru_cache(maxsize=None)
def _identifying_evidence_under(
    target_theory_index: int,
    semantics: HornExecutionSemantics,
) -> tuple[Demonstration, ...]:
    if semantics == HornExecutionSemantics.PERSISTENT_CLOSURE:
        return identifying_evidence(target_theory_index)
    remaining = tuple(range(len(THEORIES)))
    evidence: list[Demonstration] = []
    unused = set(range(len(challenge_initials())))
    while _behavioral_class_count_under(remaining, semantics) > 1:
        best: tuple[int, int, tuple[GroundAtom, ...]] | None = None
        for challenge_index in sorted(unused):
            initial = challenge_initials()[challenge_index]
            terminal = execute_horn_semantics(
                target_theory_index,
                initial,
                semantics,
            )
            survivor_count = sum(
                execute_horn_semantics(index, initial, semantics)
                == terminal
                for index in remaining
            )
            candidate = survivor_count, challenge_index, terminal
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[0] == len(remaining):
            raise ValueError("semantic twin is not behaviorally identifiable")
        _, challenge_index, terminal = best
        initial = challenge_initials()[challenge_index]
        evidence.append(Demonstration(initial, terminal))
        unused.remove(challenge_index)
        remaining = tuple(
            index
            for index in remaining
            if execute_horn_semantics(index, initial, semantics)
            == terminal
        )
    return tuple(evidence)


def _common_challenge_indices(
    target_theory_index: int,
    ambiguous_theories: tuple[int, ...],
    *,
    seed: int,
    challenge_count: int,
) -> tuple[int, ...]:
    if not 1 <= challenge_count <= len(challenge_initials()):
        raise ValueError("challenge count is out of range")
    separating = tuple(
        challenge_index
        for challenge_index, initial in enumerate(challenge_initials())
        if len(
            {
                execute_closure(THEORIES[index], initial)
                for index in ambiguous_theories
            }
        )
        > 1
    )
    if not separating:
        raise ValueError("ambiguity twin has no separating challenge")
    first = separating[0]
    remainder = [
        index
        for index in range(len(challenge_initials()))
        if index != first
    ]
    rng = random.Random(
        (seed << 12) ^ (target_theory_index << 4) ^ 0x484F524E
    )
    rng.shuffle(remainder)
    return (first, *remainder[: challenge_count - 1])


def _expected_terminals(
    kind: HornVariantKind,
    theory_index: int,
    consistent: tuple[int, ...],
    initial: tuple[GroundAtom, ...],
) -> tuple[tuple[GroundAtom, ...], ...]:
    if kind == HornVariantKind.AMBIGUITY_DELETED_TWIN:
        return tuple(
            sorted(
                {
                    execute_closure(THEORIES[index], initial)
                    for index in consistent
                }
            )
        )
    semantics = (
        HornExecutionSemantics.DERIVED_ONLY_CLOSURE
        if kind == HornVariantKind.EXECUTION_SEMANTICS_TWIN
        else HornExecutionSemantics.PERSISTENT_CLOSURE
    )
    return (
        execute_horn_semantics(theory_index, initial, semantics),
    )


def _materialize_variant(
    kind: HornVariantKind,
    *,
    seed: int,
    target_theory_index: int,
    evidence: tuple[Demonstration, ...],
    semantics: HornExecutionSemantics,
    challenge_indices: tuple[int, ...],
) -> HornVariantMaterialization:
    codebook = _make_codebook(seed, kind)
    alignment = _alignment(codebook)
    presentation = _present_evidence(
        evidence,
        kind=kind,
        codebook=codebook,
        seed=seed,
    )
    consistent = _consistent_theories_under(evidence, semantics)
    classes = _behavioral_class_count_under(consistent, semantics)
    challenges = tuple(
        HornVariantChallenge(
            challenge_index=challenge_index,
            canonical_initial=challenge_initials()[challenge_index],
            presented_initial=(
                presented := _present_atoms(
                    challenge_initials()[challenge_index],
                    kind=kind,
                    codebook=codebook,
                    seed=seed,
                    occurrence_key=f"challenge:{position}",
                )
            ),
            source=_render_challenge(presentation.schema, presented),
            expected_terminals=_expected_terminals(
                kind,
                target_theory_index,
                consistent,
                challenge_initials()[challenge_index],
            ),
            requires_abstention=(
                kind == HornVariantKind.AMBIGUITY_DELETED_TWIN
            ),
        )
        for position, challenge_index in enumerate(challenge_indices)
    )
    return HornVariantMaterialization(
        kind=kind,
        seed=seed,
        target_theory_index=target_theory_index,
        execution_semantics=semantics,
        expectation=VARIANT_EXPECTATIONS[kind],
        canonical_evidence=evidence,
        presentation=presentation,
        source=render_horn_presentation(presentation),
        assessor_only_alignment=alignment,
        consistent_theory_indices=consistent,
        behavioral_class_count=classes,
        challenges=challenges,
    )


@lru_cache(maxsize=None)
def materialize_horn_variant_set(
    target_theory_index: int,
    *,
    seed: int,
    challenge_count: int = 16,
) -> tuple[HornVariantMaterialization, ...]:
    """Materialize all seven variants with one aligned challenge set."""

    if not 0 <= target_theory_index < len(THEORIES):
        raise ValueError("target theory index is out of range")
    base_evidence = identifying_evidence(target_theory_index)
    if not base_evidence:
        raise ValueError("base theory unexpectedly needs no evidence")
    ambiguous_evidence = base_evidence[:-1]
    ambiguous_theories = consistent_theories(ambiguous_evidence)
    challenge_indices = _common_challenge_indices(
        target_theory_index,
        ambiguous_theories,
        seed=seed,
        challenge_count=challenge_count,
    )
    execution_evidence = _identifying_evidence_under(
        target_theory_index,
        HornExecutionSemantics.DERIVED_ONLY_CLOSURE,
    )
    variants = []
    for kind in VARIANT_ORDER:
        if kind == HornVariantKind.EXECUTION_SEMANTICS_TWIN:
            evidence = execution_evidence
            semantics = HornExecutionSemantics.DERIVED_ONLY_CLOSURE
        elif kind == HornVariantKind.AMBIGUITY_DELETED_TWIN:
            evidence = ambiguous_evidence
            semantics = HornExecutionSemantics.PERSISTENT_CLOSURE
        else:
            evidence = base_evidence
            semantics = HornExecutionSemantics.PERSISTENT_CLOSURE
        variants.append(
            _materialize_variant(
                kind,
                seed=seed,
                target_theory_index=target_theory_index,
                evidence=evidence,
                semantics=semantics,
                challenge_indices=challenge_indices,
            )
        )
    result = tuple(variants)
    audit_horn_variant_set(result)
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _aligned_evidence(
    variant: HornVariantMaterialization,
) -> tuple[Demonstration, ...]:
    return normalized_evidence(
        canonicalize_presentation(
            variant.presentation,
            variant.assessor_only_alignment,
        )
    )


def _aligned_challenge(
    variant: HornVariantMaterialization,
    challenge: HornVariantChallenge,
) -> tuple[GroundAtom, ...]:
    return canonicalize_surface_facts(
        challenge.presented_initial,
        schema=variant.presentation.schema,
        alignment=variant.assessor_only_alignment,
    )


def audit_horn_variant_set(
    variants: tuple[HornVariantMaterialization, ...],
) -> HornVariantAuditReport:
    """Fail closed unless every structural and semantic contract is exact."""

    _require(
        tuple(variant.kind for variant in variants) == VARIANT_ORDER,
        "Horn variant order or membership differs",
    )
    base = variants[0]
    challenge_count = len(base.challenges)
    _require(challenge_count > 0, "Horn variant challenge set is empty")
    _require(
        all(
            variant.target_theory_index == base.target_theory_index
            and variant.seed == base.seed
            and len(variant.challenges) == challenge_count
            for variant in variants
        ),
        "Horn variant identity or challenge count differs",
    )
    _require(
        len({variant.source for variant in variants}) == len(variants),
        "Horn variants do not materialize distinct world sources",
    )
    base_evidence = normalized_evidence(base.canonical_evidence)
    base_challenge_indices = tuple(
        challenge.challenge_index for challenge in base.challenges
    )
    _require(
        _aligned_evidence(base) == base_evidence,
        "base presentation does not round-trip",
    )

    for variant in variants[:5]:
        _require(
            variant.expectation.relation_to_base
            == HornExpectationRelation.CANONICALLY_INVARIANT,
            f"{variant.kind} lacks an invariance expectation",
        )
        _require(
            normalized_evidence(variant.canonical_evidence)
            == base_evidence
            and _aligned_evidence(variant) == base_evidence,
            f"{variant.kind} changes canonical evidence",
        )
        _require(
            variant.consistent_theory_indices
            == base.consistent_theory_indices
            and variant.behavioral_class_count
            == base.behavioral_class_count,
            f"{variant.kind} changes the version space",
        )
        _require(
            tuple(
                challenge.challenge_index
                for challenge in variant.challenges
            )
            == base_challenge_indices,
            f"{variant.kind} changes challenge identity",
        )
        for challenge, base_challenge in zip(
            variant.challenges,
            base.challenges,
            strict=True,
        ):
            _require(
                _aligned_challenge(variant, challenge)
                == challenge.canonical_initial
                == base_challenge.canonical_initial,
                f"{variant.kind} changes an aligned challenge",
            )
            _require(
                challenge.expected_terminals
                == base_challenge.expected_terminals,
                f"{variant.kind} changes a challenge outcome",
            )

    alpha = variants[1]
    _require(
        alpha.assessor_only_alignment.predicate_symbols
        != base.assessor_only_alignment.predicate_symbols
        and alpha.assessor_only_alignment.object_symbols
        != base.assessor_only_alignment.object_symbols,
        "alpha/reorder did not alpha-rename nominal symbols",
    )
    alias = variants[2]
    alias_object_values = tuple(
        value
        for _, value in alias.assessor_only_alignment.object_symbols
    )
    _require(
        all(alias_object_values.count(index) == 2 for index in range(6))
        and any(
            fact.relation == "@alias"
            for fact in alias.presentation.declarations
        ),
        "alias split is not a two-to-one nominal quotient",
    )
    reified = variants[3]
    _require(
        reified.presentation.schema == "reified-incidence-v1"
        and all(
            fact.relation.startswith("@")
            for demo in reified.presentation.demonstrations
            for fact in (*demo.initial, *demo.terminal)
        )
        and any(
            fact.relation == "@argument"
            for demo in reified.presentation.demonstrations
            for fact in (*demo.initial, *demo.terminal)
        ),
        "relation reification is not an incidence graph",
    )
    type_twin = variants[4]
    type_values = tuple(
        value
        for _, value in type_twin.assessor_only_alignment.type_symbols
    )
    _require(
        all(type_values.count(index) == 2 for index in range(2))
        and sum(
            fact.relation == "@type_twin"
            for fact in type_twin.presentation.declarations
        )
        == 2,
        "type twin is not a two-to-one type quotient",
    )

    execution = variants[5]
    _require(
        execution.expectation.requires_execution_separation
        and execution.execution_semantics
        == HornExecutionSemantics.DERIVED_ONLY_CLOSURE
        and execution.behavioral_class_count == 1
        and execution.target_theory_index
        in execution.consistent_theory_indices,
        "execution-semantics twin is not identifiable",
    )
    execution_separation = 0
    for challenge, base_challenge in zip(
        execution.challenges,
        base.challenges,
        strict=True,
    ):
        _require(
            _aligned_challenge(execution, challenge)
            == base_challenge.canonical_initial,
            "execution-semantics twin changes challenge identity",
        )
        if challenge.expected_terminals != base_challenge.expected_terminals:
            execution_separation += 1
    _require(
        execution_separation == challenge_count,
        "execution-semantics twin fails exact separation",
    )

    ambiguity = variants[6]
    _require(
        ambiguity.expectation.requires_version_space_separation
        and ambiguity.execution_semantics
        == HornExecutionSemantics.PERSISTENT_CLOSURE
        and len(ambiguity.canonical_evidence)
        == len(base.canonical_evidence) - 1
        and ambiguity.behavioral_class_count >= 2
        and ambiguity.target_theory_index
        in ambiguity.consistent_theory_indices,
        "ambiguity deletion does not enlarge the version space",
    )
    ambiguity_separating = 0
    for challenge, base_challenge in zip(
        ambiguity.challenges,
        base.challenges,
        strict=True,
    ):
        _require(
            challenge.requires_abstention
            and _aligned_challenge(ambiguity, challenge)
            == base_challenge.canonical_initial,
            "ambiguity twin changes challenge identity or answer policy",
        )
        if len(challenge.expected_terminals) > 1:
            ambiguity_separating += 1
    _require(
        ambiguity_separating > 0,
        "ambiguity twin has no behaviorally separating challenge",
    )

    return HornVariantAuditReport(
        target_theory_index=base.target_theory_index,
        variant_count=len(variants),
        challenge_count_per_variant=challenge_count,
        invariant_variant_count=5,
        execution_separation_count=execution_separation,
        ambiguity_separating_challenge_count=ambiguity_separating,
    )


__all__ = [
    "HornCanonicalAlignment",
    "HornExecutionSemantics",
    "HornExpectationRelation",
    "HornSurfaceDemonstration",
    "HornSurfaceFact",
    "HornSurfacePresentation",
    "HornVariantAuditReport",
    "HornVariantChallenge",
    "HornVariantExpectation",
    "HornVariantKind",
    "HornVariantMaterialization",
    "VARIANT_EXPECTATIONS",
    "VARIANT_ORDER",
    "audit_horn_variant_set",
    "canonicalize_presentation",
    "canonicalize_surface_facts",
    "execute_horn_semantics",
    "materialize_horn_variant_set",
    "normalized_evidence",
    "render_horn_presentation",
]
