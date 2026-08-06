#!/usr/bin/env python3
"""Assessor-side labels for the frozen DIVERGE-MQB1 mention board."""

from __future__ import annotations

from dataclasses import dataclass

from diverge_mei1_data import (
    EVIDENCE_COHORTS,
    ProbeEvidence,
    generate_probe_evidence,
)
from diverge_mei1_runtime import REGISTER_COUNT


BEFORE = 0
AFTER = 1
PHASE_COUNT = 2
FIELD_COUNT = PHASE_COUNT * REGISTER_COUNT


_TEMPLATES = {
    "train": (
        "delayed audit before register zero {b0} register one {b1} register two {b2} register three {b3} register four {b4} after register zero {a0} register one {a1} register two {a2} register three {a3} register four {a4}",
        "probe began with slot 0 {b0} slot 1 {b1} slot 2 {b2} slot 3 {b3} slot 4 {b4} and ended with slot 0 {a0} slot 1 {a1} slot 2 {a2} slot 3 {a3} slot 4 {a4}",
        "input cells c0 {b0} c1 {b1} c2 {b2} c3 {b3} c4 {b4} output cells c0 {a0} c1 {a1} c2 {a2} c3 {a3} c4 {a4}",
        "witness start r0 {b0} r1 {b1} r2 {b2} r3 {b3} r4 {b4} witness finish r0 {a0} r1 {a1} r2 {a2} r3 {a3} r4 {a4}",
    ),
    "lexical_shift": (
        "inspection antecedent ledger alpha {b0} beta {b1} gamma {b2} delta {b3} epsilon {b4} consequent ledger alpha {a0} beta {a1} gamma {a2} delta {a3} epsilon {a4}",
        "diagnostic entry positions first {b0} second {b1} third {b2} fourth {b3} fifth {b4} exit positions first {a0} second {a1} third {a2} fourth {a3} fifth {a4}",
    ),
    "renderer_shift": (
        "after four {a4} three {a3} two {a2} one {a1} zero {a0} separator before four {b4} three {b3} two {b2} one {b1} zero {b0}",
        "observation table final 0 colon {a0} 1 colon {a1} 2 colon {a2} 3 colon {a3} 4 colon {a4} initial 0 colon {b0} 1 colon {b1} 2 colon {b2} 3 colon {b3} 4 colon {b4}",
    ),
    "composition_shift": (
        "archive batch {noise} is irrelevant final readings second {a1} fourth {a3} first {a0} fifth {a4} third {a2} while initial readings third {b2} first {b0} fifth {b4} second {b1} fourth {b3} audit complete",
        "ignore checksum {noise} the terminal register vector has index 4 {a4} index 2 {a2} index 0 {a0} index 3 {a3} index 1 {a1} whereas the starting vector has index 1 {b1} index 3 {b3} index 0 {b0} index 4 {b4} index 2 {b2}",
    ),
}


@dataclass(frozen=True, slots=True)
class GoldEvidenceMention:
    word_index: int
    phase: int
    address: int
    value: int

    @property
    def field(self) -> int:
        return self.phase * REGISTER_COUNT + self.address


@dataclass(frozen=True, slots=True)
class MentionEvidence:
    evidence_id: str
    cohort: str
    words: tuple[str, ...]
    mentions: tuple[GoldEvidenceMention, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]
    program: int | None


def render_mention_probe(row: ProbeEvidence) -> MentionEvidence:
    """Render the MEI1 words exactly while retaining assessor-only anchors."""

    if row.cohort not in EVIDENCE_COHORTS:
        raise ValueError("unknown evidence cohort")
    markers: dict[str, tuple[int, int, int]] = {}
    fields: dict[str, str | int] = {"noise": row.noise}
    for phase, prefix, values in (
        (BEFORE, "b", row.before),
        (AFTER, "a", row.after),
    ):
        for address, value in enumerate(values):
            marker = f"MQB1_{prefix.upper()}{address}_VALUE"
            fields[f"{prefix}{address}"] = marker
            markers[marker] = (phase, address, int(value))
    template = _TEMPLATES[row.cohort][row.renderer % len(_TEMPLATES[row.cohort])]
    rendered = template.format(**fields).split()
    words: list[str] = []
    mentions: list[GoldEvidenceMention] = []
    for word_index, word in enumerate(rendered):
        if word in markers:
            phase, address, value = markers[word]
            words.append(str(value))
            mentions.append(GoldEvidenceMention(word_index, phase, address, value))
        else:
            words.append(word)
    mentions.sort(key=lambda mention: mention.field)
    if len(mentions) != FIELD_COUNT or {row.field for row in mentions} != set(range(FIELD_COUNT)):
        raise ValueError("annotated renderer lost a typed field")
    if tuple(words) != row.words:
        raise ValueError("MQB1 rendering differs from frozen MEI1 words")
    return MentionEvidence(
        evidence_id=row.evidence_id.replace("mei1-probe", "mqb1-probe"),
        cohort=row.cohort,
        words=tuple(words),
        mentions=tuple(mentions),
        before=row.before,
        after=row.after,
        program=row.program,
    )


def generate_mention_evidence(
    *,
    seed: int,
    cohort: str,
    program: int | None = None,
    sample_program: bool = True,
) -> MentionEvidence:
    return render_mention_probe(
        generate_probe_evidence(
            seed=seed,
            cohort=cohort,
            program=program,
            sample_program=sample_program,
        )
    )
