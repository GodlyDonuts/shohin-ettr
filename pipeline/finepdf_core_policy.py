#!/usr/bin/env python3
"""Deterministic, inspectable triage for the English FinePDFs-Edu candidate.

This policy is deliberately an analysis policy, not a training admission. It
separates obvious PDF/web detritus from plausible long-form educational
material and records the evidence used for every decision. A selected tier
still needs held-out human review, cross-source residualization, and an
equal-token utility ablation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping
from urllib.parse import urlparse


POLICY_SCHEMA = "shohin-finepdf-candidate-policy-v1"

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

_DOWNLOAD_SPAM = (
    re.compile(r"\bthank you (?:utterly|very|so) much for downloading\b"),
    re.compile(r"\bgetting the books? .{0,80}\bnow is not\b"),
    re.compile(r"\b(?:ebook|book) .{0,80}\b(?:answers?|solutions?)\b.{0,80}\bdownload\b"),
    re.compile(r"\b(?:harmful virus|subsequent to a mug of coffee)\b"),
    re.compile(r"\b(?:reviewing|getting) a[n]? (?:ebook|book)\b"),
    re.compile(r"\bthis online (?:statement|publication|book)\b"),
)

_LOW_DENSITY_HEAD = {
    "calendar_or_agenda": (
        re.compile(r"^(?:.{0,160}\b)?(?:calendar of events|meeting agenda)\b"),
        re.compile(r"^(?:.{0,160}\b)?(?:school|academic) calendar\b"),
    ),
    "catalog_or_prospectus": (
        re.compile(r"^(?:.{0,180}\b)?(?:course|product|college) catalog(?:ue)?\b"),
        re.compile(r"^(?:.{0,180}\b)?(?:admissions? prospectus|program catalog)\b"),
    ),
    "menu": (
        re.compile(r"^(?:.{0,180}\b)?(?:breakfast|lunch|school|cafeteria) menu\b"),
        re.compile(r"^(?:.{0,180}\b)?menu\s+(?:monday|tuesday|wednesday)\b"),
    ),
    "newsletter": (
        re.compile(r"^(?:.{0,220}\b)?(?:weekly|monthly|quarterly|school|community) newsletter\b"),
        re.compile(r"^(?:.{0,160}\b)?newsletter\s+(?:issue|volume|vol\.?|no\.?|number|\d)\b"),
    ),
    "minutes_or_directory": (
        re.compile(r"^(?:.{0,180}\b)?(?:board|committee|council) meeting minutes\b"),
        re.compile(r"^(?:.{0,180}\b)?(?:staff|member|telephone) directory\b"),
    ),
}

_ANSWER_KEY = re.compile(r"\b(?:answer key|answers and explanations|annotated test)\b")
_FORM_HEAD = re.compile(
    r"^(?:.{0,180}\b)?(?:application|registration|consent|enrollment) form\b"
)
_PERSONAL_PROFILE_HEAD = re.compile(
    r"^(?:.{0,120}\b)?(?:curriculum vitae|resume|professional profile)\b"
)

_STRONG_SIGNALS = {
    "scholarly_structure": (
        re.compile(r"\babstract\b"),
        re.compile(r"\b(?:references|bibliography)\b"),
        re.compile(r"\b(?:methodology|methods|experimental design)\b"),
        re.compile(r"\b(?:results|findings)\b"),
    ),
    "formal_reasoning": (
        re.compile(r"\b(?:theorem|lemma|proposition|corollary)\b"),
        re.compile(r"\bproof\b"),
    ),
    "instructional_structure": (
        re.compile(r"\bchapter\s+\d+\b"),
        re.compile(r"\b(?:worked example|practice problem|exercise\s+\d+)\b"),
        re.compile(r"\b(?:learning objectives?|learning outcomes?)\b"),
    ),
    "research_document": (
        re.compile(r"\b(?:doctoral dissertation|master'?s thesis|technical report)\b"),
        re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/"),
    ),
}

_BROAD_AUTHORITY_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".gc.ca",
    ".edu",
    ".ac.uk",
)

_SCHOLARLY_HOST_MARKERS = (
    "arxiv.org",
    "digitalcommons.",
    "repository.",
    "researchgate.net",
    "semanticscholar.org",
    "files.eric.ed.gov",
)


@dataclass(frozen=True)
class FinePdfDecision:
    schema: str
    tier: str
    reason_codes: tuple[str, ...]
    education_score_min: float
    education_score_max: float
    education_score_mean: float
    education_score_count: int
    word_count: int
    strong_signal_count: int
    authority_signal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_text(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def _education_scores(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    raw = metadata.get("fw_edu_scores")
    if not isinstance(raw, (list, tuple)):
        return ()
    scores = []
    for value in raw:
        if isinstance(value, bool):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append(score)
    return tuple(scores)


def _hostname(metadata: Mapping[str, Any], domain: str | None) -> str:
    if domain:
        return str(domain).strip().casefold().split(":", 1)[0]
    value = metadata.get("url")
    if not isinstance(value, str):
        return ""
    try:
        return (urlparse(value).hostname or "").casefold()
    except ValueError:
        return ""


def _authority_signal(hostname: str) -> bool:
    return (
        any(
            hostname == suffix.removeprefix(".") or hostname.endswith(suffix)
            for suffix in _BROAD_AUTHORITY_SUFFIXES
        )
        or any(marker in hostname for marker in _SCHOLARLY_HOST_MARKERS)
    )


def _strong_signals(text: str) -> tuple[str, ...]:
    signals = []
    for name, patterns in _STRONG_SIGNALS.items():
        hits = sum(pattern.search(text) is not None for pattern in patterns)
        minimum = 2 if name in {"scholarly_structure", "formal_reasoning"} else 1
        if hits >= minimum:
            signals.append(name)
    return tuple(signals)


def classify_finepdf_candidate(
    *,
    text: str,
    metadata: Mapping[str, Any],
    domain: str | None = None,
) -> FinePdfDecision:
    """Classify one retained FinePDF document into core/residual/reject.

    The policy intentionally avoids treating the upstream educational score as
    ground truth. Two-ended PDFs use the lower score as a continuity signal;
    strong long-form structure and source authority can rescue moderate-score
    records, but never the explicit download-spam class.
    """

    normalized = _normalized_text(text)
    head = normalized[:2_500]
    words = len(_WORD.findall(text))
    scores = _education_scores(metadata)
    # FinePDF educational scores are nonnegative. The finite sentinel keeps
    # text-free audit JSON standards-compliant while missingness remains an
    # explicit hard-reject reason.
    score_min = min(scores) if scores else -1.0
    score_max = max(scores) if scores else -1.0
    score_mean = sum(scores) / len(scores) if scores else -1.0
    hostname = _hostname(metadata, domain)
    authority = _authority_signal(hostname)
    strong = _strong_signals(normalized)
    reasons: list[str] = []

    spam_hits = sum(pattern.search(normalized) is not None for pattern in _DOWNLOAD_SPAM)
    if spam_hits >= 1:
        reasons.extend(("download_aggregation_spam", "hard_reject"))
        tier = "reject"
    else:
        low_density = []
        for name, patterns in _LOW_DENSITY_HEAD.items():
            if any(pattern.search(head) is not None for pattern in patterns):
                low_density.append(name)
        if _FORM_HEAD.search(head):
            low_density.append("form")
        if _PERSONAL_PROFILE_HEAD.search(head):
            low_density.append("personal_profile")
        if _ANSWER_KEY.search(head) and not strong:
            low_density.append("answer_key_without_exposition")

        if not scores:
            reasons.extend(("missing_education_score", "hard_reject"))
            tier = "reject"
        elif low_density:
            reasons.extend(f"low_density:{name}" for name in sorted(set(low_density)))
            reasons.append("document_type_excluded_from_core")
            tier = "reject"
        else:
            if strong:
                reasons.extend(f"strong:{name}" for name in strong)
            if authority:
                reasons.append("authority_origin")
            if len(scores) == 2 and score_min >= 2.0:
                reasons.append("two_ended_score_continuity")
            elif len(scores) == 2:
                reasons.append("two_ended_score_tail_weak")

            high_score_core = score_min >= 2.50
            structured_core = (
                bool(strong)
                and score_mean >= 1.75
                and score_min >= 1.00
                and words >= 900
            )
            authority_core = (
                authority
                and bool(strong)
                and score_mean >= 1.50
                and score_min >= 0.75
                and words >= 1_200
            )
            if words < 350:
                tier = "residual"
                reasons.extend(("low_word_count", "requires_equal_token_ablation"))
            elif high_score_core or structured_core or authority_core:
                tier = "core"
                if high_score_core:
                    reasons.append("high_score_core")
                if structured_core:
                    reasons.append("structured_core")
                if authority_core:
                    reasons.append("authority_structured_core")
            else:
                tier = "residual"
                if score_max < 1.50 and not strong:
                    reasons.append("low_education_score")
                reasons.append("requires_equal_token_ablation")

    return FinePdfDecision(
        schema=POLICY_SCHEMA,
        tier=tier,
        reason_codes=tuple(sorted(set(reasons))),
        education_score_min=score_min,
        education_score_max=score_max,
        education_score_mean=score_mean,
        education_score_count=len(scores),
        word_count=words,
        strong_signal_count=len(strong),
        authority_signal=authority,
    )
