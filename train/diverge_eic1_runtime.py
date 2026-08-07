"""Exact candidate-involution projection for DIVERGE-EIC1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch

from diverge_cgl1_runtime import (
    ANSWERS,
    CGL1Config,
    CausalGroundingInterpreter,
    CGL1RuntimeError,
)
from diverge_nve1_data import symbol_occurrence_groups
from diverge_pqi1_runtime import canonicalize_query


SCHEMA = "shohin-diverge-eic1-runtime-v1"
EIC1Control = Literal["normal", "scrub_context", "swap_mentions"]
ProjectionMode = Literal["involution", "duplicate"]
SYSTEM = (
    "Judge whether a claim follows from an instruction. "
    "Answer exactly YES or NO without explanation."
)


class EIC1RuntimeError(RuntimeError):
    """An EIC1 identity action or projected transaction differs."""


@dataclass(frozen=True, slots=True)
class EIC1Config(CGL1Config):
    projection_mode: ProjectionMode = "involution"

    def validate(self) -> None:
        CGL1Config.validate(self)
        if self.projection_mode not in ("involution", "duplicate"):
            raise EIC1RuntimeError("EIC1 projection mode differs")


def project_candidate_scores(
    first: torch.Tensor,
    partner: torch.Tensor,
    *,
    mode: ProjectionMode,
) -> torch.Tensor:
    if first.ndim != 2 or first.shape[-1] != 2 or partner.shape != first.shape:
        raise EIC1RuntimeError("EIC1 score geometry differs")
    if mode == "involution":
        return 0.5 * (first + partner.flip(dims=(-1,)))
    if mode == "duplicate":
        return 0.5 * (first + partner)
    raise EIC1RuntimeError("EIC1 projection mode differs")


def _canonical_query(
    record: Mapping[str, Any],
    *,
    scrub_context: bool,
    swap_mentions: bool,
) -> str:
    text = str(record["source_text"])
    symbols = tuple(str(value) for value in record["symbols"])
    groups = symbol_occurrence_groups(text, symbols)
    if len(groups) != 2:
        raise EIC1RuntimeError("EIC1 query does not expose two mention groups")
    masks = []
    for _, spans in groups:
        mask = [False] * len(text)
        for left, right in spans:
            mask[left:right] = [True] * (right - left)
        masks.append(mask)
    canonical = canonicalize_query(text, masks, scrub_context=scrub_context).text
    if swap_mentions:
        canonical = canonical.replace("alpha", "__eic1_swap__")
        canonical = canonical.replace("beta", "alpha")
        canonical = canonical.replace("__eic1_swap__", "beta")
    return canonical


def render_claim_prompt(
    record: Mapping[str, Any],
    candidate: int,
    *,
    scrub_context: bool = False,
    swap_mentions: bool = False,
) -> str:
    if candidate not in (0, 1):
        raise EIC1RuntimeError("EIC1 candidate differs")
    query = _canonical_query(
        record,
        scrub_context=scrub_context,
        swap_mentions=swap_mentions,
    )
    target = "alpha" if candidate == 0 else "beta"
    distractor = "beta" if candidate == 0 else "alpha"
    return (
        f"Instruction: {SYSTEM}\n"
        f"Source: {query}\n"
        f"Claim: {target} is the requested answer source and "
        f"{distractor} is the distractor.\nAnswer:"
    )


class EquivariantIdentityCommitter(CausalGroundingInterpreter):
    """CGL-compatible owner whose only commit scores are group projected."""

    config: EIC1Config

    def __init__(self, backbone, tokenizer, config: EIC1Config) -> None:
        super().__init__(backbone, tokenizer, config)

    def _candidate_rows_eic1(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        scrub_context: bool,
        swap_mentions: bool,
    ) -> list[tuple[list[int], list[int]]]:
        rows = []
        for record in records:
            for candidate in (0, 1):
                prompt = list(
                    self.tokenizer.encode(
                        render_claim_prompt(
                            record,
                            candidate,
                            scrub_context=scrub_context,
                            swap_mentions=swap_mentions,
                        ),
                        add_special_tokens=False,
                    ).ids
                )
                if not prompt:
                    raise EIC1RuntimeError("EIC1 prompt tokenized empty")
                for suffix in self.answer_ids:
                    if len(prompt) + len(suffix) > int(self.backbone.cfg.seq_len):
                        raise EIC1RuntimeError("EIC1 sequence exceeds context")
                    rows.append((prompt, list(suffix)))
        return rows

    def _raw_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
        scrub_context: bool,
        swap_mentions: bool,
    ) -> torch.Tensor:
        likelihoods = self._score_rows(
            self._candidate_rows_eic1(
                records,
                scrub_context=scrub_context,
                swap_mentions=swap_mentions,
            ),
            device=device,
        ).reshape(len(records), 2, len(ANSWERS))
        return likelihoods[:, :, 0] - likelihoods[:, :, 1]

    def _projected_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
        scrub_context: bool,
        outer_swap: bool,
    ) -> torch.Tensor:
        first = self._raw_scores(
            records,
            device=device,
            scrub_context=scrub_context,
            swap_mentions=outer_swap,
        )
        partner_swap = (
            outer_swap
            if self.config.projection_mode == "duplicate"
            else not outer_swap
        )
        partner = self._raw_scores(
            records,
            device=device,
            scrub_context=scrub_context,
            swap_mentions=partner_swap,
        )
        return project_candidate_scores(
            first,
            partner,
            mode=self.config.projection_mode,
        )

    def training_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not records:
            raise EIC1RuntimeError("EIC1 training batch is empty")
        return self._projected_scores(
            records,
            device=device,
            scrub_context=False,
            outer_swap=False,
        )

    @torch.no_grad()
    def candidate_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
        batch_size: int,
        control: EIC1Control = "normal",
    ) -> torch.Tensor:
        if batch_size <= 0 or control not in (
            "normal",
            "scrub_context",
            "swap_mentions",
        ):
            raise EIC1RuntimeError("EIC1 evaluation contract differs")
        outputs = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            outputs.append(
                self._projected_scores(
                    batch,
                    device=device,
                    scrub_context=control == "scrub_context",
                    outer_swap=control == "swap_mentions",
                ).cpu()
            )
        return torch.cat(outputs)


__all__ = [
    "EIC1Config",
    "EIC1Control",
    "EIC1RuntimeError",
    "EquivariantIdentityCommitter",
    "ProjectionMode",
    "SCHEMA",
    "project_candidate_scores",
    "render_claim_prompt",
]
