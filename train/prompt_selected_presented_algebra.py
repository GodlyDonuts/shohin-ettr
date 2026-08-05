"""Whole-presentation completion and falsification over anonymous generators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class PresentedAlgebraError(RuntimeError):
    """The presented-algebra interface violated its structural contract."""


@dataclass(frozen=True, slots=True)
class PresentedAlgebraConfig:
    carrier_size: int = 11
    maximum_generators: int = 3
    maximum_observations: int = 27
    maximum_challenges: int = 8
    maximum_word_length: int = 12

    @property
    def candidates(self) -> int:
        return 1 << self.maximum_generators

    def validate(self) -> None:
        if min(
            self.carrier_size,
            self.maximum_generators,
            self.maximum_observations,
            self.maximum_challenges,
            self.maximum_word_length,
        ) <= 0:
            raise PresentedAlgebraError("presentation dimensions must be positive")
        expected = self.maximum_generators * (self.carrier_size - 2)
        if self.maximum_observations < expected:
            raise PresentedAlgebraError("observation board cannot hold two omissions")


@dataclass(frozen=True, slots=True)
class PresentedAlgebraResult:
    answer_logits: torch.Tensor
    answer: torch.Tensor
    selected_candidate: torch.Tensor
    selected_tables: torch.Tensor
    candidate_tables: torch.Tensor
    candidate_mismatches: torch.Tensor
    challenge_exact: torch.Tensor
    selection_margin: torch.Tensor


def _apply_word(
    tables: torch.Tensor,
    start: torch.Tensor,
    word: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a generator word to starts for one table set per batch row."""

    if tables.ndim != 4:
        raise PresentedAlgebraError("tables must be [batch, generator, input, output]")
    if word.shape != word_mask.shape or word.shape[:-1] != start.shape:
        raise PresentedAlgebraError("word and start geometry differs")
    batch = tables.shape[0]
    leading = start.shape[1:]
    expanded_tables = tables
    for _ in leading:
        expanded_tables = expanded_tables.unsqueeze(1)
    expanded_tables = expanded_tables.expand(
        batch, *leading, *tables.shape[1:]
    )
    state = start
    for position in range(word.shape[-1]):
        generator = word[..., position]
        gather_index = generator[..., None, None, None].expand(
            *generator.shape,
            1,
            tables.shape[-2],
            tables.shape[-1],
        )
        action = torch.gather(
            expanded_tables, -3, gather_index
        ).squeeze(-3)
        next_state = action.gather(-2, state[..., None, None].expand(
            *state.shape, 1, action.shape[-1]
        )).squeeze(-2).argmax(-1)
        state = torch.where(word_mask[..., position], next_state, state)
    return state


class PromptSelectedPresentedAlgebra:
    """Complete candidate actions, falsify them, and execute one lineage."""

    def __init__(self, config: PresentedAlgebraConfig):
        config.validate()
        self.config = config

    def _base_tables(
        self,
        observation_generator: torch.Tensor,
        observation_input: torch.Tensor,
        observation_output: torch.Tensor,
        observation_mask: torch.Tensor,
        generator_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, observations = observation_generator.shape
        config = self.config
        if (
            observation_input.shape != (batch, observations)
            or observation_output.shape != (batch, observations)
            or observation_mask.shape != (batch, observations)
            or generator_mask.shape != (batch, config.maximum_generators)
        ):
            raise PresentedAlgebraError("observation geometry differs")
        if observation_generator.max().item() >= config.maximum_generators:
            raise PresentedAlgebraError("observation generator is outside carrier")
        if max(
            observation_input.max().item(), observation_output.max().item()
        ) >= config.carrier_size:
            raise PresentedAlgebraError("observation state is outside carrier")

        table = torch.zeros(
            batch,
            config.maximum_generators,
            config.carrier_size,
            config.carrier_size,
            dtype=torch.float32,
            device=observation_generator.device,
        )
        batch_index = torch.arange(batch, device=table.device)[:, None].expand(
            -1, observations
        )
        table.index_put_(
            (
                batch_index[observation_mask],
                observation_generator[observation_mask],
                observation_input[observation_mask],
                observation_output[observation_mask],
            ),
            torch.ones(observation_mask.sum(), device=table.device),
            accumulate=True,
        )
        if table.max().item() > 1:
            raise PresentedAlgebraError("duplicate observation conflicts with table")
        row_seen = table.sum(-1).gt(0)
        output_seen = table.sum(-2).gt(0)
        identity = torch.eye(config.carrier_size, device=table.device)
        table = torch.where(
            generator_mask[..., None, None], table, identity[None, None]
        )
        row_seen = torch.where(
            generator_mask[..., None], row_seen, torch.ones_like(row_seen)
        )
        output_seen = torch.where(
            generator_mask[..., None], output_seen, torch.ones_like(output_seen)
        )
        return table, row_seen, output_seen

    def _candidate_tables(
        self,
        base: torch.Tensor,
        row_seen: torch.Tensor,
        output_seen: torch.Tensor,
        generator_mask: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        batch = base.shape[0]
        candidates = base[:, None].expand(-1, config.candidates, -1, -1, -1).clone()
        for row in range(batch):
            for generator in range(config.maximum_generators):
                if not generator_mask[row, generator]:
                    continue
                missing_inputs = (~row_seen[row, generator]).nonzero().flatten()
                missing_outputs = (~output_seen[row, generator]).nonzero().flatten()
                if missing_inputs.numel() != 2 or missing_outputs.numel() != 2:
                    raise PresentedAlgebraError(
                        "each active generator must omit exactly two permutation rows"
                    )
                for candidate in range(config.candidates):
                    swapped = (candidate >> generator) & 1
                    first_output = missing_outputs[swapped]
                    second_output = missing_outputs[1 - swapped]
                    candidates[
                        row, candidate, generator, missing_inputs[0], first_output
                    ] = 1
                    candidates[
                        row, candidate, generator, missing_inputs[1], second_output
                    ] = 1
        if not torch.equal(
            candidates.sum(-1), torch.ones_like(candidates.sum(-1))
        ):
            raise PresentedAlgebraError("candidate action is not total")
        if not torch.equal(
            candidates.sum(-2), torch.ones_like(candidates.sum(-2))
        ):
            raise PresentedAlgebraError("candidate action is not a permutation")
        return candidates

    def compile(
        self,
        observation_generator: torch.Tensor,
        observation_input: torch.Tensor,
        observation_output: torch.Tensor,
        observation_mask: torch.Tensor,
        generator_mask: torch.Tensor,
        challenge_start: torch.Tensor,
        challenge_word: torch.Tensor,
        challenge_word_mask: torch.Tensor,
        challenge_outcome: torch.Tensor,
        challenge_mask: torch.Tensor,
        *,
        shuffle_challenges: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compile source-only generator tables; the late query is not accepted."""

        config = self.config
        if challenge_word.shape != (
            observation_generator.shape[0],
            config.maximum_challenges,
            config.maximum_word_length,
        ):
            raise PresentedAlgebraError("challenge word geometry differs")
        if challenge_word_mask.shape != challenge_word.shape:
            raise PresentedAlgebraError("challenge word mask geometry differs")
        if challenge_start.shape != challenge_outcome.shape or (
            challenge_start.shape != challenge_mask.shape
        ):
            raise PresentedAlgebraError("challenge outcome geometry differs")

        base, row_seen, output_seen = self._base_tables(
            observation_generator,
            observation_input,
            observation_output,
            observation_mask,
            generator_mask,
        )
        candidates = self._candidate_tables(
            base, row_seen, output_seen, generator_mask
        )
        used_outcome = (
            challenge_outcome.roll(1, 0)
            if shuffle_challenges
            else challenge_outcome
        )
        predictions = []
        for candidate in range(config.candidates):
            predictions.append(
                _apply_word(
                    candidates[:, candidate],
                    challenge_start,
                    challenge_word,
                    challenge_word_mask,
                )
            )
        predicted = torch.stack(predictions, 1)
        mismatches = predicted.ne(used_outcome[:, None])
        mismatches = (
            mismatches & challenge_mask[:, None]
        ).sum(-1)
        selected = mismatches.argmin(-1)
        batch_index = torch.arange(candidates.shape[0], device=candidates.device)
        selected_tables = candidates[batch_index, selected]
        sorted_mismatch = mismatches.sort(-1).values
        margin = sorted_mismatch[:, 1] - sorted_mismatch[:, 0]
        exact = mismatches[batch_index, selected].eq(0)
        return selected_tables, candidates, mismatches, exact, margin

    def __call__(
        self,
        observation_generator: torch.Tensor,
        observation_input: torch.Tensor,
        observation_output: torch.Tensor,
        observation_mask: torch.Tensor,
        generator_mask: torch.Tensor,
        challenge_start: torch.Tensor,
        challenge_word: torch.Tensor,
        challenge_word_mask: torch.Tensor,
        challenge_outcome: torch.Tensor,
        challenge_mask: torch.Tensor,
        query_start: torch.Tensor,
        query_word: torch.Tensor,
        query_word_mask: torch.Tensor,
        *,
        shuffle_challenges: bool = False,
        lineage_swap: bool = False,
    ) -> PresentedAlgebraResult:
        selected, candidates, mismatches, exact, margin = self.compile(
            observation_generator,
            observation_input,
            observation_output,
            observation_mask,
            generator_mask,
            challenge_start,
            challenge_word,
            challenge_word_mask,
            challenge_outcome,
            challenge_mask,
            shuffle_challenges=shuffle_challenges,
        )
        used = selected.roll(1, 0) if lineage_swap else selected
        answer = _apply_word(used, query_start, query_word, query_word_mask)
        logits = F.one_hot(answer, self.config.carrier_size).float().mul(20.0)
        return PresentedAlgebraResult(
            answer_logits=logits,
            answer=answer,
            selected_candidate=mismatches.argmin(-1),
            selected_tables=selected,
            candidate_tables=candidates,
            candidate_mismatches=mismatches,
            challenge_exact=exact,
            selection_margin=margin,
        )
