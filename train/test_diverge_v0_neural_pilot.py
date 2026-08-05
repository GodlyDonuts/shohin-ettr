#!/usr/bin/env python3
"""Focused tests for the bounded DIVERGE learned component pilot."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from diverge_v0_neural_pilot import (
    CompilerPrediction,
    DivergePilotCompiler,
    _training_batch,
    generate_episode,
    score_episode,
)


def _perfect_prediction(episode) -> CompilerPrediction:
    selected = tuple(record.is_fault_line for record in episode.records)
    programs = tuple(
        tuple(option.program for option in record.options) for record in episode.records
    )
    priors = tuple(
        tuple(option.prior_class for option in record.options)
        for record in episode.records
    )
    evidence_record = next(
        index
        for index, record in enumerate(episode.records)
        if record.record_id == episode.primary_record_id
    )
    evidence_option = next(
        index
        for index, option in enumerate(episode.records[evidence_record].options)
        if option.alias == episode.evidence_alias
    )
    return CompilerPrediction(
        selected,
        programs,
        priors,
        evidence_record,
        evidence_option,
    )


class DivergeNeuralPilotTests(unittest.TestCase):
    def test_primary_gold_is_always_the_low_prior_noncommuting_program(self) -> None:
        for seed in range(50):
            episode = generate_episode(
                seed=seed,
                split="train",
                width=1 + seed % 4,
                renderer=seed % 2,
                ontology="register-workshop",
            )
            primary = next(
                record for record in episode.records if record.record_id == episode.primary_record_id
            )
            gold = primary.options[primary.gold_option]
            self.assertEqual(gold.program, 1)
            self.assertEqual(gold.prior_class, 1)

    def test_perfect_compiler_recovers_only_with_conflict_revision(self) -> None:
        episode = generate_episode(
            seed=20260805,
            split="confirmation",
            width=6,
            renderer=3,
            ontology="signal-routing",
        )
        prediction = _perfect_prediction(episode)
        model = DivergePilotCompiler()
        with mock.patch(
            "diverge_v0_neural_pilot.predict_episode",
            return_value=prediction,
        ):
            result = score_episode(model, episode, torch.device("cpu"))
        self.assertEqual(result["gold_support_recalled"], 1)
        self.assertEqual(result["packet_exact"], 1)
        self.assertEqual(result["evidence_binding_exact"], 1)
        self.assertEqual(result["A_single"], 0)
        self.assertEqual(result["F_no_conflict"], 0)
        self.assertEqual(result["G_diverge"], 1)
        self.assertEqual(result["G_joint_packet_answer"], 1)

    def test_missing_fault_line_is_a_support_failure(self) -> None:
        episode = generate_episode(
            seed=20260806,
            split="development",
            width=5,
            renderer=2,
            ontology="parcel-relation",
        )
        prediction = _perfect_prediction(episode)
        selected = list(prediction.selected)
        primary = next(
            index
            for index, record in enumerate(episode.records)
            if record.record_id == episode.primary_record_id
        )
        selected[primary] = False
        damaged = CompilerPrediction(
            tuple(selected),
            prediction.programs,
            prediction.priors,
            prediction.evidence_record,
            prediction.evidence_option,
        )
        model = DivergePilotCompiler()
        with mock.patch(
            "diverge_v0_neural_pilot.predict_episode",
            return_value=damaged,
        ):
            result = score_episode(model, episode, torch.device("cpu"))
        self.assertEqual(result["gold_support_recalled"], 0)
        self.assertEqual(result["G_joint_packet_answer"], 0)

    def test_one_training_update_is_finite(self) -> None:
        torch.manual_seed(7)
        episodes = [
            generate_episode(
                seed=100 + index,
                split="train",
                width=1 + index,
                renderer=index % 2,
                ontology="register-workshop",
            )
            for index in range(3)
        ]
        model = DivergePilotCompiler(width=48, char_width=24)
        loss, metrics = _training_batch(episodes, model, torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(metrics["loss"], 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
