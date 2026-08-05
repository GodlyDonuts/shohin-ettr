#!/usr/bin/env python3
"""Exact CPU contract tests for DIVERGE-v0."""

from __future__ import annotations

import ast
import inspect
import unittest

import diverge_v0_reference
from diverge_delayed_board import build_delayed_board, build_delayed_episode
from diverge_v0 import (
    ANSWER,
    ABSTAIN,
    OVERFLOW,
    DivergeContractError,
    FaultLine,
    Guard,
    GuardedPatch,
    HardFactor,
    Literal,
    PacketCaps,
    Query,
    SupportFactor,
    TypedCell,
    TypedState,
    TypedTransaction,
    append_verified_nogood,
    assignment_mass,
    account_packet,
    build_packet,
    certify_binary_option_evidence,
    commuting_patch_schedule,
    enumerate_assignments,
    execute_packet,
    merge_certified_classes,
    materialized_world_bytes,
    named_commitment,
    packet_bytes,
    packet_commitment,
    query_execution,
    structural_merge_classes,
)
from diverge_v0_reference import (
    compare_execution,
    reference_behavioral_classes,
    reference_query,
    verify_nogood,
)


def _remap_guard(guard: Guard, remap: dict[int, int]) -> Guard:
    return Guard(
        tuple(Literal(remap[item.variable_id], item.option) for item in reversed(guard.literals))
    )


def _permuted_equivalent(packet):
    remap = {
        variable.variable_id: 100 + len(packet.variables) - variable.variable_id
        for variable in packet.variables
    }
    variables = tuple(
        FaultLine(remap[item.variable_id], item.options, item.provenance)
        for item in reversed(packet.variables)
    )
    hard = tuple(
        HardFactor(
            tuple(remap[variable] for variable in reversed(item.scope)),
            tuple(tuple(reversed(row)) for row in reversed(item.allowed)),
            item.provenance,
        )
        for item in reversed(packet.hard_factors)
    )
    support = tuple(
        SupportFactor(
            tuple(remap[variable] for variable in reversed(item.scope)),
            tuple((tuple(reversed(row)), mass) for row, mass in reversed(item.masses)),
            item.provenance,
        )
        for item in reversed(packet.support_factors)
    )
    patches = tuple(
        GuardedPatch(
            item.index,
            _remap_guard(item.guard, remap),
            item.transaction,
            item.provenance,
        )
        for item in reversed(packet.patches)
    )
    return build_packet(
        source_commitment=packet.source_commitment,
        shared_state=packet.shared_state,
        variables=variables,
        hard_factors=hard,
        support_factors=support,
        patches=patches,
        caps=packet.caps,
    )


class DivergeV0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = build_delayed_board()

    def test_board_covers_frozen_worlds_depths_and_ontology(self) -> None:
        worlds = {episode.represented_worlds for episode in self.board}
        self.assertEqual(worlds, {2, 4, 8, 16, 32, 64})
        self.assertEqual(
            {episode.ontology for episode in self.board},
            {"register-workshop", "parcel-relation", "signal-routing"},
        )
        by_split = {
            split: {item.command_depth for item in self.board if item.split == split}
            for split in {"calibration", "development", "confirmation"}
        }
        self.assertEqual(by_split["calibration"], {2, 3, 4})
        self.assertEqual(by_split["development"], {5, 6})
        self.assertEqual(by_split["confirmation"], {9})
        self.assertTrue(all(item.initial_top1 != item.gold_assignment for item in self.board))
        self.assertTrue(
            all(item.gold_assignment in enumerate_assignments(item.packet) for item in self.board)
        )

    def test_candidate_matches_independent_enumerator_everywhere(self) -> None:
        for episode in self.board:
            with self.subTest(episode=episode.episode_id):
                candidate = execute_packet(episode.packet)
                report = compare_execution(candidate, episode.packet)
                self.assertTrue(report.exact, report.mismatches)
                self.assertEqual(
                    query_execution(candidate, episode.invariant_query),
                    reference_query(episode.packet, episode.invariant_query),
                )
                self.assertEqual(
                    query_execution(candidate, episode.underdetermined_query),
                    reference_query(episode.packet, episode.underdetermined_query),
                )

    def test_independent_assessor_does_not_call_candidate_semantics(self) -> None:
        tree = ast.parse(inspect.getsource(diverge_v0_reference))
        forbidden = {
            "enumerate_assignments",
            "assignment_mass",
            "apply_transaction",
            "execute_packet",
            "read_query",
            "query_execution",
            "structural_merge_classes",
        }
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertFalse(forbidden & called)

    def test_verified_nogood_recovers_without_gold_support_loss(self) -> None:
        for episode in self.board:
            with self.subTest(episode=episode.episode_id):
                initial = query_execution(
                    execute_packet(episode.packet), episode.underdetermined_query
                )
                self.assertEqual(initial.disposition, ABSTAIN)
                verification = verify_nogood(
                    episode.packet,
                    guard=episode.evidence.reject_guard,
                    evidence_commitment=episode.evidence.evidence_commitment,
                    valid_assignments=episode.evidence.valid_assignments,
                )
                self.assertTrue(verification.accepted)
                assert verification.nogood is not None
                refined = append_verified_nogood(episode.packet, verification.nogood)
                remaining = set(enumerate_assignments(refined))
                self.assertEqual(remaining, set(episode.evidence.valid_assignments))
                self.assertIn(episode.gold_assignment, remaining)
                sensitive = query_execution(execute_packet(refined), episode.sensitive_query)
                self.assertEqual(sensitive.disposition, ANSWER)
                self.assertEqual(sensitive, reference_query(refined, episode.sensitive_query))
                self.assertEqual(sensitive.answer, 13)

    def test_invalid_nogood_is_rejected_by_independent_verifier(self) -> None:
        episode = self.board[-1]
        primary = episode.evidence.reject_guard.literals[0].variable_id
        invalid = verify_nogood(
            episode.packet,
            guard=Guard((Literal(primary, 1),)),
            evidence_commitment=episode.evidence.evidence_commitment,
            valid_assignments=episode.evidence.valid_assignments,
        )
        self.assertFalse(invalid.accepted)
        self.assertEqual(invalid.reason, "would-remove-valid-world")
        self.assertIsNone(invalid.nogood)

    def test_delayed_option_evidence_binds_from_sealed_commitments_only(self) -> None:
        episode = self.board[-1]
        primary = episode.evidence.reject_guard.literals[0].variable_id
        confirmed = 1
        certificate = certify_binary_option_evidence(
            episode.packet,
            option_commitment=episode.packet.variables[primary].options[confirmed],
            evidence_commitment=episode.evidence.evidence_commitment,
        )
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertEqual(certificate.variable_id, primary)
        self.assertEqual(certificate.confirmed_option, confirmed)
        self.assertEqual(certificate.nogood.guard, episode.evidence.reject_guard)
        verification = verify_nogood(
            episode.packet,
            guard=certificate.nogood.guard,
            evidence_commitment=certificate.nogood.evidence_commitment,
            valid_assignments=episode.evidence.valid_assignments,
        )
        self.assertTrue(verification.accepted)
        self.assertIsNone(
            certify_binary_option_evidence(
                episode.packet,
                option_commitment=named_commitment("unknown-option", "absent"),
                evidence_commitment=episode.evidence.evidence_commitment,
            )
        )

    def test_query_invariance_abstention_and_source_sealing(self) -> None:
        for episode in self.board:
            receipt = execute_packet(episode.packet)
            invariant = query_execution(receipt, episode.invariant_query)
            uncertain = query_execution(receipt, episode.underdetermined_query)
            self.assertEqual(invariant.disposition, ANSWER)
            self.assertEqual(invariant.answer, 104)
            self.assertEqual(uncertain.disposition, ABSTAIN)
            self.assertGreater(len(uncertain.marginals), 1)
            self.assertNotIn(episode.source_text.encode("utf-8"), packet_bytes(episode.packet))

    def test_canonical_bytes_ignore_ids_and_insertion_order(self) -> None:
        episode = self.board[-1]
        equivalent = _permuted_equivalent(episode.packet)
        self.assertEqual(packet_bytes(equivalent), packet_bytes(episode.packet))
        self.assertEqual(packet_commitment(equivalent), packet_commitment(episode.packet))

    def test_runtime_overflow_is_sticky_and_fail_closed(self) -> None:
        variable = FaultLine(
            7,
            (
                named_commitment("test-option", "zero"),
                named_commitment("test-option", "one"),
            ),
            named_commitment("test-variable", "overflow"),
        )
        packet = build_packet(
            source_commitment=named_commitment("test-source", "overflow"),
            shared_state=TypedState((TypedCell(0, 0, 7),)),
            variables=(variable,),
            patches=(
                GuardedPatch(
                    0,
                    Guard(),
                    TypedTransaction("ADD_VALUE", (0, 7)),
                    named_commitment("test-patch", "overflow-0"),
                ),
                GuardedPatch(
                    1,
                    Guard(),
                    TypedTransaction("ADD_VALUE", (0, 7)),
                    named_commitment("test-patch", "overflow-1"),
                ),
            ),
            caps=PacketCaps(max_integer_bits=4),
        )
        receipt = execute_packet(packet)
        self.assertTrue(receipt.overflow)
        self.assertEqual(query_execution(receipt, Query("READ_VALUE", (0,))).disposition, OVERFLOW)
        self.assertEqual(reference_query(packet, Query("READ_VALUE", (0,))).disposition, OVERFLOW)
        self.assertTrue(compare_execution(receipt, packet).exact)

        episode = self.board[2]
        overflowed = build_packet(
            source_commitment=episode.packet.source_commitment,
            shared_state=episode.packet.shared_state,
            variables=episode.packet.variables,
            support_factors=episode.packet.support_factors,
            patches=episode.packet.patches,
            caps=PacketCaps(max_variables=1, max_patches=32),
        )
        self.assertTrue(overflowed.overflow)
        self.assertFalse(enumerate_assignments(overflowed))
        verification = verify_nogood(
            overflowed,
            guard=Guard((Literal(0, 0),)),
            evidence_commitment=named_commitment("test-evidence", "overflow"),
            valid_assignments=((1,),),
        )
        self.assertFalse(verification.accepted)

    def test_certified_merging_preserves_mass_and_rejects_cross_state_merge(self) -> None:
        episode = self.board[-1]
        after = len(episode.packet.patches)
        structural = structural_merge_classes(episode.packet, after_patches=after)
        behavioral = reference_behavioral_classes(
            episode.packet,
            after_patches=after,
            queries=(episode.sensitive_query, episode.invariant_query),
        )
        self.assertEqual({frozenset(x) for x in structural}, {frozenset(x) for x in behavioral})
        merged = merge_certified_classes(
            episode.packet,
            after_patches=after,
            certified_classes=behavioral,
        )
        self.assertEqual(
            sum(item.mass for item in merged),
            sum(assignment_mass(episode.packet, item) for item in enumerate_assignments(episode.packet)),
        )
        with self.assertRaises(DivergeContractError, msg="unsafe cross-state merge accepted"):
            merge_certified_classes(
                episode.packet,
                after_patches=after,
                certified_classes=(enumerate_assignments(episode.packet),),
            )

    def test_factorization_has_a_measured_sharing_advantage_at_64_worlds(self) -> None:
        episode = self.board[-1]
        receipt = execute_packet(episode.packet)
        accounting = account_packet(episode.packet, receipt)
        self.assertEqual(accounting.represented_worlds, 64)
        self.assertGreater(accounting.duplicated_transactions, accounting.unique_transactions)
        self.assertGreater(accounting.shared_transactions, 0)
        self.assertLess(accounting.packet_bytes, accounting.materialized_world_bytes)
        self.assertEqual(
            sum(materialized_world_bytes(episode.packet, world) for world in receipt.worlds),
            accounting.materialized_world_bytes,
        )

    def test_disjoint_commuting_schedule_preserves_states_and_ordered_semantics(self) -> None:
        for episode in self.board:
            original = execute_packet(episode.packet)
            scheduled = execute_packet(episode.packet, commute_disjoint=True)
            self.assertEqual(
                tuple(world.record() for world in scheduled.worlds),
                tuple(world.record() for world in original.worlds),
            )
            self.assertLessEqual(scheduled.unique_transactions, original.unique_transactions)

        episode = self.board[0]
        primary = [
            patch
            for patch in commuting_patch_schedule(episode.packet.patches)
            if 0 in {literal.variable_id for literal in patch.guard.literals}
        ]
        option_zero = [
            patch.transaction.opcode
            for patch in primary
            if patch.guard == Guard((Literal(0, 0),))
        ]
        option_one = [
            patch.transaction.opcode
            for patch in primary
            if patch.guard == Guard((Literal(0, 1),))
        ]
        self.assertEqual(option_zero, ["ADD_VALUE", "SWAP_VALUE"])
        self.assertEqual(option_one, ["SWAP_VALUE", "ADD_VALUE"])

    def test_hard_factor_support_is_exact(self) -> None:
        episode = build_delayed_episode(
            split="calibration",
            ontology="register-workshop",
            renderer="factor-check",
            width=2,
            serial=999,
        )
        variables = episode.packet.variables
        factor = HardFactor(
            (0, 1),
            ((0, 0), (1, 1)),
            named_commitment("test-hard-factor", "equal"),
        )
        constrained = build_packet(
            source_commitment=episode.packet.source_commitment,
            shared_state=episode.packet.shared_state,
            variables=variables,
            hard_factors=(factor,),
            support_factors=episode.packet.support_factors,
            patches=episode.packet.patches,
            caps=episode.packet.caps,
        )
        self.assertEqual(enumerate_assignments(constrained), ((0, 0), (1, 1)))
        self.assertTrue(compare_execution(execute_packet(constrained), constrained).exact)


if __name__ == "__main__":
    unittest.main()
