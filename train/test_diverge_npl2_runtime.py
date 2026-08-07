#!/usr/bin/env python3
"""Contract tests for the source-deleted DIVERGE-NPL2 runtime."""

from __future__ import annotations

import hashlib
import unittest

from diverge_npl1_data import natural_public_record, render_feedback
from diverge_pl1_data import build_episode
from diverge_pl1_runtime import run_episode
from diverge_npl2_runtime import (
    DecodedEvidence,
    run_natural_episode,
    typed_episode_from_public,
)


def oracle_semantics(public, episode):
    evidence = {}
    for plan in public["feedback_plan"]:
        attempt = int(plan["attempt"])
        branch = int(plan["branch"])
        depth = len(episode.acquisition[attempt].symbols)
        for code in (0, *range(2, depth + 2)):
            text = render_feedback(plan, code)
            evidence[(attempt, branch, code)] = DecodedEvidence(
                attempt=attempt,
                target_branch=str(plan["target_branch"]),
                distractor_branch=str(plan["distractor_branch"]),
                certificate_code=code,
                commitment=hashlib.sha256(text.encode("ascii")).hexdigest(),
            )
    selectors = tuple(int(query["register_index"]) for query in public["queries"])
    return evidence, selectors


class NPL2RuntimeTest(unittest.TestCase):
    def test_oracle_semantics_are_exact_pl1_parity(self) -> None:
        episode = build_episode(split="npl2-test", seed=17, serial=0)
        public = natural_public_record(episode)
        typed = typed_episode_from_public(public)
        evidence, selectors = oracle_semantics(public, episode)
        oracle = run_episode(episode, arm="PL1", seed=2026080799)
        natural = run_natural_episode(
            typed,
            episode,
            evidence=evidence,
            query_selectors=selectors,
            arm="PL1",
            candidate_label="NPL2",
            proposal_arm="PL1",
            seed=2026080799,
        )
        self.assertEqual(natural.selected_mapping, oracle.selected_mapping)
        self.assertEqual(natural.policy_state, oracle.policy_state)
        self.assertEqual(natural.transfer_exact, oracle.transfer_exact)
        self.assertEqual(natural.query_exact, 2 * oracle.transfer_exact)
        self.assertEqual(natural.semantic_rejections, 0)

    def test_wrong_semantic_value_fails_closed(self) -> None:
        episode = build_episode(split="npl2-test", seed=17, serial=1)
        public = natural_public_record(episode)
        typed = typed_episode_from_public(public)
        evidence, selectors = oracle_semantics(public, episode)
        evidence = {
            key: DecodedEvidence(
                attempt=item.attempt,
                target_branch=item.distractor_branch,
                distractor_branch=item.target_branch,
                certificate_code=item.certificate_code,
                commitment=item.commitment,
            )
            for key, item in evidence.items()
        }
        result = run_natural_episode(
            typed,
            episode,
            evidence=evidence,
            query_selectors=selectors,
            arm="PL1",
            proposal_arm="PL1",
            seed=2026080799,
        )
        self.assertGreater(result.semantic_rejections, 0)

    def test_protected_owner_mutation_rejects(self) -> None:
        episode = build_episode(split="npl2-test", seed=17, serial=2)
        public = natural_public_record(episode)
        evidence, selectors = oracle_semantics(public, episode)
        with self.assertRaises(RuntimeError):
            run_natural_episode(
                typed_episode_from_public(public),
                episode,
                evidence=evidence,
                query_selectors=selectors,
                arm="PL1",
                proposal_arm="PL1",
                seed=2026080799,
                inject_protected_mutation=True,
            )


if __name__ == "__main__":
    unittest.main()
