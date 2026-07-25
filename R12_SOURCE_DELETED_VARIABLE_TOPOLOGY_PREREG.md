# R12 Source-Deleted Variable-Topology Qualification

## Decision

The learned global semantic-partition compiler passed this frozen gate on
2026-07-25:

`global_semantic_partition_passes_variable_topology_gate`

Treatment reached 360/360 across five seeds and all three
leave-one-family-out folds. Same-weight direction swap reached 152/360;
negating the learned global key partition or learned query roles reached
0/360. Full evidence:
`R12_VARIABLE_TOPOLOGY_SEMANTIC_COMPILER_RESULT.md`.

This board authorizes no continuation pretraining and changes nothing about
the protected step-300k checkpoint.

## Frozen Board

- families: affine modular, bitwise rotate/xor, and arbitrary permutation
- state cardinalities: 4, 8, and 16
- action counts: 2 through 5
- fitting topologies: 4x3, 8x3, 8x5, and 16x3
- development topology: 16x5
- incidence-collision topologies: 4x2 and 8x4
- renderers: five fitting forms and one held-out passive form
- development cells: unseen law, longer composition, unseen topology,
  incidence collision, held-out renderer, and joint collision-plus-renderer
- every episode exposes a complete permutation transition table
- every law is unique across the frozen board

For complete permutation tables, each action key occurs `cardinality` times
and each state key occurs `2 * action_count` times. In the 4x2 and 8x4 cells
these frequencies are equal. The successful fixed-incidence structural typer
therefore cannot distinguish semantic key type in those cells.

## Independent Audit

The frozen receipt contains 132 rows:

- exact execution after source deletion: 132/132
- role-neutral opaque-key rows: 132/132
- family-name leaks: 0
- incidence-separable rows: 108
- incidence-ambiguous collision rows: 24
- collision rows with exactly equal key frequencies: 24/24
- law-swap answer changes: 113/132
- action-reversal answer changes: 51/78 eligible
- renderer orbits exact: 21/21
- renderer orbits compiling to identical packets: 21/21

Receipt:
`artifacts/r12/source_deleted_variable_topology_board_v1/audit_receipt.json`

Receipt SHA-256:
`6711d17d4b21e0f034852e403d75ad20ffe59eb99ce05c96dc8784820adb9197`

Candidate manifest SHA-256:
`c569fd1b7dd409b1c69aab0e383de43a05416df4c8e62674605ce9730c3ac86f`

Supervisor manifest SHA-256:
`54b56fc9ba7f9e5115dcd45fedcb5fad15242ea7b54951fb740058320a3b9cdf`

## Candidate Contract

A claim-bearing candidate must:

1. compile one sealed anonymous machine before source bytes are deleted;
2. import no exact board parser;
3. make zero candidate-time oracle, search, or verifier calls;
4. use one shared learned mechanism across all families and topologies;
5. infer state/action type in collision rows without frequency labels;
6. infer source/target direction under the held-out renderer;
7. execute the late action word only from the sealed packet;
8. remain below 200,000,000 unique complete-system parameters; and
9. preserve the immutable protected Shohin checkpoint.

Preparation may use the exact parser only to construct counted supervision.
All such calls must be reported.

## Matched Controls

The claim-bearing protocol trained one model per fold and intervened on the
same weights at inference:

- swap learned source/target direction;
- negate the episode-global learned state/action key scores; and
- swap learned query start/action roles.

The originally proposed type-channel swap was rejected because occurrence
evidence could cancel without changing the selected global partition. The
direct score negation removes that ambiguity.

The source-deleted candidate must also survive law swaps and action-order
reversal. A host exact compiler is an upper bound only and cannot be counted
as candidate reasoning.

## Promotion Gate

Across five seeds and all three leave-one-family-out folds:

- fitting exactness: at least 95%;
- aggregate development exactness: at least 85%;
- every family and every development cell: at least 80%;
- joint collision-plus-renderer cell: at least 80%;
- every treatment-control seed/fold direction positive;
- treatment margin over both direction-shuffled and type-shuffled controls:
  at least 10 percentage points; and
- zero candidate-time oracle/search/verifier calls.

Passing this gate would establish stronger systematic anonymous-machine
compilation. It still would not, by itself, establish general reasoning
because transition tables are complete and all tasks share a finite-machine
ontology. The next boundary after a pass would be incomplete-law induction
and natural-language post-training, not more proxy-specific recurrence.

## Final Conclusion

The learned compiler infers latent semantic key types even when incidence
statistics are uninformative. This removes the fixed-topology shortcut and is
a genuine bounded systematic-compilation advance.

It remains a complete-table finite-machine sidecar rather than Shohin-native
general reasoning. The next gate is sparse latent-law induction under a new
ontology and process-level source deletion. Do not resume pretraining or spend
more compute on this solved board.
