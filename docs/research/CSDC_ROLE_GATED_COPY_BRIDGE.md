# CSDC Role-Gated Copy Bridge

Status: gate frozen; implementation, mechanics tests, and Newton CPU smoke
pass; H100 gate pending.

## Hypothesis

The closed semantic bridge identifies records, states, outcomes, and lengths
almost perfectly but loses ordered generator words when it compresses each
record into one summary vector and regenerates up to twelve operators through
independent output heads. CSDC itself remains near exact with typed fields.

The Role-Gated Copy Bridge changes the interface rather than tuning the closed
parser. A token encoder predicts four source roles: `OTHER`, `START`,
`OUTCOME`, and `WORD`. At inference it selects challenge records, copies the
state tokens at the predicted start/outcome positions, and copies every
predicted word token in original source order. It cannot invent or reorder
operator identities through a decoder. The complete copied challenges feed
the frozen CSDC candidate selector and late executor.

This is not claimed as a novel pointer-network primitive. The architectural
test is whether preserving source-token identity at the semantic-to-symbolic
boundary removes CSDC's only measured end-to-end failure while retaining the
one-coherent-lineage reasoning path.

## Fixed controls and information boundary

- identical seed-47 frozen row-local compiler and CSDC reasoner;
- identical generated episodes, renderer templates, development cohorts,
  held-out field order, candidate constructor, evaluator, and interventions;
- supervision only for source record kind and token roles;
- no query, answer, table, selected-presentation, terminal-state, or CSDC
  loss; and
- copied values come only from model-selected input token positions.

## Gate

One seed trains for 1,500 updates at batch 128. It advances only if:

1. development end-to-end exact answer is at least 95% and within five points
   of typed oracle CSDC;
2. development complete challenge tuples and selected presentations are each
   at least 95%;
3. shuffled outcomes and lineage swaps each lose at least 20 points;
4. held-out-renderer exact answer and complete tuples are each at least 90%;
5. held-out answer stays within five points of typed oracle CSDC; and
6. every family/depth cohort reaches at least 90% end-to-end exactness.

Failure closes this copy interface without width, duration, seed, template,
threshold, or loss variants. Passing establishes controlled rendered-source
CSDC, not unrestricted natural-language or public-benchmark reasoning.

## Resource envelope

Focused CPU mechanics and a two-update end-to-end smoke precede one H100 job.
Hard ceiling is one H100-hour; expected use is below 0.25 H100-hour. No long
pretraining or Shohin integration follows from this gate alone.
