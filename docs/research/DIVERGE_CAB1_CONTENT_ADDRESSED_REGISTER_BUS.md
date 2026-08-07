# DIVERGE-CAB1: Content-Addressed Register Bus

**Status:** frozen before corpus materialization or neural scoring

**Parents:** confirmed NCP1 command owner, confirmed EAL2 temporal owner, closed JRB1 attribution

**Objective:** replace canonical register coordinates with one episode-local content-addressed basis

## Why this successor exists

JRB1 learned evidence and natural initial-state binding exactly and preserved
exact recurrent execution, but its pooled query reader reached only 96.58%.
JRB1's purported shuffled-table negative also exposed a representational
symmetry: every owner learned the same global coordinate swap, so canonical
state tuples changed while semantic answers remained correct. A coherent basis
change is not an information-destroying intervention.

CAB1 removes the artificial canonical-coordinate requirement. Every learned
register owner predicts a position in a randomly ordered, episode-local table.
Law packets, initial state, recurrent execution, and late query reading stay in
that table-relative basis. Canonical coordinates exist only inside the
independent assessor. The query path replaces whole-sentence mean pooling with
token-level pointer evidence aggregated by `logsumexp`.

## Frozen mechanism

The trainable owner is a shared byte encoder with dynamic register candidates:

1. numeric mention spans point to table positions for evidence and initial
   state;
2. each query token scores both table entries;
3. a learned scalar gate selects informative query tokens;
4. token evidence is aggregated into one query-to-position decision;
5. the already-qualified EAL2 temporal owner and NCP1 command owner remain
   bit-identical;
6. all downstream state is represented and executed in the selected table
   basis.

The candidate sees raw natural evidence, raw commands, natural initial-state
text, natural late queries, operation aliases, and the two-entry register
table. It does not see canonical register identities, numeric role labels,
typed initial states, typed operation sequences, terminal states, or answers.

Exact numeric-span boundaries, the explicit two-entry table, exact law support,
the bounded Z/97 executor, and output comparison remain engineered scaffolds.
CAB1 tests basis-coherent semantic binding; it is not an open-domain reasoning
claim.

## Frozen corpus

- Training seed: `2026080821`
- Development seed: `2026080822`
- Conditional confirmation seeds: `2026080823` through `2026080827`
- Training: 100,000 generated records
- Development and each confirmation board: 256 episodes
- Each episode: 24 evidence transitions, 16 noncommuting programs at depths
  12--32, and 32 late natural queries
- Table order is deterministic but pseudorandom and balanced independently of
  canonical identity
- Opaque names and source texts are disjoint from JRB1 training and every JRB1
  development/confirmation board; JRB1's signed transitive overlap receipt
  binds the older EAL2/NCP1 lineage
- Every artifact is deterministically regenerated and hash-bound before model
  access

Confirmation files are materialized and sealed with the same generator and
evaluator before development scoring. They may open only after an unchanged
development PASS.

## Frozen optimization

Two matched 290k-parameter-scale arms start from identical initialization and
receive the same sampled rows, update count, batch size, and optimizer:

- **Treatment:** candidates are the two source-owned register names in random
  table order.
- **Decoy-table control:** candidates are two equally shaped episode-local
  names absent from evidence, initial state, and query while targets and all
  other inputs remain unchanged.

Each arm receives exactly 1,000 AdamW updates, batch 128, learning rate 0.001,
weight decay `1e-4`, and the unweighted sum of evidence, initial, and query
cross-entropies.

## Frozen development arms

1. **Treatment:** ordinary source and table.
2. **Renamed:** unseen register names consistently replace source and table.
3. **Whole-table permutation:** evidence, initial, state, and query owners all
   receive the reversed table. This is a positive equivariance test.
4. **Cross-owner permutation:** only evidence compilation receives the reversed
   table; initial state and query remain in the ordinary basis. This is the
   causal coherence-breaking control.
5. **Source scrub:** source register names are replaced by unrelated names while
   the ordinary candidate table remains.
6. **Decoy-table model:** the independently trained matched control is evaluated
   on the ordinary board.

No arm averages fields across bases. The executor sees only the committed law
packet, table-relative initial state, NCP1 operation sequence, and table-relative
late query.

## Conjunctive gate

CAB1 passes development only when every condition is true:

- qualified EAL2 and NCP1 hashes and reports match;
- command exactness is at least 99%, with at least 95% at every depth;
- treatment, renamed, and whole-table-permutation evidence binding, temporal
  assignment, complete roles, initial binding, query binding, and law commit
  are each at least 99%;
- those three positive arms reach at least 99% terminal-state and answer
  exactness, with at least 95% terminal exactness at every depth;
- cross-owner permutation reaches at most 5% canonical terminal-state exactness
  and at most 55% answer exactness;
- source scrub reaches at most 30% evidence grouping, 35% initial binding, 55%
  query binding, 20% state exactness, and 35% answer exactness;
- the decoy-table model stays below those same calibrated ceilings;
- treatment/control initialization, data, charged examples, schedule, and
  configuration match exactly;
- checkpoint/report/data/runtime hashes match, parent weights remain
  bit-identical, typed initial/query carriers are absent, and runtime source has
  no exact register-name scanner.

Every one of the five confirmation boards must independently pass the unchanged
evaluator. A development failure closes this exact CAB1 rule without width,
duration, seed, threshold, renderer, or loss variants. One read-only attribution
may locate the failed owner but cannot rescue the result.

## Claim boundary and next decision

A confirmed PASS qualifies one content-addressed coordinate bus that preserves
whole-state coherence across natural evidence, initial state, recurrent
execution, and late query. It does not authorize continuation pretraining or a
frontier-capability claim. The next successor would remove another explicit
scaffold, selected from numeric spans, law support, or the exact executor.

A FAIL means this token-evidence bus does not clear the controlled composition
gate under the frozen budget. Its exact failure is preserved and not tuned.
