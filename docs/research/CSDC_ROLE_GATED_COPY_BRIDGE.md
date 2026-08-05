# CSDC Role-Gated Copy Bridge

Status: gate passed; promoted controlled rendered-source CSDC baseline.

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

## Result

Immutable job `739448` ran commit `6359453` on one H100 for 1,500 updates,
192,000 examples, and 461.997 training seconds. The copy parser contains
71,622 trainable parameters, 4,290 fewer than the closed summary decoder. The
seed-47 row-local compiler and every CSDC reasoning component stayed frozen.

| Metric | Development | Held-out renderer |
|---|---:|---:|
| End-to-end exact answer | **99.593%** | **99.723%** |
| Typed CSDC oracle | 99.593% | 99.723% |
| Complete challenge tuple | **100.000%** | **100.000%** |
| Complete ordered word | **100.000%** | **100.000%** |
| Selected presentation | 99.007% | 99.284% |
| Shuffled outcome answer | 53.630% | 53.630% |
| Swapped lineage answer | 13.623% | 13.346% |

All six cohorts in both splits exceed 98.9% answers; every complete challenge
tuple is exact. The learned rendered-source system exactly matches typed-oracle
CSDC on both aggregate splits. Shuffling copied challenge outcomes costs
45.963/46.094 points, while swapping the selected whole lineage costs
85.970/86.377 points. The copied source fields and one coherent committed
presentation are therefore both causally necessary.

This passes every frozen condition. It establishes a model-owned compiler from
controlled rendered source records into CSDC's symbolic challenge interface.
The winning principle is:

`learn semantic source roles -> preserve token identity and order by copying ->
enumerate residual complete hypotheses -> falsify with source evidence ->
commit one lineage -> execute the late query`.

The result does not establish unrestricted natural-language parsing, external
knowledge, or public-benchmark reasoning. The next scaling step must test a
broader lexical and compositional language interface while preserving this
copy/commit boundary; it must not return to summary-vector sequence
regeneration.

Runtime SHA256SUMS SHA-256 is
`145c87d760e2c7ee3aee0433c2609ef5849b3d31feb2c7d0c763aeb42dc9afa6`.
Report SHA-256 is
`808f50e6e3a1026761f7fa0e29aa022346bde6befd419e9051e719fd9448ea37`;
checkpoint SHA-256 is
`55b5ef79110625f383f6800ac89a20dba9d0a1420bd554fd928ee70f42fdf956`.

Decision:
`promote_role_gated_copy_csdc_as_the_controlled_rendered_source_baseline_then_test_broader_lexical_semantics_without_changing_the_reasoning_core`.
