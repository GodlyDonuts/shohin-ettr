# R12 ETTR-IL-v2 Phase-1 Architecture Handoff

**Protocol:** `R12-ETTR-IL-v2`
**Phase:** 1, architecture and no-update systems qualification
**Decision:** `r12_ettr_il_v2_phase1_architecture_frozen_phase2_requires_explicit_user_authorization`

## 1. What is ready

Shohin now has one complete, trainable architecture for testing
architecture-native, source-deleted systematic reasoning:

| Component | Parameters |
|---|---:|
| Protected Shohin base | 125,081,664 |
| Endogenous theory compiler | 21,466,377 |
| Generic transaction reactor | 29,757,217 |
| Source-deleted query reader | 16,474,177 |
| Added ETTR parameters | 67,697,771 |
| Complete system | 192,779,435 |
| Remaining below the 200M cap | 7,220,565 |

The architecture maps a raw `WORLD` into a hard categorical packet, applies a
separately disclosed `COMMAND` through a 64-position recurrent transaction
reactor, then answers a separately disclosed `QUERY` from the sealed terminal
packet. Raw WORLD tokens, raw COMMAND tokens, base residuals, and KV state do
not cross the stage interfaces.

The fixed neural state geometry is:

- 64 packet slots;
- 8 slot types;
- 16 relation roles;
- 256 categorical values;
- 256 relation edges;
- 64 recurrent transaction positions; and
- exact model-visible transport widths of 192 WORLD, 96 COMMAND, and 48 QUERY
  tokens.

## 2. What was implemented

Phase 1 includes executable, fail-closed implementations for:

1. typed Horn closure, typed term rewriting, and guarded resource-process
   semantic families;
2. independent primary and replay oracles through dependent depths 1-6;
3. strict surface ASTs and exact parsers;
4. six presentation families:
   `base`, `alpha_reorder`, `alias_split`, `relation_reification`,
   `type_twin`, and `execution_semantics_twin`;
5. lossless token-native transport under the fixed 192/96/48 widths;
6. exact projection into native ETTR packets and 64-position transaction
   traces, with independent replay to the terminal packet;
7. deterministic semantic candidate scanning and fail-closed cardinality
   receipts;
8. exact split, fold, quota, invariant-pair, schedule, source, and materialized
   batch validators;
9. deterministic binding derangement and leakage/custody controls;
10. five matched causal arms:
    treatment, state reset, binding derangement, query-only, and a
    parameter/optimizer-family/static-operation-matched dense controller;
11. exact resume, RNG, parameter, token, objective, and zero-update readiness
    validation;
12. physical WORLD/COMMAND/QUERY source deletion in separate processes;
13. a locked 75-run, update-6000 evaluator with 292 preregistered statistical
    endpoints and a 100,000-replicate semantic-core bootstrap; and
14. a no-replace Phase-1 source, test, evidence, tokenizer, parameter, and
    protected-checkpoint freeze receipt.

The dense control is a real alternative recurrent mechanism, not a label in a
receipt. It replaces exactly 27,302,912 reactor parameters while retaining the
common transaction shell, preserves the exact Muon/AdamW ownership totals, and
executes fixed orthogonal replay work to close the protocol's static
loss-connected scalar-product budget.

## 3. Frozen scientific experiment

Each training fold uses two ontologies and withholds the third. The intended
claim-bearing experiment contains five arms, three folds, and five fixed
seeds, or 75 fits. Each fit has:

- 6,000 optimizer updates;
- 24,000 invariant-pair exposures;
- 48,000 semantic-rectangle exposures;
- 192,000 causal-rectangle exposures;
- 768,000 query-row exposures;
- 405,504,000 charged encoded tokens; and
- 67,697,771 trainable ETTR parameters with the protected base frozen.

Only update 6,000 is claim-bearing. There is no early stopping, best-checkpoint
selection, seed removal, retry, decoder selection, or threshold tuning.

The evaluator checks transfer across unseen rules, composition depths,
renderers, combinations of all three shifts, and a completely withheld
ontology. A positive result additionally requires causal superiority to all
four controls, independent source-deleted execution, and one separately
authorized confirmation opening.

## 4. Frozen evidence

The protected checkpoint remains untouched:

```text
step: 300000
sha256: 211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
```

The exact tokenizer is:

```text
path: artifacts/shohin-tok-32k.json
bytes: 2309567
sha256: 87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
```

Deterministic Phase-1 evidence:

| Artifact | SHA-256 |
|---|---|
| Integrated specification audit | `cc98f616da365215d01611f16c6ae84f7c3e42cdddbe7a5baa543f1eccb524ef` |
| Fixed token-transport capacity audit | `d9021b128f08fc431063a98e04b4c3bbc60f838696c6889dfc51edbdedc9ab43` |
| Three-ontology end-to-end canary | `066f8ee5d7b5d3f0a1852fb4f74f5c5a488146e41712b6ee51dfde995fb829cb` |

The end-to-end canary materializes 16 rows and four causal rectangles for each
ontology, proves exact 192/96/48 transport, and emits a source-free receiving
batch. It is a systems test, not learned-capability evidence.

The canonical Phase-1 freeze receipt is:

```text
path: artifacts/r12/ettr_il_v2_phase1_architecture_freeze.json
sha256: 74eed0408d0328105b4433eadcf818a3ceca07e19edd50ac53f2fb50cb2fbee8
```

That receipt binds:

```text
source files: 37
source inventory sha256: bfcf7be8e95a510b4856b67afa36850e303f129e5843cf24ac3a6ec4df821dbd
test files: 22
test inventory sha256: b505638e7645ad6c59ebd978c6ddecf4a080de4433fcde1e6720ded34b2b939c
evidence inventory sha256: d2fefbbfac990ed2f2a4869c9497b744ded859cfb7005ee5fcfcd82d1d47cf3e
```

Final local verification completed with **350/350 tests passing**: 337
non-arm checks and 13 arm checks. The arm suite includes a real
production-geometry dense-control forward. Ruff, Python byte compilation,
specification regeneration, canonical-artifact comparison, hash checks, and
`git diff --check` all passed. Separately, the locked evaluator completed a
full synthetic production-path rehearsal of all 100,000 bootstrap replicates
and 292 endpoints and returned `OPEN_CONFIRMATION`; it did not open real
scores.

## 5. Explicit non-claims

Phase 1 performed zero weight updates. It did not:

- continue Shohin pretraining;
- fit ETTR or any control arm;
- modify or rewrite the protected checkpoint;
- materialize the literal 20,736-rectangle production population;
- open development or confirmation scores;
- establish that the newly initialized parameters already reason; or
- establish native general reasoning.

The implementation demonstrates that the architecture, transport,
materialization, controls, custody, budgets, and evaluator are mechanically
defined and testable. Capability remains an empirical Phase-2 question.

## 6. Phase-2 entry order

Phase 2 must preserve this order:

1. Generate the literal candidate population and recompute every ontology,
   presentation, and execution-semantics-twin outcome.
2. Certify cardinality and select the exact quotas without replacement.
3. Materialize all selected batches, freeze them, reload them independently,
   and replay every transaction trace.
4. Run the complete raw-row, semantic, token-sequence, normalized 13-gram,
   graph-isomorphism, opaque-name, path, length, mask, ordering, and metadata
   leakage audits.
5. Validate exact split disjointness, packet sufficiency, invariant pairs,
   schedules, derangements, parameters, optimizer ownership, token counts,
   and static operation budgets.
6. Execute one zero-update readiness batch for every arm and verify exact
   resume/RNG receipts.
7. Request explicit user authorization.
8. Fit only after authorization.

Any cardinality, transport, leakage, source-deletion, parameter, budget, or
readiness failure blocks fitting. It is invalid evidence rather than a
negative reasoning result.

## 7. Remaining scientific risks

- The production population has not been enumerated. Some theory/depth/stratum
  quota may fail and require a protocol revision before fitting.
- Execution-semantics twins require fresh oracle execution after
  transformation; mechanical surface invertibility is not sufficient.
- The dense matched control is deliberately expensive because it closes the
  native reactor's static operation budget. Its H100 runtime must be measured
  before the 75-fit campaign.
- Same-account custody proves mechanics, not independent scientific
  confirmation. A public positive claim still needs independently owned
  authorization and immutable opening records.
- A passing synthetic protocol would establish bounded systematic transfer,
  not unrestricted natural-language reasoning. A failing protocol must be
  localized using the preregistered component precedence rather than repaired
  post hoc.

## 8. Authority

The normative protocol is
`R12_ETTR_ISOLATED_LEARNABILITY_PREREG_V2.md` plus its four exact component
specifications. This handoff summarizes those frozen contracts. Where wording
differs, the hash-bound preregistration and executable validators control.
