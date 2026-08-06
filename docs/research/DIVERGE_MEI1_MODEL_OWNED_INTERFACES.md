# DIVERGE-MEI1: Model-Owned Evidence, Execution, and Query Interfaces

Status: frozen before neural results on 2026-08-06.

## Decision

The accepted DIVERGE-ULC1-HSC1 gate establishes that a frozen learned source
compiler can retain the valid interpretation and that a factorized version
space can recover after delayed evidence. It does not establish model-owned
reasoning: the accepted runtime still receives an assessor-issued effect code,
calls an exact typed host executor, and calls an exact query reader.

MEI1 is one bounded interface gate. It keeps the accepted HSC1 source scores,
K=2 semantic envelope, source seal, coherent lineages, and exact factorized
bookkeeping fixed. It replaces all three semantic host interfaces:

1. delayed evidence is natural-language before/after state evidence interpreted
   by a learned evidence reader;
2. every candidate transaction is applied by one tied learned register
   transition cell; and
3. every late value query is answered by a learned state-query reader.

PCSD and FCPT remain closed. MEI1 is not continuation pretraining, a public
benchmark, or a relaunch of the failed hard HSC1 compiler.

## Causal boundary

```text
WORLD
  -> frozen learned HSC1 K=2 compiler
  -> sealed factorized packet
  -> delayed natural-language probe evidence
  -> learned evidence interpreter
  -> learned shared guarded execution
  -> coherent factorized terminal states
  -> late typed QUERY
  -> learned query reader
```

After the WORLD packet is sealed, candidate execution receives no WORLD token,
WORLD residual, WORLD KV cache, gold parse, answer label, gold program, or exact
successor state. Exact operations may generate supervisor targets and assess
outputs, but the candidate runtime must not import or call `apply_transaction`,
`read_query`, or an exact semantic solver.

The factorized expression DAG remains host-side structural bookkeeping. It may
extend and union whole lineages and merge exactly equal predicted complete
states. It may never average register fields across hypotheses.

## Delayed-evidence contract

Each record receives a deterministic random five-register probe state. The
independent assessor executes the record's actual behavior and renders only the
probe's before and after values in natural language. It does not reveal a
program ID, source alias, parse index, query, answer, or fault-line label.

The learned evidence reader predicts all ten values. For each candidate choice,
the learned executor applies that choice to the model-predicted before state.
The guard retains the choice only when its complete predicted after state equals
the model-predicted observed after state. Empty support fails closed. Evidence
is bound to the sealed source commitment and record provenance.

Train renderers and vocabulary are disjoint from held lexical and renderer
templates. Composition evaluation adds irrelevant audit clauses and changes
field order. Numeric values are covered during training; state combinations,
renderer combinations, source aliases, source records, depths, and complete
program compositions are held out.

## Learned transition algebra

The packet exposes one of four typed primitive action tokens:

- add three to register zero;
- swap registers zero and one;
- swap registers two and three; or
- swap registers three and four.

The executor is a learned structured stochastic operator, not a hard-coded
implementation. For action `a` and output register `o`, it learns a distribution
over input-register routes and a distribution over bounded integer deltas:

```text
p(v'_o | v, a) = sum_i sum_d p(i | a,o) p(d | a,o)
                  1[v'_o = v_i + d]
```

Route and delta parameters are tied across every occurrence and recurrent step.
The bounded algebra is an architectural prior; the route and delta semantics of
each action are learned only from successor-state supervision. Out-of-range
mass fails closed. Autonomous execution uses hard argmax decisions and feeds its
own predicted state into the next step.

## Learned query reader

The late reader receives only a complete predicted terminal state and a typed
`READ_VALUE(slot)` query. It learns a query-conditioned route over registers and
returns a categorical value. The evaluator aggregates only whole-state answers
across compatible lineages. Agreement answers; disagreement abstains.

## Frozen data and controls

Component training uses deterministic synthetic supervisor data independent of
the four frozen HSC1 evaluation cohorts:

- random register states and all primitive actions for one-step execution;
- free-running random action programs for held-depth execution;
- random state/query pairs for late reading; and
- random probe states rendered into natural-language before/after evidence.

The full gate uses the same train, lexical-shift, renderer-shift, and
composition-shift episode generator, seeds, HSC1 checkpoint, K=2 envelope, and
source-sealed evaluation contract as the accepted ULC1-HSC1 result. Exact-host
ULC1 remains the oracle ceiling. Required interface controls are:

- exact evidence with learned execution/query;
- learned evidence with exact assessor scoring, reported only as diagnosis;
- shuffled delayed evidence;
- packet/evidence provenance swap;
- terminal-state reset; and
- conflict-disabled factorized support.

## Frozen pass/kill gates

All conditions are conjunctive. No threshold changes after the first full
MEI1 report.

### Component gates

- evidence before-state exact >= 99.0% on every held renderer cohort;
- evidence after-state exact >= 99.0% on every held renderer cohort;
- one-step complete-state exact >= 99.9% on a held random-state board;
- free-running terminal-state exact >= 99.0% at every held depth;
- late-query exact >= 99.9% on held state/query pairs;
- no out-of-range probability mass accepted; and
- candidate runtime source audit finds no exact transaction/query import.

### Full composition gates

- frozen HSC1 gold semantic support >= 95% in every cohort;
- full model-owned exact sensitive answers >= 90% in every cohort;
- full model-owned exactness is within five points of exact-host DIVERGE in
  every cohort;
- recovery conditional on initially wrong top-1 >= 90% in every cohort;
- represented gold is never removed when both learned evidence states and the
  gold candidate execution are correct;
- underdetermined model-owned reads always abstain;
- shuffled evidence and terminal-state reset each reduce shifted exactness by
  at least 20 points;
- packet/evidence provenance swaps are rejected 100%; and
- post-seal WORLD replacement cannot change the result.

MEI1 is killed as a composed architecture if any component gate fails. If all
components pass but full composition fails, the interface between factorized
support and model-owned semantics is defective; no width, seed, duration, or
loss variant is authorized. A pass authorizes one broader learned-world board,
not long pretraining or a general reasoning claim.

## Claim boundary

A pass means a frozen learned-language compiler, learned delayed-evidence
interpreter, learned recurrent executor, and learned late reader can jointly
recover coherent answers over the existing synthetic DIVERGE board. It does not
mean Shohin can yet reason over unrestricted language, and the extreme ULC1
sharing ratios remain properties of a redundant grammar-specific board.

## Frozen result

Job `743224` ran the only authorized seed on one H100. It completed 1,600
updates / 102,400 evidence examples in 112 seconds and exited nonzero solely
because the conjunctive component gate failed. No full-composition job ran.

The structured model-owned algebra succeeded:

- one-step complete-state exactness: 100% on 20,000 held states;
- free-running terminal-state exactness: 100% at depths 4, 8, 16, and 24,
  with 2,000 held programs at each depth;
- late-query exactness: 100% on 20,000 held state/query pairs;
- accepted out-of-range mass: zero; and
- candidate source audit: pass.

The monolithic fixed-field evidence reader failed sharply. Complete before /
after state exactness was `94.629% / 95.020%` in distribution,
`9.473% / 7.812%` on lexical shift, `0.098% / 1.855%` on renderer shift,
and zero on composition shift. Fieldwise exactness was 98.936%, 65.283%,
27.549%, and 16.680%, respectively. This is a binding failure: the head can
learn values in familiar fixed templates but cannot preserve address/value and
before/after identity under renamed, reordered, or interleaved evidence.

MEI1 is therefore closed without a second seed, longer training, new renderer,
width, optimizer, or loss. The learned route-plus-delta executor and learned
query reader remain qualified reusable components; the fixed ten-query evidence
head does not. Any successor must change the evidence interface structurally,
using explicit whole-mention address/value binding rather than another pooled
field head.

The immutable external report and checkpoint currently verify at SHA-256
`a081ab0b3257149643b20b7f320269a7e2193df0fd6665d26fc283210aa80429` and
`bed9abefa2ecd2401c11515fe182d89871ab537bd8f8716bd5688ae693b79c29`.
The checkpoint-embedded report is byte-identical to the external report.
