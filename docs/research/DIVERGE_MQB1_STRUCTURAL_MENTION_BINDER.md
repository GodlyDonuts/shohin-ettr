# DIVERGE-MQB1: Structural Whole-Mention Evidence Binder

Status: frozen before implementation or neural results on 2026-08-06.

## 1. Decision boundary

DIVERGE-MEI1 qualified its model-owned recurrent executor and late query
reader, but closed its delayed-language evidence interface. The fixed ten-query
reader reached 94.629%/95.020% before/after complete-state exactness in
distribution and collapsed to 0--9.5% under lexical, renderer, and composition
shift. Fieldwise accuracy remained materially higher than complete-state
accuracy. The failure is therefore binding: values are detected but detached
from their phase/address identity.

MQB1 tests one structural replacement and no repair of the failed pooled head:

> If every numeric observation remains an indivisible contextual mention and
> the runtime chooses one complete one-to-one assignment of mentions to typed
> phase/address fields, then the model can preserve address/value identity
> across lexical renaming, reordering, reversal, and distractor interleaving.

MQB1 is not a broad reasoning, language-understanding, or novelty claim. The
earlier whole-span quotient gate already showed that mention boundaries alone
do not solve unseen nominal grounding. MQB1 differs only in the combination
being tested here: model-predicted values and semantic fields remain attached
to one source mention, a complete typed assignment is decoded globally, and
the result feeds the already-qualified model-owned recurrent algebra.

## 2. Frozen source and board

The source encoder, tokenizer, HSC1 checkpoint, MEI1 executor/query checkpoint,
train renderer set, and four evaluation cohorts are unchanged from MEI1.

- train evidence: the four original train templates only;
- lexical shift: the two original disjoint lexical templates;
- renderer shift: the two original reordered/reversed templates;
- composition shift: the two original interleaved templates with one numeric
  distractor;
- value domain: 0--127;
- typed state: BEFORE/AFTER x five registers;
- train budget: one seed, 1,600 updates, batch 64, exactly 102,400 examples;
- evaluation: 20,000 examples per cohort.

The assessor records the word position of each rendered value solely as a
supervision label. It must verify byte-for-byte that deleting the metadata
leaves the same words emitted by the frozen MEI1 renderer. Held templates,
aliases, and labels never enter training or threshold selection.

## 3. Candidate architecture

### 3.1 Atomic mention objects

The frozen source encoder emits one contextual vector per source word. MQB1
constructs a local mention vector for every word from the contextual word,
bounded left/right context, and a shared mention encoder. The mention vector is
the only object from which the following predictions may be made:

1. value class: 0--127 or NONE;
2. phase class: BEFORE, AFTER, or NONE;
3. address class: register 0--4 or NONE; and
4. pair-equivalence: whether two selected mentions refer to the same address.

The runtime may not parse decimal strings, inspect template IDs, call the exact
renderer, recover supervisor word positions, or copy a value from a different
mention after selecting an address. A selected mention carries its predicted
value with it as one object.

### 3.2 Complete typed assignment

For a source with T words, the learned score for assigning mention t to typed
field f=(phase,address) is

```
S(t,f) = log p(value(t) != NONE)
       + log p(phase(f) | t)
       + log p(address(f) | t).
```

The runtime keeps at most the twelve strongest learned numeric candidates and
uses an exact dynamic program to select ten distinct mentions, one for every
typed field. Skipping up to two candidates permits the composition distractor
without identifying it through host logic. The decoder never averages fields
or values and never combines the field score from one mention with the value
prediction from another.

The packet fails closed unless all ten fields receive distinct mentions, each
selected value beats NONE, and the learned same-address score is positive for
the selected BEFORE/AFTER pair of every register. Any duplicate, missing field,
candidate overflow, invalid value, or failed pair certificate yields ABSTAIN.

### 3.3 Learning objective

Only train-renderer examples contribute gradients. The fixed loss is:

```
L = L_value + L_phase + L_address + L_field_pointer + L_pair.
```

- `L_value` is class-balanced wordwise CE over 128 values plus NONE.
- `L_phase` and `L_address` are class-balanced wordwise CE with NONE.
- `L_field_pointer` is ten pointer CEs over source words, each targeting the
  gold word for one typed field.
- `L_pair` is balanced BCE over positive same-address cross-phase pairs and
  deterministically sampled negative pairs.

No result-dependent loss coefficient, threshold, renderer, width, seed,
duration, or optimizer change is permitted.

## 4. Source-sealed runtime contract

Candidate runtime inputs after encoding are only contextual word tensors and a
valid-word mask. Candidate runtime outputs are zero or one complete pair of
five-register states plus mention/provenance indices. Runtime source audit must
reject imports or calls to:

- the exact MEI1/DIVERGE transaction or query functions;
- the assessor renderer or gold mention labels;
- regex, integer parsing, tokenizer decode, or template lookup; and
- any fieldwise fallback or host repair.

The already-qualified `StructuredRegisterExecutor` and
`StructuredQueryReader` are loaded from the immutable MEI1 checkpoint and
remain frozen. Their state hashes must match the preserved checkpoint before
and after MQB1 training.

## 5. Frozen component gates

All conditions are conjunctive:

- byte-for-byte renderer parity with MEI1 on 10,000 deterministic examples;
- 100% gold mentions represented and no distractor labeled as a field;
- BEFORE complete-state exact >=99.0% on every cohort;
- AFTER complete-state exact >=99.0% on every cohort;
- complete ten-field assignment >=99.0% on every cohort;
- selected mention value exact >=99.9% on every cohort;
- zero accepted duplicate/missing fields, invalid values, or candidate
  overflows;
- zero field/value provenance mismatches;
- shuffled mention values reduce shifted complete-state exactness by >=50
  percentage points;
- shuffled phase/address scores reduce shifted complete-state exactness by
  >=50 percentage points;
- pair-equivalence intervention causes rejection whenever its selected
  certificate is negated;
- frozen executor/query state hashes are unchanged; and
- candidate source audit passes.

The historical MEI1 pooled reader is the protected baseline. Parameter count,
training examples, updates, H100 time, peak memory, and throughput are reported
for both. MQB1 may pass as an interface gate only; it does not earn a DIVERGE
or reasoning claim from component accuracy.

## 6. Ordered full-composition gate

The full learned HSC1 -> ULC1 -> MQB1 -> model-owned executor -> learned query
gate may run exactly once only if every component condition passes. It reuses
the frozen MEI1 full-composition evaluator with only the evidence-interpreter
adapter changed. It must preserve the existing source commitment and all
provenance, source-poison, shuffled-evidence, packet-swap, conflict-disabled,
terminal-reset, invariant-query, and underdetermined-query controls.

The full gate remains:

- model-owned exact >=90% on every cohort;
- within five points of exact-host DIVERGE on every cohort;
- represented gold never removed when the model evidence is exact;
- provenance swaps rejected 100%; and
- every frozen causal/integrity control passes.

## 7. Stop rules

MQB1 receives one seed and one budget. If any component gate fails, close MQB1
without a width, duration, loss, threshold, renderer, optimizer, or seed retry.
Retain the qualified MEI1 executor/query and move to a materially different
language-to-epistemic-packet interface. If MQB1 passes components but fails
composition, do not train separately fitted modules longer; the next candidate
must jointly optimize the complete source-to-state trajectory.

No continuation pretraining, public benchmark optimization, or general
reasoning claim is authorized by this gate.

## 8. Immutable result

Status: **closed negative**. Newton job `743263` completed the frozen one-seed
run on `evc30` in 6m03s. Its scientific process exited 1 only because the
conjunctive gate failed. It trained exactly 1,600 updates / 102,400 examples,
used 1,953,676 trainable parameters, and preserved the MEI1 executor/query
state hashes exactly. Renderer parity and the candidate source audit passed.
No full-composition job ran.

| Cohort | Valid | Before | After | Complete assignment | Complete state pair | Selected value |
|---|---:|---:|---:|---:|---:|---:|
| train | 100.000% | 98.325% | 98.085% | 100.000% | 96.865% | 99.6385% |
| lexical | 2.885% | 0.000% | 0.000% | 0.000% | 0.000% | 39.7405% |
| renderer | 39.310% | 0.000% | 0.010% | 0.000% | 0.000% | 35.5405% |
| composition | 6.955% | 0.760% | 1.315% | 0.000% | 0.755% | 35.9425% |

The exact assignment layer never accepted duplicate mentions or overflow and
the pair-certificate negation rejected 100% of otherwise valid packets. It did
not recover semantic grounding: accepted shifted packets contained 4,477,
74,920, and 13,903 field/provenance mismatches for lexical, renderer, and
composition cohorts. Value and field interventions consequently had no
50-point drop to demonstrate because the untreated shifted system was already
near zero.

MQB1 therefore disproves the hypothesis that preserving atomic value mentions
and adding a complete one-to-one assignment is sufficient. The repeated
failure shared by the span quotient, pooled MEI1 reader, and MQB1 is now the
frozen one-pass source representation's inability to ground unseen
phase/address language, not fieldwise averaging or assignment combinatorics.
MQB1 receives no seed, width, duration, renderer, optimizer, threshold, or loss
repair.

Report/checkpoint SHA-256 values are
`265ef25b99a64ee58f38acc1b0d7506a3e08adb3194b98069c1fc29f1672b24a` /
`19e82c966adb753d3159235a60f3923bc591459476b3fbc8e148b39311cb3eed`.
Both are read-only and hash-verified on Newton and in local disaster recovery.

The next admissible interface must alter source computation itself. A
query-conditioned typed gatherer, jointly adapting the source-side memory
encoder while asking one explicit semantic question per field, is materially
different from another unconditioned token/span classifier. It must be frozen
as a new gate before results and may not reuse MQB1's failed head.
