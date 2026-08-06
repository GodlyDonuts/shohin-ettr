# DIVERGE-QTG1: Query-Conditioned Typed Gatherer

Status: frozen before implementation or neural results on 2026-08-06.

## 1. Evidence and hypothesis

Three different unconditioned source interfaces now fail the same boundary:

- the whole-span quotient preserves mention identity but reaches only 17.920%
  complete shifted tuples;
- the pooled MEI1 reader reaches 0--9.5% shifted complete states; and
- MQB1 enforces atomic mentions plus exact one-to-one assignment but reaches
  0% shifted complete assignments.

MQB1 proves that assignment combinatorics are not the remaining blocker. The
frozen one-pass source representation does not ground unseen phase/address
language into canonical typed fields.

QTG1 tests one materially different source computation:

> Re-encode the evidence while conditioning on one explicit typed natural-
> language question per target field, jointly adapt the small source-side
> memory encoder, and gather one complete value mention for each question.

This is a bounded extractive source-interface gate, not a claim that
query-conditioned attention is novel. Its purpose is to determine whether the
qualified model-owned DIVERGE algebra can receive correctly grounded state
evidence without an exact host parser.

## 2. Frozen board and budget

QTG1 uses the exact MEI1/MQB1 train, lexical, renderer, and composition words,
values, splits, and assessor labels. Held aliases/templates never enter
training or query construction. The ten fixed canonical questions are:

```
retrieve the initial value for slot 0
...
retrieve the initial value for slot 4
retrieve the final value for slot 0
...
retrieve the final value for slot 4
```

Every question is prepended to the unchanged evidence and encoded by the same
shared Smol/HSC1 source stack. The budget remains one seed, 1,600 updates,
batch 64 source records, and 102,400 source records. Because every record is
read under ten questions, the report must separately count 1,024,000
question-conditioned sequences and all actual source FLOPs/wall time.

Evaluation remains 20,000 records in each of the four cohorts. No held query
paraphrase, held evidence label, or result-dependent threshold is allowed.

## 3. Candidate architecture

For typed field f and evidence E, the source stack reads:

```
[canonical question f] evidence [unchanged E]
```

The frozen Smol backbone remains frozen. The existing HSC1 source-side
`memory_norm`, `memory_projection`, and two-layer `memory_encoder` are copied
from the immutable checkpoint and jointly adapted; they contain about one
million parameters. The HSC1 compiler itself is not mutated. A new shared
gatherer receives query and evidence word states and predicts:

1. a pointer score over evidence words; and
2. a value class 0--127 or NONE at each evidence word.

The same gatherer weights serve all ten questions. The candidate runtime does
not receive a field ID outside the natural-language question. It keeps at most
twelve learned numeric candidates and reuses the exact one-to-one assignment
decoder only to ensure that ten questions cannot detach values or select the
same mention twice. The selected value must come from the selected evidence
mention under that question. Missing fields, duplicate mentions, candidate
overflow, invalid values, or nonpositive value/NONE margins fail closed.

The complete loss is fixed:

```
L = L_pointer + L_value.
```

`L_pointer` is ten extractive pointer CEs. `L_value` is class-balanced CE in
which each question's gold mention has its numeric value and all other evidence
words are NONE. No exact numeric parser, regex, template ID, gold pointer, or
held label is present at candidate inference.

## 4. Controls and source boundary

After the ten source reads, raw question/evidence tokens, source residuals, and
KV state are deleted. The sealed output is only one complete pair of typed
five-register states plus source/provenance commitments. The preserved MEI1
executor and query reader remain frozen by hash.

The component report must include:

- full query-conditioned QTG1;
- questions shuffled across typed output fields;
- value logits shuffled across evidence mentions;
- adapted source memory reset to its immutable HSC1 initialization;
- duplicate/overflow/provenance accounting; and
- forbidden-import/runtime source audit.

Question shuffling is the primary causal control. The reset-source arm is
diagnostic: a pass may come from pretrained semantics without requiring an
adaptation claim, but the adapted full 20,000-record scores and reset-source
2,000-record-per-cohort diagnostic must both be reported.

## 5. Frozen gates

All conditions are conjunctive:

- byte-identical evidence rendering and 100% supervisor mention coverage;
- BEFORE complete-state exact >=99.0% on every cohort;
- AFTER complete-state exact >=99.0% on every cohort;
- complete ten-field assignment >=99.0% on every cohort;
- selected mention value exact >=99.9% on every cohort;
- zero accepted duplicate/missing fields, invalid values, candidate overflow,
  or provenance mismatches;
- shuffled questions reduce shifted complete-state exactness by >=50 points in
  every shifted cohort;
- shuffled mention values reduce shifted complete-state exactness by >=50
  points in every shifted cohort;
- frozen Smol backbone, MEI1 executor, and MEI1 query hashes remain unchanged;
  and
- candidate source audit passes.

Only if every component gate passes may one unchanged full
HSC1 -> ULC1 -> QTG1 -> learned executor -> learned query composition run.
That full gate retains the MEI1 >=90% every-cohort threshold, <=5-point loss
from exact-host DIVERGE, perfect provenance/source controls, and all existing
causal interventions.

## 6. Stop rule

QTG1 receives one seed and one budget. No query wording, update count, width,
source layer, learning rate, optimizer, threshold, renderer, or seed repair is
allowed after the first report. If QTG1 fails, close this synthetic evidence-
compiler sequence and move to a jointly trained end-to-end source-to-state
trajectory on a capable development backbone. Do not run another isolated
token/span/pointer reader.

No continuation pretraining, public benchmark optimization, or general
reasoning claim is authorized by this gate.

## 7. Frozen result

Newton job `743269` completed the only authorized seed on one H100 in 38m40s.
The optimizer executed exactly 1,600 updates / 102,400 source records /
1,024,000 query-conditioned sequences. The optimizer phase reached update
1,600 after 1,400.67 seconds; the complete training plus four-cohort and
source-reset evaluation took 2,302.67 seconds. The model used 2,170,241
trainable parameters and 2,381,004,800 peak allocated CUDA bytes.

The in-distribution fit is nearly exact: before and after states are each
99.94%, complete state pairs 99.905%, complete assignments 99.975%, and
selected values 99.993%. That fit does not transfer:

| cohort | before | after | complete pair | assignment | selected value | valid |
|---|---:|---:|---:|---:|---:|---:|
| train | 99.940% | 99.940% | 99.905% | 99.975% | 99.993% | 99.975% |
| lexical | 27.755% | 28.155% | 23.080% | 28.985% | 79.0815% | 34.160% |
| renderer | 3.745% | 0.780% | 0.710% | 0.000% | 37.570% | 31.990% |
| composition | 0.000% | 0.015% | 0.000% | 0.000% | 23.7525% | 4.645% |

Source adaptation is causally necessary but not sufficient. Resetting the
source adapter to its immutable HSC1 initialization produces zero complete
state pairs and zero valid packets in all four 2,000-record diagnostics.
Question and value shuffles also produce zero complete pairs, but the shifted
treatment is too weak to satisfy the frozen 50-point causal-drop gates. There
are no accepted duplicate mentions or overflows, renderer parity is 100%, and
the frozen backbone, learned executor, and learned query hashes remain
identical. Shifted valid packets nevertheless contain 2,188 lexical, 63,973
renderer, and 9,194 composition provenance mismatches.

This is a semantic-generalization failure, not an assignment, executor, query,
or infrastructure failure. Canonical typed questions plus joint adaptation of
the small HSC1 source stack memorize the training renderer while failing unseen
nominal and compositional grounding. The conjunctive gate fails, so no full
HSC1 -> ULC1 -> QTG1 composition run is authorized.

QTG1 and the entire isolated token/span/pointer evidence-reader sequence are
closed without repair. The ordered successor must jointly train the language
source, persistent state trajectory, recurrent execution, and late query path
on a capable development backbone. Report/checkpoint SHA-256 values are
`0b30d6698b583901c67b1a9095d99238e5eb2aabb295d41476988005748f2d18` /
`623172dee51317cc01d1b5f07c0048637581beab83fd96ee00bc7f75af74e9f0`.
