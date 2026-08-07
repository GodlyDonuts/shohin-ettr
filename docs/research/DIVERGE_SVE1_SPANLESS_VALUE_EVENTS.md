# DIVERGE-SVE1: Spanless Value-Event Transduction

Status: frozen before final data materialization or model scoring.

## Capability Hypothesis

OQB1 established that exact repeated surface identity can be quotiented into an
anonymous episode-local address while a learned owner attaches semantic value
and query roles. SVE1 asks whether the remaining host-provided numeric spans and
parsed integers can be removed. A byte-level model must emit complete,
value-bearing transition and initialization events from raw source text. Its
output, not a host digit parser, is the only value input to law compilation and
state initialization.

SVE1 preserves the already-qualified OQB1 late-query owner and NCP1 command
owner bit-for-bit. It changes only the evidence and initial-value interface.
Long continuation pretraining and public-benchmark claims remain unauthorized.

## Candidate Interface

The generic OQB1 occurrence quotient replaces the two declared raw register
names with anonymous table-position markers and deletes the names. A shared
472,136-parameter two-layer bidirectional byte GRU emits CTC sequences through
two heads:

- evidence: four tokens from `role * 97 + value`, where `role` is one of
  before-slot-0, before-slot-1, after-slot-0, after-slot-1;
- initial state: two tokens from `slot * 97 + value`.

The candidate runtime receives no numeric boundaries, integer values, typed
initial state, terminal state, target register, or answer. It greedily decodes
CTC outputs and fails closed unless the evidence roles are exactly `{0,1,2,3}`
or initial slots exactly `{0,1}`. Exact bounded Z/97 law support, alias lookup,
the modular executor, the two-name declaration, and exact repeated-surface
identity remain engineered scaffolds.

Teacher-only frame alignment is allowed during training to accelerate the
falsifier. Those alignments never enter the checkpoint interface, candidate
runtime, development evaluator, or confirmation evaluator.

## Frozen Data And Training

- training: 100,000 fresh records, seed `2026080841`;
- development: 256 episodes, seed `2026080842`;
- conditional confirmation: five 256-episode boards, seeds `2026080843` through
  `2026080847`;
- treatment and shuffled-target control: identical initialization, architecture,
  data sources, sampler, 1,500 AdamW updates, batch 128, learning rate 0.001;
- shuffled control pairs every source with targets offset by 50,021 rows;
- all final source, name, and identity commitments must be transitively disjoint
  from the qualified OQB1 lineage and independently regenerate byte-for-byte.

The development and confirmation renderers, names, matrices, commands, depths,
and identities are held out by the inherited generator split. Confirmation is
not readable unless development passes every condition below.

## Matched Arms

1. treatment;
2. unseen register renaming;
3. coherent table reindex across evidence, initial state, and query;
4. evidence-only table reindex;
5. value scrub replacing every digit byte without identifying spans;
6. occurrence-level quotient break;
7. independently trained shuffled-target model.

NCP1 program predictions and OQB1 query predictions are shared where the arm
does not causally intervene on their interface.

## Frozen Gate

Development passes only if all conditions are conjunctively true:

- NCP1 complete-program exactness is at least 99%;
- treatment, unseen rename, and coherent reindex each achieve at least 99% exact
  evidence events, initial events, query binding, law commits, terminal states,
  and answers;
- every positive transfer-depth cell is at least 95% exact;
- evidence-only reindex is at most 5% states and 10% answers;
- value scrub is at most 1% evidence events, initial events, states, and answers;
- occurrence break is at most 30% evidence events, 35% initial events, 20%
  states, and 35% answers;
- shuffled-target model is at most 1% evidence/initial events, 5% states, and
  10% answers;
- schedules, charged examples, parameterization, initialization, data, reports,
  parent hashes, and checkpoint hashes match exactly;
- candidate runtime contains no numeric-span scanner or raw-integer parser.

A miss closes exact SVE1 without width, duration, seed, threshold, renderer,
loss, alignment, vocabulary, or parser variants. A pass opens each fixed
confirmation board exactly once; all five must pass unchanged.

## Claim Boundary

A confirmed pass would qualify only spanless byte-to-value-event transduction
inside this controlled bounded algebra and the already-qualified OQB/NCP
system. It would not establish open-domain numerical parsing, alias/coreference
equivalence, learned law search, unrestricted natural-program reasoning, or
public benchmark capability.
