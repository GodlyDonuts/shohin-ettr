# DIVERGE-SVE1: Spanless Value-Event Transduction

Status: confirmed across development and five fixed source-disjoint boards.

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

## Confirmed Result

Matched one-H100 jobs `744855/744856` completed all 1,500 frozen updates from
identical initialization. Treatment reached `2048/2048` on its fixed training
sample; the 50,021-row-offset shuffled-target control reached `0/2048`.
Treatment/control checkpoint SHA-256 values are
`4d037d61b313d4237bf1d2309b00cf0905fe32478c2b399db56d2204d923d901`
and
`643ef2a9193ecb6a40d6957e3590bd0286f73329e24d86845624660e0bfebe15`.

Development job `744857` passed every frozen condition. NCP1 programs were
`4096/4096`. Treatment, unseen rename, and coherent table reindex were each
exact on `6144/6144` evidence event sequences, `4096/4096` initial event
sequences, `4096/4096` terminal states, and `8192/8192` answers, with a 100%
floor at every depth 12--32. Value scrub and the shuffled-target model produced
zero complete events, states, or answers. Occurrence break produced zero
states/answers. Cross-owner reindex preserved the local events but collapsed
to `2/4096` states and `95/8192` answers. Development report SHA-256 is
`93096b61b34528157ac6c1eb71dd43807aa2d87eff4bd1b077be0282051472d6`.

The five fixed confirmation jobs `744860`--`744864` ran once without retraining
and all passed. Aggregate treatment, unseen rename, and coherent reindex were
each exact on `30,720/30,720` evidence sequences, `20,480/20,480` initial
sequences and terminal states, and `40,960/40,960` answers. Value scrub,
occurrence break, and the shuffled-target model each produced zero states and
answers. Cross-owner reindex retained only `6/20,480` states and `400/40,960`
answers. Aggregate SHA-256 is
`41b31368e26d00dcb161c9f55c29f132520e78bff95d3a13e647254fd9603c4b`.

SVE1 therefore qualifies the narrow mechanism claimed above. The next
scaffold-removal target is the host's exact episode-law support solver; exact
operation alias binding, bounded coefficient vocabulary, modular execution,
and the two-name occurrence quotient remain explicit limitations.
