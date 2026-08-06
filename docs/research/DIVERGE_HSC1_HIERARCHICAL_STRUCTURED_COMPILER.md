# DIVERGE-HSC1 Hierarchical Structured Compiler

**Status:** neural seed-one negative; closed without variants

**Decision date:** 2026-08-05

## 1. Evidence and hypothesis

The autonomous source boundary has one sharp asymmetry:

- the option-local token-role compiler is exact on five of five seeds when a
  complete option is supplied as one source object;
- SC1 loses lineage by multiplying independent token and pair decisions; and
- WRA1 has exact record boundaries and bounded cardinality, but permutation-
  matched parallel slots still fail to bind all fields into one object.

HSC1 tests a distinct interface:

> First infer one globally normalized monotonic hierarchy over the raw source.
> Then infer each complete option through one globally normalized finite-state
> semantic template. No field is normalized independently, and the runtime
> never constructs a pair graph, Cartesian proposal set, exchangeable slot,
> Hungarian assignment, retry set, or beam.

This is not a WRA1 decoder, matching, width, duration, seed, threshold, or loss
variant. WRA1 remains closed. HSC1 uses source contiguity and ordered semantic
templates rather than unordered object slots.

## 2. Frozen architecture

### One raw source pass

Load the immutable failed-SC1 checkpoint only as a frozen SmolLM2 layer-17
source encoder and frozen record-boundary detector. Encode the raw WORLD once.
Pair positive record boundaries in source order and fail closed under the
existing nine-record / 108-word limits. No gold record, option, renderer,
ontology, query, answer, execution state, or delayed evidence enters runtime.

### Globally normalized record parse

For each predicted record segment, three learned gap-score channels identify:

1. the start of source-ordered option A;
2. the start of source-ordered option B; and
3. the start of the nonoperative trailer.

Only triples `0 < a < b < t < record_width` are legal. Training uses the exact
log-partition over every legal triple. Inference uses exact Viterbi decoding,
not three independent argmaxes. The resulting phases are:

```text
HEADER | OPTION_A | OPTION_B | TRAILER
```

A globally normalized cue-role head chooses exactly one source position and
one of `fault-line` or `background` inside `HEADER`.

### Globally normalized option object

The two predicted option spans are source-ordered, not exchangeable. A shared
two-layer local encoder reads each span's already-computed raw-source states.
It emits token-role scores for `OTHER`, alias begin/inside, favored/reserve,
and four action identities.

The candidate runtime enumerates only a fixed semantic grammar:

- four possible component orders;
- prior class in two values;
- program class in four values; and
- alias length from one to four words.

For each finite template, an exact dynamic program sums or maximizes over all
legal monotonic source alignments. Alias continuation positions must be
physically consecutive. The training loss is one global log-partition over all
templates and alignments minus the gold complete-object path. Viterbi returns
one coherent alias/prior/program/witness object. Unselected words contribute a
shared `OTHER` baseline. There is no independently normalized alias pointer,
prior pointer, action pointer, or class head.

The record and two options are validated, canonicalized by source position,
committed into the existing DIVERGE packet, and then all raw source bytes,
residuals, and KV state are deleted.

## 3. Frozen training charge

The parent, tokenizer, source layer, source encoder, record-boundary detector,
batch size, total update count, and optimizer family remain unchanged.

- stage A: 200 updates / 1,600 episodes train only record cut and cue heads;
- freeze stage-A weights permanently;
- stage B: 1,000 updates / 8,000 new episodes train only the shared local
  option encoder and globally normalized semantic-template scores;
- stage B receives only stage-A Viterbi option spans; records whose predicted
  cuts do not equal the supervised record are charged and reported but provide
  no hidden gold-span fallback;
- total: 1,200 updates / exactly 9,600 charged episodes;
- 256 fresh episodes each for train, lexical, renderer, and composition
  cohorts; and
- one seed first, four more only if seed one clears every floor.

## 4. CPU mechanics gate

Before neural execution:

1. cut log-partition and Viterbi must match exhaustive triple enumeration;
2. template log-partition and Viterbi must match exhaustive path enumeration;
3. calibrated raw-source scores must reconstruct 1,000/1,000 complete packets;
4. semantic-role and cut-channel shuffles must each lose at least 20 points;
5. malformed/nonmonotonic phases and overlapping witnesses must fail closed;
6. post-seal source poisoning must change no packet semantics; and
7. accounting must remain linear in source words plus the fixed 128-template
   grammar, with no pair matrix, complete-particle materialization, retry, or
   answer-guided path.

## 5. Neural pass/kill contract

Seed one passes only if every cohort reaches:

- record segmentation `>=99%`;
- hierarchical option-phase parsing `>=99%`;
- gold-support recall `>=95%`;
- exact sealed packets `>=95%`;
- zero overflow and zero accepted invalid/overlapping objects;
- post-seal source-poison invariance exactly 100%;
- at least 20 points exact collapse under semantic-template lineage shuffle;
  and
- at least 15 points exact advantage over closed SC1 and WRA1 on every shift.

Five-seed promotion requires every seed to beat both closed compilers and at
least four of five to clear every absolute floor. Failure of seed one closes
HSC1 without a threshold, stage-length, width, source-layer, template,
renderer, seed, optimizer, or loss variant.

## 6. Claim boundary

Linear-chain CRFs, semi-Markov parsing, finite-state templates, Viterbi,
structured prediction, and source copying are established methods. HSC1 is not
the architecture novelty claim. It is one bounded attempt to provide
DIVERGE's source-sealed factorized epistemic state with an autonomous compiler
that preserves complete-object lineage. A pass authorizes exactly one
unchanged `>=8`-world DIVERGE recovery gate. It does not reopen the failed
broad resource claim or authorize continuation pretraining.

## 7. Exact CPU result

The frozen seed-`202608056800` board passes all gates on 1,000 episodes / 5,904
records across the four cohorts:

- complete packet reconstruction: `1000/1000`;
- cut-channel shuffle exactness: `0/1000`;
- semantic-role shuffle exactness: `0/1000`;
- malformed option-width fail-closed: `1000/1000`;
- post-seal source-poison invariance: `1000/1000`;
- overflow: `0/1000`; and
- exact linear score/object accounting: `1000/1000`.

The receipt charges 280,803 source words, 195,251 option words, 798,297 cut
cells, 798,297 cue cells, 2,147,761 semantic-role cells, and exactly 1,511,424
fixed-template evaluations. It allocates no pair matrix. The cut and option
dynamic programs also match independent exhaustive references; the
differentiable implementations match the CPU partitions and have finite
gradients.

Canonical report digest is
`23fa3a02f4299a2ff2b29dde415e8fcefc7bc8ecd2a56877cdee68d446c81809`;
stored JSON SHA-256 is
`ef795c698a9ab4195eba43a6d8f84b46e55528aad66f708893de59dfd22dbad2`.
This authorizes only the frozen two-update real-checkpoint smoke and, if that
is mechanically clean, one 1,200-update neural seed.

## 8. Neural result and decision

The portable immutable runtime is commit `fce7efb`, archive SHA-256
`317e85f1182ba6bb63d7bcf2e1256a751b59d487a26e0118aa89d9f71b4e2c0e`,
and manifest SHA-256
`b738e6f31b4bee91de3b033307027228b9801dfd83563ec70f52f9fb38f6afd0`.
Smoke job `742764` reaches 100% stage-A phase parsing and exercises one exact
stage-B update over all 96 predicted option spans without fallback. Full job
`742775` completes the frozen 1,200 updates / 9,600 episodes in 8m09s. The
893,393 trainable parameters peak at 759,383,040 allocated GPU bytes. Stage-B
loss falls from `9.47937` to `4.849e-5`.

The autonomous result fails the gate:

| Cohort | Segmentation | Phase parse | Gold support | Exact packet |
|---|---:|---:|---:|---:|
| Train | 100% | 95.313% | 96.875% | 96.094% |
| Lexical shift | 100% | 97.656% | 47.266% | 8.594% |
| Renderer shift | 100% | 99.219% | 96.094% | 0% |
| Composition shift | 100% | 99.609% | 11.328% | 0% |

Overflow and accepted overlap remain zero, and source-poison invariance stays
100%. Those mechanical passes cannot override first-pass support loss and zero
shifted packet exactness. Report/checkpoint SHA-256 values are
`62ac144d66818e32ba261fade1ac9103d5adbe962d3572004f3a249ba94c56ad` /
`34c7eaee885ba5201e6e07335add1737b7b7d26b2709861b7967e0b97be64a05`.

HSC1 is closed after seed one. Do not run another seed, width, duration,
source layer, stage allocation, optimizer, loss, cue repair, template repair,
or threshold. The result localizes the remaining failure to shifted language
grounding rather than exact structured decoding. It does not authorize a
DIVERGE composition, continuation pretraining, or reasoning claim.
