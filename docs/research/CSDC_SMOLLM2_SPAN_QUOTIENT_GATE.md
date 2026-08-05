# CSDC SmolLM2 Whole-Mention Span-Quotient Gate

**Status:** frozen before implementation and neural evaluation

**Decision date:** 2026-08-05

**Purpose:** resolve exactly one remaining interface question before DIVERGE:
can a frozen pretrained language backbone identify and preserve an entire
episode-local alias mention, independent of its BPE segmentation, well enough
to drive the protected CSDC hypothesis selector?

This is one pass/kill experiment. No seed, width, update-count, loss-weight,
threshold, renderer, or candidate-width repair follows its result.

## 1. Evidence and narrow hypothesis

The frozen first-subtoken lexical bridge reaches `99.691%` development and
`95.915%` lexical-shift answers. Shuffled outcomes and lineage swaps cause the
predicted large failures. It therefore carries causal evidence into CSDC, but
it recovers all eight shifted challenge tuples in only `17.920%` of episodes.
The interface labels and copies only the first BPE token overlapping each
alias.

The frozen hypothesis is:

> Pooling and classifying each complete model-selected alias span, while
> sharing evidence across exact-surface occurrences, will preserve lexical
> identity across unseen aliases and tokenizations and raise shifted all-eight
> tuple exactness from `17.920%` to at least `90%` without changing CSDC.

This gate tests occurrence identity, not general coreference, unrestricted
language understanding, DIVERGE, or public reasoning.

## 2. Immutable inputs and budget

The treatment reuses the exact inputs from the first-subtoken bridge:

| Input | Frozen SHA-256 |
|---|---|
| SmolLM2 parent | `8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02` |
| tokenizer | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| warm lexical adapter | `abd22528da0d8dc4718c7a89d9c94520540a2f38b7f0b1d9a9e623d0af23cf4d` |
| frozen CSDC reasoner | `c374e3b566808cb317ffcd2725653c9073d2e7aebeb75e93ed7ea2a7e2e27044` |

Training remains seed `2026080507`, 1,500 updates, batch 128, 192,000
episodes, AdamW at learning rate `3e-4` and weight decay `0.01`. Data,
renderer, evaluation, and evaluation-renderer seeds remain
`202608051000`, `202608052000`, `202608053000`, and `202608054000`.
Templates 0--2 and the training alias pools are visible in training. Template
3 and the disjoint shift alias pools remain evaluation-only. Evaluation uses
1,024 episodes in each family-by-depth cohort at depths 8 and 12.

The frozen first-subtoken report
`b3ae0526e9e28ef21e93f2b32bacd7845f40cdb663d8d4b48d74ed9b7cfc05c5`
is the matched control. It is not retrained.

## 3. Only architectural change

The frozen SmolLM2 layer-19 residual extractor and warm 384-wide, five-layer
context encoder remain unchanged. Record-kind prediction remains unchanged.
The token role head is replaced by a bounded whole-span quotient head:

1. Enumerate every nonempty contiguous token span of width one through four
   wholly inside one rendered record. Enumeration has no alias dictionary,
   field label, target value, query, or answer.
2. Represent a span by its first residual, last residual, and mean residual,
   followed by one learned span projection.
3. Quotient candidates by exact normalized source bytes within an episode.
   Mean-pool candidate representations inside each exact-surface class and
   send that class message back to every occurrence.
4. Classify each candidate as `OTHER`, `START`, `OUTCOME`, or `WORD` from its
   local whole-span representation and its occurrence-class message.
5. Decode nonoverlapping whole mentions inside each model-selected challenge
   record. Select one `START`, one `OUTCOME`, and up to twelve `WORD` mentions;
   order `WORD` mentions solely by source position.
6. Convert only an exactly selected annotated source mention to the same
   episode-local state or generator identity used by the frozen typed CSDC
   compiler. Partial spans, supersets, overlapping selections, missing fields,
   duplicate `START`/`OUTCOME`, and excess words fail closed.

The maximum span width of four is frozen before the board. The executable must
audit that every labeled alias occurrence in both alias pools is representable
under that cap before training. Any miss is an infrastructure failure and no
neural result may be reported.

The trainable span projection is `LayerNorm(3*384) -> Linear(3*384,384) ->
GELU`. The class projection is `LayerNorm(384) -> Linear(384,384) -> GELU`.
The role head is `LayerNorm(768) -> Linear(768,384) -> GELU -> Linear(384,4)`.
The loss is record-kind cross entropy plus span-role cross entropy over
candidates inside true challenge records. Span-role class weights are
`[0.05, 1, 1, 1]` for `OTHER/START/OUTCOME/WORD`. No auxiliary downstream,
contrastive, table, answer, or consistency loss is allowed.

The quotient groups exact surface mentions only. It may not merge aliases,
infer synonyms, or repair a malformed prediction. Complete particles,
version-space state, conflict clauses, and query-time ambiguity are absent.

## 4. Ownership boundary

### Model-owned

- challenge-record selection;
- every selected mention start and end;
- every selected mention role;
- the number and source order of generator mentions;
- all challenge tuples consumed by CSDC.

### Architectural

- bounded contiguous-span enumeration;
- exact source-byte equality among model-selected candidates;
- permutation-invariant class pooling;
- nonoverlap, cardinality, source-order, and fail-closed validation;
- unchanged complete-candidate construction, counterexample scoring, coherent
  CSDC commitment, tied execution, and late answer scoring.

### Forbidden at inference

- gold spans or role labels;
- candidate-name or alias dictionaries for proposal or selection;
- raw query, answer, true table, selected table, or terminal state in the
  compiler;
- retries, beam search, host semantic repair, or answer-guided selection;
- changing candidate width or thresholds after evaluation.

Span/value annotations exist only to supervise and score exact source copying.
The deterministic conversion of an exactly copied source mention to its
episode-local anonymous identity is the same disclosed copy mechanism used by
the protected bridge. This experiment does not claim a learned nominal
quotient between different surface strings.

## 5. Required reports and controls

The report must include, by split and by cohort:

- answer, oracle-answer, selected-table, all-eight tuple, and all-valid exact;
- per-field mention exactness for `START`, `OUTCOME`, and `WORD`;
- exact span-boundary and exact role accuracy over gold mentions;
- attempted and accepted partial-span, strict-superset, and overlap counts,
  plus duplicate and missing-field counts;
- exact-surface class consistency and class-use counts;
- shuffled-outcome and whole-lineage-swap answer accuracy;
- parameters, charged episodes, elapsed time, throughput, peak memory, input
  hashes, source commit, checkpoint hash, and report hash.

Required controls in the one executable are:

1. unchanged oracle typed challenges;
2. frozen first-subtoken metrics loaded from the immutable report;
3. shuffled challenge outcomes;
4. whole committed-lineage swaps;
5. class-message-zeroed inference; and
6. exact-surface class-ID reindexing, which must be bit-identical.

The class-message-zeroed result is diagnostic; it does not need to lose for
the treatment to pass. Class-ID reindexing is a hard correctness gate.

## 6. Frozen pass/kill gate

Every condition must pass:

1. 100% representability of every labeled alias occurrence under width four;
2. development answer, all-eight tuple, and selected-table exactness >=95%;
3. lexical-shift answer, all-eight tuple, and selected-table exactness >=90%;
4. lexical-shift exact gold-mention boundaries and roles >=90%;
5. every lexical-shift family-by-depth cohort answer >=90%;
6. development and lexical-shift oracle answer >=98%;
7. shuffled-outcome and lineage-swap drops each >=20 points on both splits;
8. zero accepted partial or strict-superset spans as exact identity copies;
9. exact-surface class-ID reindexing changes no decoded tuple, selected table,
   or answer; and
10. all hashes, seeds, budget, and source-access receipts match this document.

A pass preserves this whole-mention compiler as the language front end for
DIVERGE-v0 board construction. A miss closes this exact span architecture and
records whole-mention grounding as an unresolved prerequisite. In either case,
the next architecture work is DIVERGE-v0 CPU mechanics; this gate is not
extended.

## 7. Ordered successor

After this result is immutable, begin DIVERGE as a source-sealed factorized
extension of protected CSDC. PCSD and FCPT remain closed. The first DIVERGE
deliverable is CPU-only packet semantics, exact enumeration parity, verified
nogoods, safe merging, query invariance/abstention, canonical accounting, and
fail-closed overflow on the Delayed Disambiguation/Recovery board. No long
pretraining or public benchmark run is authorized by this span result.

## 8. Immutable result

Status: **closed negative**. Newton job `741299` completed on `evc30` with
exit code zero in `00:30:02`. It trained the frozen 9,495,302-parameter span
adapter for exactly 1,500 updates / 192,000 charged episodes. Training took
1,572.116 seconds at 122.128 episodes/s and reported 10,029,806,592 peak GPU
bytes. The one hard minibatch at update 1,200 recovered immediately; no OOM,
nonfinite value, or runtime failure occurred.

| Split | Answer | Complete tuple | Selected table | Gold mention | Oracle |
|---|---:|---:|---:|---:|---:|
| Development | 99.691% | 100.000% | 99.202% | 100.000% | 99.691% |
| Lexical shift | **84.294%** | **17.920%** | **74.447%** | **83.021%** | 99.463% |

All six development cohorts pass. On lexical shift, the weakest answer cohort
is 77.832%. Shuffled outcomes reduce shifted answers to 51.807%, and whole-
lineage swaps reduce them to 12.500%; the source challenges and coherent CSDC
lineage are therefore still causally used. Exact-surface class-ID reindexing
is bit-identical. No partial, strict-superset, or overlapping mention is ever
accepted as an exact identity copy.

The failure is not tokenizer representability: every gold alias fits under
the frozen width-four cap. It is joint role/identity generalization. Across
the six shifted cohorts the strict decoder records 16,625 duplicate outcomes,
5,050 duplicate starts, 1,984 missing outcomes, 2,496 missing starts, 116
nonexact identities, and 223 selected partial spans that are correctly
rejected. Shifted start/outcome mentions remain about 93.2%/93.4% exact, but
ordered WORD mentions fall to 78.1%, so complete challenge records survive in
only 17.9% of episodes. Exact-surface occurrence pooling solves neither unseen
nominal grounding nor globally coherent role assignment.

The report/checkpoint SHA-256 values are
`d81a1c9648b10f8afb409116463b3ca8b5084abc472a33cd4922d0e5d17ebcca` /
`a2b16103dcc63d1a1b08ac9e24be23520066b5cd772feb8e679b21e9a315b19b`.
Both artifacts are read-only and hash-verified on Newton and in local disaster
recovery storage. This exact span architecture receives no repair run.
DIVERGE-v0 begins from protected typed/role-copy CSDC and treats learned
raw-language fault-line compilation as an unresolved later gate.
