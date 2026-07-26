# R12 ETTR Isolated Learnability Preregistration

**Protocol:** `R12-ETTR-IL-v1`  
**Status:** preregistered design only; no dataset materialization, training, or
scored access authorized by this document  
**Question:** can completed ETTR acquire a synthetic, source-deleted learned
capability while the step-300k Shohin base remains frozen?  
**Absolute prohibition:** no continuation pretraining, flagship shard access,
flagship optimizer access, or write into any flagship checkpoint/output path

## 1. Claim boundary

This is the first isolated learned-capability experiment for the completed
Endogenous Typed Theory Reactor (ETTR). It is not continuation pretraining and
does not test natural-language general reasoning. It tests whether one shared
ETTR can learn to:

1. compile a raw `WORLD` into its categorical packet;
2. lose all access to the raw world, residuals, and KV state;
3. execute a separately disclosed `COMMAND`;
4. lose all access to the command package;
5. answer a separately disclosed `QUERY`; and
6. transfer across unseen rules, longer compositions, unseen renderers,
   combinations of those shifts, and an entirely withheld ontology/task
   family.

A pass establishes bounded synthetic systematic reasoning by ETTR. It does
not establish unrestricted intelligence, natural-language transfer, or a
justification to claim that Shohin already reasons generally.

## 2. Frozen system identity

| Item | Frozen value |
|---|---|
| Repository commit at preregistration | `854d3d4d63ee467fc6bed10ffee8fdd2d97a07d1` |
| Protected Shohin checkpoint | step 300,000 |
| Protected checkpoint SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| Protected parameters | 125,081,664 |
| ETTR parameters | 67,697,771 |
| Complete parameters | 192,779,435 |
| Headroom below 200M | 7,220,565 |
| ETTR architecture source SHA-256 | `7b8a1f98267268240766775c558d9f3e98cd62c680a181993fd5be477ea9cd0a` |
| Episode runner SHA-256 | `daf47408eb7db53c4a2e2e50d8490d4900ab7014b0dc9bb6721c9e73a058a7d3` |
| Objective source SHA-256 | `94c4112bb6861fa7e4e89889c09bdb55730866ae0c4a73a39bbd5a1e02975bb8` |
| Qualification source SHA-256 | `00ad5f07f80bd14008fa68dc85c3646c49b44e4df95cf7b06131e71b47b25920` |
| Stage supervisor SHA-256 | `4a2511a37be5f24501e26d0aa976e1c2d9f92cbf02bff87c1c7bc65685b63207` |

The base is loaded read-only, put in evaluation mode, and frozen. All base
gradients and parameter deltas must be exactly zero. The 67,697,771 ETTR
parameters are initialized independently for each declared model seed; no
ETTR weights or optimizer tensors from another experiment may be loaded.

## 3. Absolute isolation contract

The experiment launcher must fail before CUDA initialization unless all of
the following are true:

- the only model input is the immutable protected checkpoint above;
- the only data inputs are the frozen synthetic manifests defined below;
- no path containing a flagship shard, `flagship_out`, a continuation
  manifest, or a prior optimizer state is mounted;
- no optimizer file from step-300k or any continuation run is opened;
- outputs are confined to a new
  `train/ettr_isolated_learnability_v1/<fold>/<arm>/<seed>/` root;
- the protected checkpoint and every source file remain hash-identical;
- filesystem-open accounting reports zero reads from flagship data/shard
  roots and zero writes outside the isolated output root; and
- no code path calls a pretraining data loader or continuation trainer.

Any violation invalidates the entire run. It is not repaired by deleting the
offending artifact afterward.

## 4. Frozen semantic board

### 4.1 Ontologies

The board contains exactly three assessor-side task families:

1. typed Horn closure;
2. typed term rewriting; and
3. guarded resource processes.

Candidate bytes contain no family name, ontology label, theory index, split
name, renderer index, oracle output, or expected answer.

The exact fit and score-only theory pools are:

| Ontology | Fit theory indices | Score-only theory indices |
|---|---|---|
| Horn | `1,3,4,6,8,9,11,13,14,16,17,18` | `0,2,5,7,10,12,15,19` |
| Rewrite | `0,4,5,7,9,12,14` | `1,2,3,6,8,10,11,13` |
| Resource | `1-6,8-13,15-20,22-29,31-38,40-47,49-58` | `0,7,14,21,30,39,48,59` |

No score-only theory, behavior signature, or exact rule combination may
appear in fitting.

### 4.2 Leave-one-ontology-out folds

| Fold | Fit ontologies | Completely withheld ontology |
|---:|---|---|
| 0 | rewrite, resource | Horn |
| 1 | Horn, resource | rewrite |
| 2 | Horn, rewrite | resource |

The withheld ontology contributes no fitting row, auxiliary label, language
model target, renderer-consistency pair, or optimizer update in that fold.

### 4.3 One semantic rectangle

The atomic unit is a 2x2 `WORLD x COMMAND` rectangle:

- two worlds share one theory but differ in initial state;
- two commands share one command grammar but differ in the operation or
  ordered operation sequence;
- all four terminal packets are distinct;
- both WORLD edges and both COMMAND edges change at least one answer;
- each packet receives two semantic queries;
- each semantic query has two meaning-preserving paraphrases; and
- the four query rows for one packet have an identical terminal packet.

Thus one rectangle contains four packets and sixteen query rows. Rectangles
that do not satisfy every contrast are rejected before splitting.

Composition depth is the number of dependent command operations. Fit depths
are exactly 1, 2, and 3. Held-out depths are exactly 4, 5, and 6.

### 4.4 Renderers and structural presentations

Renderer IDs 0 and 1 are fit-only lexical encodings. Renderer 2 is
development-only. Renderer 3 is development and sealed-confirmation
score-only. They are respectively canonical JSON, prefix S-expression,
record-delimited infix, and reverse/postfix encodings of the same anonymous
typed syntax tree. Complete renderer IDs are disjoint; no scored renderer
template or literal separator occurs in fitting.

Fitting uses only `base`, `alpha_reorder`, and `alias_split` presentations.
The all-axes score cell additionally uses score-only
`relation_reification`, `type_twin`, `execution_semantics_twin`, and
`ambiguity_deleted_twin` presentations. Invariance variants retain the same
answer; semantic and ambiguity twins use their independently computed changed
answer or `ABSTAIN`.

### 4.5 Exact split sizes

For each fit ontology, training contains 1,152 rectangles: 384 at each depth
1, 2, and 3. Every rectangle contributes all sixteen rows.

| Population, per fold | Rectangles | Packets | Query rows |
|---|---:|---:|---:|
| Fit ontology A | 1,152 | 4,608 | 18,432 |
| Fit ontology B | 1,152 | 4,608 | 18,432 |
| **Training total** | **2,304** | **9,216** | **36,864** |

Development and confirmation each contain eight strata for every ontology:

`seen_id`, `rule`, `composition`, `renderer`, `rule_composition`,
`rule_renderer`, `composition_renderer`, and `all_axes`.

Each ontology/stratum cell contains exactly 96 rectangles, 384 packets, and
1,536 query rows. Therefore each scored split contains:

| Population, per fold | Rectangles | Packets | Query rows |
|---|---:|---:|---:|
| One ontology, eight strata | 768 | 3,072 | 12,288 |
| **Three ontologies** | **2,304** | **9,216** | **36,864** |

Development and confirmation use independent seed domains and share no
semantic world, command sequence, opaque symbol, raw row, or 13-gram.

## 5. Deterministic generation and hash custody

The master data commitment is:

`3b796eef284f523a125a18f5c94ae01b1d8305723751c9710086d670d35867aa`

It is SHA-256 of the ASCII string:

`R12_ETTR_ISOLATED_LEARNABILITY_V1|2026-07-26|source-deleted`

The canonical split-specification SHA-256 is:

`d98a6895ec11a52bce5625b8c7aa85c7be755d52d4d5082ab7d3a34b2dedff10`

The fold commitments are:

| Fold | Split-spec SHA-256 |
|---:|---|
| 0 | `66e1039afba221c9e591ec171c89107c1243c20ba00e85641164cd0be08e83c9` |
| 1 | `abff905152652119039f83cac199288b59a8ef4b744db9e73e496a41bef3113b` |
| 2 | `f82c82c0475c2d10b4c45c30bd07f1cb3282951fd825b6f1f0a77d029baacfeb` |

For every domain, candidate tuples are ordered by:

```text
SHA256(master_commitment_bytes || "|" || fold || "|" || split ||
       "|" || ontology || "|" || stratum || "|" || canonical_tuple_bytes)
```

The lowest admissible unique tuples are selected until the exact cell count is
reached. Canonical JSON is ASCII, sorted-key, compact-separator JSON followed
by one newline. File SHA-256 is over literal bytes. Dataset SHA-256 is over
the sorted list of `{path, bytes, sha256, row_count}` records in that same
encoding.

Before any model seed is instantiated, a CPU-only freeze process must produce
immutable train, development, and encrypted confirmation files plus one
root-signed manifest containing their literal SHA-256 values. The generator
source hash, tokenizer hash, split-spec hash, row counts, per-cell counts, and
all overlap audits are part of that manifest. This document deliberately does
not invent hashes for files that do not yet exist: absence of the signed
literal-file manifest keeps training unauthorized. The manifest may report
hashes; it may not change any split rule above.

Three already frozen boards are score-only and must never enter training:

| Board | Payload SHA-256 |
|---|---|
| Seven-variant cross-ontology matrix, 2,688 rows | `d1904b54a0fab8e59cfcb0b0dd464f5c8778e5b828907028ec8614aeae76d5d5` |
| Direct WORLD-COMMAND-QUERY factorial board, 48 rows | `18686ff7f0476b5a4432830f2a301f693833cf867656d3997a010cf17bb0149a` |
| Cross-ontology hybrid board, 48 cases | `d155f868494f9379b214028c8d7475cc2cde08192c9b3a5bbdea5a73b29f98e2` |

These boards are opened only after all arm checkpoints for all seeds and
folds are immutable.

## 6. Model seeds and optimization budget

The model-seed root is
`4d2666338e6a8d080f31058b73e7ccb6be7b1b6686db207dba0b0057630b893e`.
The five exact signed-63-bit seeds are:

`827771697280926998`, `9160563446168054265`,
`5619173084519213573`, `2431337583064323711`, and
`8750822315343322697`.

Every arm receives exactly:

- 6,000 optimizer updates;
- eight rectangles per global update;
- two rectangles per microbatch and four accumulation microsteps;
- the same SHA-ordered rectangle schedule, token rows, precision, and loss
  evaluations;
- BF16 forward/backward with FP32 loss accumulation;
- Muon for eligible matrices and AdamW for remaining architecture tensors;
- `train_base=False`;
- architecture Muon LR `0.005`, architecture AdamW LR `0.001`;
- Adam betas `(0.9, 0.95)`, weight decay `0.0`;
- 500 linear warmup updates, linear decay over the final 20% to 0.1x LR;
- gradient clip `1.0`; and
- the frozen ETTR objective weights:
  token LM `1.0`, packet `1.0`, WORLD intervention `1.0`, COMMAND
  intervention `1.0`, WORLD query binding `1.0`, COMMAND query binding `1.0`,
  transaction `1.0`, equivariance `0.25`, commit/halt `0.5`, sparsity `0.01`,
  anti-bypass `0.1`.

There is no early stopping, adaptive curriculum, retry, best-checkpoint
selection, or threshold tuning. Updates 1,000 and 3,000 are diagnostic only.
Only update 6,000 is claim-bearing.

## 7. Matched arms and ablations

All arms use the same five seeds in all three folds.

1. **ETTR treatment:** completed architecture unchanged.
2. **State-reset ablation:** identical parameters and computation, but the
   recurrent typed state is reset to the sealed initial state before every
   command step.
3. **Binding-deranged control:** identical architecture and schedule; WORLD
   and COMMAND supervision is deterministically reassociated within the same
   ontology/depth/renderer/answer stratum.
4. **Query-only control:** identical architecture and schedule; the terminal
   packet supplied to the reader is the canonical empty packet. Compiler and
   reactor still execute, preserving parameter/update/FLOP budget.
5. **Dense-state control:** compiler and reader are unchanged; typed sparse
   message/update layers are replaced by a dense recurrent block. Before
   source freeze it must be width-adjusted to match active trainable
   parameters exactly and measured forward-plus-backward FLOPs within 1%.

Arms 1-4 have exactly 67,697,771 trainable parameters. Arm 5 is inadmissible
unless its signed preflight receipt proves the same count and the FLOP bound.
No dormant padding parameters count toward matching.

Two noncompetitive ceilings are also reported:

- gold initial packet with learned reactor and reader; and
- gold terminal packet with learned reader.

They receive no optimizer advantage and cannot satisfy the treatment claim.

## 8. Anti-leakage and custody gates

Every gate is mandatory:

1. zero exact raw-row, semantic-world, theory-hash, command, opaque-name,
   graph-isomorphism, token-sequence, and normalized 13-gram overlap across
   train/development/confirmation;
2. zero score-only theory or renderer material in training;
3. exact 50/50 target balance per ontology, stratum, query semantic, and
   paraphrase, with `ABSTAIN` scored separately;
4. every wrong-WORLD, wrong-COMMAND, and shuffled-state donor changes the
   assessor target;
5. no candidate-visible family/split/renderer/theory/variant identifier;
6. a metadata-only classifier using byte length, mask length, row order, and
   package size remains at or below 52% binary accuracy and chance macro
   accuracy for multiclass targets on development;
7. no target, oracle product, answer suffix, evaluator output, or donor index
   exists in a candidate stage mount;
8. physical `WORLD -> delete WORLD package and residual/KV state -> COMMAND
   -> delete COMMAND package -> QUERY -> delete QUERY package -> assessor`
   execution;
9. only the seven allowlisted categorical state tensors and bound metadata
   cross stage boundaries;
10. post-seal source poisoning, source replacement, and source absence produce
    bit-identical terminal packets and answers;
11. the external verifier validates the exact WORLD, COMMAND, and QUERY launch
    receipts and their parent/output hash chain; and
12. development is opened once only after immutable endpoints exist;
    confirmation is opened once only after every development gate passes.

A custody or leakage failure invalidates evidence. It does not count as an
architecture failure until the custody defect is repaired and the entire run
is repeated from untouched initial seeds.

## 9. Metrics and promotion thresholds

Primary accuracy is exact joint correctness: initial packet, complete
transaction sequence, terminal packet, disposition, and query answer must all
be correct. Packet fields, transactions, halt/commit, and answer are also
reported separately.

The treatment must satisfy every condition for every seed:

- fit joint accuracy at least 99%;
- `seen_id` joint accuracy at least 95% in each fit ontology;
- single-axis `rule`, `composition`, and `renderer` joint accuracy at least
  85% per ontology;
- every two-axis joint cell at least 80% per ontology;
- `all_axes` at least 75% per ontology;
- completely withheld-ontology macro joint accuracy at least 80%, and its
  `all_axes` cell at least 70%;
- exact invariant-variant agreement at least 99%;
- noncommuting/order-twin and semantic-twin separation at least 95%;
- ambiguity/contradiction `ABSTAIN` or `REJECT` correctness at least 95%;
- WORLD and COMMAND counterfactual answers at least 90% exact;
- post-seal deletion/poisoning invariance exactly 100%;
- treatment exceeds every qualified matched control by at least ten
  percentage points overall and five points in every ontology/stratum cell;
  and
- all five seeds have the same positive treatment-control direction.

The dense control is qualified for comparison only if it reaches 99% fit and
90% `seen_id`. Failure of that control does not waive treatment thresholds.
The gold-terminal reader ceiling must be at least 99.5%, and the gold-initial
reactor/reader ceiling at least 99%.

After development passes, the unchanged gates are applied once to sealed
confirmation and to the three pre-existing score-only boards. No rescore,
prompt change, decoder change, or checkpoint choice is allowed.

## 10. Architecture-rejection and localization criteria

The completed ETTR architecture is rejected for this learned-capability target
if any of the following occurs with valid custody:

1. fewer than four of five treatment seeds reach 99% fit by update 6,000;
2. gold-terminal reader accuracy is below 99.5%: reject or redesign the query
   reader;
3. gold-initial reactor/reader accuracy is below 99%: reject or redesign the
   reactor/transaction interface;
4. gold ceilings pass but learned initial-packet exactness is below 95%:
   reject or redesign the WORLD compiler;
5. fit passes but treatment fails any single-axis threshold in three or more
   seeds: reject the claimed systematic inductive bias;
6. treatment does not beat both state-reset and binding-deranged controls by
   the required margins: reject recurrence/binding as the causal mechanism;
7. dense-state control matches or exceeds treatment in every transfer cell
   within two points: reject the claim that ETTR's typed sparse state provides
   a material generalization advantage;
8. withheld-ontology macro accuracy is below 70% or its all-axes accuracy is
   below 60% in the five-seed mean: reject cross-family generality;
9. packet/query intervention effects do not match independently computed
   counterfactuals: reject architecture-native causal use; or
10. confirmation reverses any required treatment-control direction: reject
    the result.

After an architecture-rejection outcome, more updates, larger synthetic data,
threshold relaxation, or selective seed removal are forbidden under
`R12-ETTR-IL-v1`. A successor must name the failed component, change the
architecture or objective before accessing a new sealed board, and receive a
new protocol identifier.

## 11. Decision rule

Only a complete pass yields:

`ettr_isolated_synthetic_learnability_and_systematic_transfer_confirmed`

Any scientific failure yields:

`ettr_isolated_learnability_rejected_at_<localized_component>`

Any custody, leakage, hash, budget, or equalization failure yields:

`ettr_isolated_learnability_invalid_no_capability_claim`

This preregistration authorizes no execution by itself.
