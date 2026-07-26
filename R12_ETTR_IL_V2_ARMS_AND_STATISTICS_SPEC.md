# R12 ETTR IL v2 Arms and Statistics Specification

**Protocol:** `R12-ETTR-IL-v2`  
**Status:** design-only specification; no data materialization, model fit,
development read, confirmation read, job, or training is authorized  
**Scope:** equal-budget learning arms, repetition schedule, resource accounting,
development selection, sealed confirmation, statistical testing, promotion,
rejection, and failure localization

## 1. Claim and frozen architecture boundary

The claim is that a frozen step-300k Shohin base plus a newly initialized ETTR
can learn source-deleted synthetic reasoning and transfer systematically across
rules, composition depths, renderers, and a completely withheld ontology. The
claim is not continuation pretraining, natural-language general reasoning, or
evidence that the protected Shohin checkpoint already has the capability.

The protected base is read-only and frozen:

| Item | Exact value |
|---|---:|
| Protected Shohin parameters | `125,081,664` |
| Protected checkpoint SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| ETTR compiler parameters | `21,466,377` |
| ETTR reactor parameters | `29,757,217` |
| ETTR query-reader parameters | `16,474,177` |
| Trainable architecture parameters | `67,697,771` |
| Complete-system parameters | `192,779,435` |
| Headroom below `200,000,000` | `7,220,565` |

The inspected production architecture uses `state_width=512`, `64` slots, `8`
types, `16` relation roles, `256` value codes, at most `256` edges, three
compiler layers, six reactor layers, two query-reader layers, and at most `64`
reactor steps. The base must have zero gradients and a byte-identical state
before and after every arm.

All arms consume the same 2,309,567-byte tokenizer payload with SHA-256
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`
under the runtime identity frozen by the custody specification.

Under the existing optimizer ownership rule, a two-dimensional architecture
parameter whose name contains neither `tok` nor `head` belongs to Muon; every
other architecture parameter belongs to AdamW. The treatment receipt is:

| Optimizer group | Parameters |
|---|---:|
| Architecture Muon | `67,024,896` |
| Architecture AdamW | `672,875` |
| **Unique trainable** | **`67,697,771`** |

All arms use the existing eleven objective families and exact weights:

| Objective | Weight |
|---|---:|
| token LM | `1.0` |
| initial plus terminal packet | `1.0` |
| WORLD intervention | `1.0` |
| COMMAND intervention | `1.0` |
| WORLD query binding | `1.0` |
| COMMAND query binding | `1.0` |
| transaction sequence | `1.0` |
| packet/transaction equivariance | `0.25` |
| commit/halt | `0.5` |
| sparsity | `0.01` |
| anti-bypass | `0.1` |

The query-binding margin is `1.0`; the causal LM shift is one token. No arm may
drop an objective evaluation. An ablation changes only the named causal input
or binding, never an objective weight.

## 2. Experimental units

The word `rectangle` is not used without a qualifier.

### 2.1 Semantic core

One `semantic_core` fixes one theory/evidence instance, two WORLD factors, two
COMMAND factors, one composition depth, two query semantics, and all four
corner executions. Presentation and renderer views of that same core share
those semantics. The core is the indivisible split, leakage-audit, and
statistical-bootstrap cluster.

Training emits four semantic rectangles per core arranged as two immutable
equivariance pairs. Non-`all_axes` score cells emit three views per core;
`all_axes` score cells emit four. No view of one core may cross a split.

### 2.2 Semantic rectangle

One `semantic_rectangle` contains:

- two semantic WORLD factors, `W0` and `W1`;
- two semantic COMMAND factors, `C0` and `C1`;
- all four WORLD x COMMAND terminal packets;
- two query semantics, `Q0` and `Q1`; and
- two meaning-preserving paraphrases, `P0` and `P1`, for each query.

It therefore contains four packet executions and `4 x 2 x 2 = 16` query rows.
All sixteen rows stay in one split and one microbatch.

### 2.3 Causal rectangle

For each `(Q,P)` pair, the four `(W,C)` corners form one
`causal_rectangle` with tensor geometry `[2,2]`. One semantic rectangle expands
in this exact order:

```text
(Q0,P0), (Q0,P1), (Q1,P0), (Q1,P1)
```

Within each causal rectangle, rows are ordered:

```text
(W0,C0), (W0,C1), (W1,C0), (W1,C1)
```

Every causal rectangle must independently satisfy the current
`ETTRCausalRectangle` contract:

1. one identical query prefix and read index across all four corners;
2. identical initial-packet targets along each fixed-WORLD row;
3. different initial packets between `W0` and `W1`;
4. identical terminal support masks across corners;
5. every WORLD edge changes the complete terminal target and next-token query
   label; and
6. every COMMAND edge changes the complete terminal target and next-token query
   label.

It is not sufficient for one query semantic to witness an edge that the other
does not. Failure of any of the four causal slices rejects the parent semantic
rectangle before splitting.

### 2.4 Scoring units

- `query_row_exact`: packet fields applicable to the row, full transaction
  sequence, disposition, and autonomous answer are all exact.
- `causal_rectangle_exact`: all four query rows in one causal rectangle are
  `query_row_exact`.
- `semantic_rectangle_exact`: all four causal rectangles, all four unique
  initial packets, all four unique terminal packets, and all four unique
  transaction sequences are exact.
- `semantic_core_exact`: every semantic rectangle emitted from one core is
  `semantic_rectangle_exact`.

Repeated packet executions caused by the four query slices are deduplicated by
`(semantic_rectangle_id,W,C)` for packet and transaction diagnostics. They are
not deduplicated for query scoring or training compute.

`causal_rectangle_exact` is the primary transfer endpoint.
`semantic_core_exact` is the primary fit endpoint and the statistical cluster
is the semantic core.

## 3. Folds, seeds, and initialization

The three leave-one-ontology-out folds are immutable:

| Fold | Fit ontologies | Completely withheld ontology |
|---:|---|---|
| `0` | rewrite, resource | Horn |
| `1` | Horn, resource | rewrite |
| `2` | Horn, rewrite | resource |

The withheld ontology contributes zero rows, auxiliary labels, equivariance
pairs, LM targets, or optimizer updates in its fold.

The five exact signed-63-bit model seeds are:

```text
827771697280926998
9160563446168054265
5619173084519213573
2431337583064323711
8750822315343322697
```

Every arm is fitted for every `(fold,seed)`, producing `5 arms x 3 folds x 5
seeds = 75` primary fits. Shared modules in treatment, state-reset,
binding-deranged, and query-only begin byte-identically within a `(fold,seed)`.
The dense arm uses the same seed through the tagged initializer
`SHA256("R12-ETTR-IL-v2|dense|" || fold || "|" || seed || "|" ||
canonical_parameter_name)`. No arm may load ETTR weights, optimizer tensors, or
RNG state from another fit.

## 4. Exact training stream and repetition

Each fold has exactly `576` fit semantic cores: `288` from each fit ontology,
balanced as `96` at each fit depth `1`, `2`, and `3`. Every core emits four
semantic rectangles arranged as two invariant pairs. Thus each fold has
`1,152` pair IDs and `2,304` semantic rectangles. Each semantic rectangle
expands to four causal rectangles and sixteen rows.

Each optimizer update consumes:

| Unit | Per microstep | Per update |
|---|---:|---:|
| invariant pairs | `1` | `4` |
| semantic rectangles | `2` | `8` |
| causal rectangles | `8` | `32` |
| query rows | `32` | `128` |
| accumulation microsteps | - | `4` |

There are exactly `6,000` optimizer updates, hence exactly `24,000` invariant
pair exposures, `48,000` semantic rectangle exposures, `192,000` causal
rectangle exposures, and `768,000` query row exposures per fitted model.

For each `(fold,seed)`, epoch `e` orders all `1,152` invariant pair IDs by:

```text
SHA256(
  "R12-ETTR-IL-v2|schedule|" ||
  decimal(fold) || "|" || decimal(seed) || "|" || decimal(e) || "|" ||
  invariant_pair_id
)
```

Digest ties are broken by the ASCII pair ID. Epochs `0..19` are consumed
completely, giving `23,040` pair exposures. Epoch `20` contributes exactly its
first `960` pairs. No wrap, reshuffle, dropped tail, replacement,
adaptive curriculum, or arm-specific order exists. Consecutive groups of eight
semantic rectangles do not form updates directly: each pair is one microstep
and consecutive groups of four pairs form updates.

Thus every invariant pair, and both semantic rectangles inside it, is exposed
either `20` or `21` times. The exact `960` pairs receiving the twenty-first
exposure are seed-dependent but identical across arms within that
`(fold,seed)`.

## 5. Exact optimization and resource budget

Every primary fit uses:

- BF16 forward/backward and FP32 loss accumulation;
- Muon LR `0.005` and AdamW LR `0.001`;
- Adam betas `(0.9,0.95)`, weight decay `0.0`;
- `500` linear-warmup updates;
- constant peak LR through update `4,800`;
- linear decay over updates `4,801..6,000` to exactly `0.1` times peak LR;
- global gradient clip `1.0`;
- no dropout, early stopping, retry, checkpoint averaging, or best-checkpoint
  selection; and
- only the post-update-`6,000` checkpoint as claim-bearing.

All serialized WORLD, COMMAND, and QUERY segments are padded, never truncated,
to exactly `192`, `96`, and `48` tokens. A row exceeding a bound invalidates
the board before training. Transaction targets have exactly 64 positions,
matching the frozen reactor horizon. One semantic operation may compile to
multiple generic packet edits; the valid prefix contains all materialized
command-register, ontology-state, cursor, outcome, and final-disposition
transactions, followed by right padding. Padding masks are part of the
manifest and objective. A valid trace longer than 64 is inadmissible and is
never truncated.

The existing full-objective accounting executes `WORLD + 2*COMMAND + 3*QUERY`
per row. Therefore each model has:

| Budget | Exact value |
|---|---:|
| encoded tokens per row exposure | `528` |
| encoded tokens per update | `67,584` |
| encoded tokens per fitted model | `405,504,000` |
| factual padded segment tokens per fitted model | `258,048,000` |
| supervised non-reset token positions per fitted model | `255,744,000` |
| optimizer updates per fitted model | `6,000` |
| trainable parameters per fitted model | `67,697,771` |

The preregistered charged-FLOP convention is exactly `6*N*D`, where `N` is the
unique trainable parameter count and `D` is encoded tokens, including factual
and intervention calls. The exact charged budget is therefore:

```text
6 * 67,697,771 * 405,504,000
= 164,710,301,589,504,000 charged FLOPs per fitted model
```

Across the 75 primary fits, the budget is `450,000` updates,
`30,412,800,000` encoded tokens, `57,600,000` query-row exposures, and
`12,353,272,619,212,800,000` charged FLOPs.

Charged FLOPs are an equal-budget ledger, not a claim about vendor instruction
counts. Before source freeze, an operator-level static forward/backward FLOP
ledger must also be produced for one microstep of each arm using the exact
shapes above. Treatment defines the target. State-reset, binding-deranged, and
query-only must be bit-equal to that count. The dense arm must equal it exactly,
not within one percent. Its tied active compute equalizer may apply additional
parameter-free orthogonal transforms whose outputs are added to the dense
hidden state with frozen nonzero coefficient `2^-20`; every counted operation
must therefore lie on the loss path. Dummy detached work is forbidden. A
negative remainder or unequal integer count makes the dense arm and protocol
inadmissible before any fit.

## 6. Five equal-budget arms

### A0: ETTR treatment

The architecture, packet, recurrent update, reader, objectives, and hard
transaction path are unchanged.

### A1: state-reset

All model calls and losses are unchanged. Immediately before each of the 64
recurrent transaction positions, the recurrent packet input is replaced by a
fresh clone of the compiler's sealed initial packet with `step=0`,
`committed=0`, and `halted=0`. The policy is evaluated and its transaction is
applied once to that clone. Only the output of transaction position 64 is
passed as the terminal packet to the reader and terminal losses. Earlier
one-step outputs remain attached to their transaction losses but cannot become
later inputs. Target losses remain masked by the factual 64-position
right-padded transaction mask.

The reset is tensor reassignment, not a skipped reactor call. This arm has the
same parameters, rows, calls, targets, objective weights, updates, tokens, and
FLOPs as A0. It isolates recurrent state carry.

### A2: binding-deranged

Candidate WORLD, COMMAND, and QUERY bytes remain factual. One deterministic
permutation reassigns the entire target bundle of each training semantic
rectangle, including initial packets, factual and intervention transaction
sequences, terminal packets, dispositions, autonomous answer labels, and
equivariance alignments.

The matching key is:

```text
(fold, ontology, depth, renderer, presentation,
 query_semantic_pair_signature, paraphrase_pair_signature,
 initial_support_shape, terminal_support_shape, transaction_mask)
```

The answer is deliberately absent from the key. For recipient `i`, donor `j`
is admissible only if:

1. `i != j`;
2. all geometry and support tensors named in the key agree;
3. each of the four donor terminal packets differs from the corresponding
   recipient terminal packet;
4. each of the four donor transaction sequences differs from its recipient;
5. all sixteen donor autonomous answer labels differ from their corresponding
   recipient labels; and
6. the donor bundle itself passes all four causal-rectangle validators.

Within each key, recipients are ASCII-sorted by semantic rectangle ID. Donors
are ranked for each recipient by:

```text
SHA256(
  "R12-ETTR-IL-v2|binding-derangement|" || decimal(fold) || "|" ||
  recipient_id || "|" || donor_id
)
```

Among all admissible perfect matchings, choose the lexicographically smallest
vector of donor ranks in recipient order. This defines a total, deterministic
no-fixed-point derangement without a random retry. If any key lacks a perfect
matching, the training board is inadmissible; groups may not be merged after
inspection.

This resolves the v1 contradiction: the control preserves marginal geometry,
not the answer, while every donor changes packet, trajectory, and assessor
answer. The schedule follows recipient IDs, so A2 receives exactly the same
source rows and exposure order as A0.

### A3: query-only

The compiler and reactor execute normally and receive every non-query loss.
For every factual, WORLD-intervention, COMMAND-intervention, correct, and foil
reader call, the reader receives the canonical empty packet:

```text
all value/type/relation/active/root/committed/halted tensors = 0
step = 64
```

The true packet remains attached to packet, transaction, intervention,
equivariance, commit/halt, sparsity, and anti-bypass losses. Query LM and both
query-binding losses are evaluated unchanged through the empty packet. No call,
loss, parameter, token, or update is removed. At scoring, A3 again receives
only the canonical empty terminal packet.

### A4: parameter-matched dense-state control

A4 keeps the exact A0 compiler, hard generic transaction interface, query
reader, packet geometry, objectives, and all reactor tensors except
`relation_projection` and the six-layer typed transformer core. Those removed
tensors contain exactly `27,302,912` parameters. The retained reactor tensors
contain exactly `2,454,305`.

The replacement is a favorable dense continuous controller:

1. A fixed normalized signed feature-hash sketch maps every initial packet
   field, including every relation-ledger cell, to `512` floats. The sketch
   signs and column assignments are derived from
   `SHA256("R12-ETTR-IL-v2|dense-sketch|" || flat_field_index)` and have no
   trainable parameters.
2. The sketch is added to the unchanged non-relational packet-control path,
   which continues to use the retained type/value embeddings, active/root/status
   projections, control seed, and step embedding. A `512 -> 1,241`
   packet-initialization linear map is added to each learned layer initial
   state. The GRU receives that complete packet control plus the unchanged
   command-attention result at every transaction position.
3. The two-layer GRU has input width `512` and hidden width `1,241`.
4. A GELU residual MLP has geometry `1,241 -> 4,123 -> 1,241`.
5. A `1,241 -> 512` map feeds the unchanged transaction heads. Predicted hard
   transactions update the same packet shell consumed by the unchanged reader.
6. Both layer initial states are learned and every replacement parameter has a
   nonzero path to at least one transaction logit. Every retained reactor
   tensor remains on its corresponding non-relational input or transaction-head
   path.

The exact replacement count is:

| Dense replacement component | Parameters |
|---|---:|
| two-layer GRU | `15,789,243` |
| packet initialization `512 -> 1,241` | `636,633` |
| output map `1,241 -> 512` | `635,904` |
| residual MLP `1,241 -> 4,123 -> 1,241` | `10,238,650` |
| two learned initial states | `2,482` |
| **Replacement** | **`27,302,912`** |
| retained reactor tensors | `2,454,305` |
| **Dense reactor** | **`29,757,217`** |

To preserve the treatment's optimizer-family budget, exactly `7,999 = 19 x
421` functionally active residual-projection weights are stored as
`dense_head_adapter.weight` and applied by fixed gather/scatter maps. The
remaining dense matrices are non-head tensors. This partition yields exactly
`27,262,976` Muon and `39,936` AdamW parameters in the replacement, matching
the removed tensors. Consequently A4 has exactly `67,024,896` Muon,
`672,875` AdamW, `67,697,771` trainable, and `192,779,435` complete-system
parameters. No mask, unused row, frozen padding tensor, or non-loss-bearing
parameter counts toward matching.

A4 is a valid favorable control only if every seed/fold reaches at least `99%`
fit `semantic_core_exact` and at least `90%` `seen_id`
`causal_rectangle_exact` in both fit ontologies. Failure to qualify A4 blocks
the typed-sparse advantage claim; it does not waive any A0 threshold.

## 7. Noncompetitive localization ceilings

The A0 update-6000 checkpoint is additionally evaluated without fitting:

- `gold-initial`: replace the compiler packet by the assessor's gold initial
  packet, then run the learned reactor and reader.
- `gold-terminal`: replace the terminal packet by the assessor's gold terminal
  packet, then run the learned reader.

They use the same query prefixes and autonomous decoding. They are diagnostic
ceilings, not arms, receive no optimizer update, and cannot satisfy a treatment
comparison.

## 8. Development access and selection

All 75 update-6000 checkpoints, optimizer receipts, schedules, source hashes,
parameter receipts, and budget ledgers must be immutable before development is
mounted. Updates `1,000` and `3,000` may be logged from training only; they may
not select a checkpoint, seed, arm, loss, decoder, or schedule.

Development is opened exactly once and scored in one atomic evaluation of all
arms, folds, seeds, ceilings, and controls. Development performs only the
binary selection:

```text
OPEN_CONFIRMATION
or
CLOSE_V2_WITHOUT_CONFIRMATION
```

There is no hyperparameter choice, seed removal, four-of-five rule, checkpoint
choice, arm repair, calibration, threshold fitting, or decoder choice. All five
seeds in all three folds must pass. Exactly four passing seeds is a failed
stability gate, not authorization to continue.

## 9. Absolute scientific gates

The following point-estimate gates apply separately to development and, if
authorized, confirmation. Unless stated as a five-seed mean, each gate must
hold for every seed and fold.

### 9.1 Treatment gates

1. Fit `semantic_core_exact >= 99%`.
2. `seen_id` `causal_rectangle_exact >= 95%` in each fit ontology.
3. Each fit ontology's `rule`, `composition`, and `renderer` cell is at least
   `85%`.
4. Each fit ontology's `rule_composition`, `rule_renderer`, and
   `composition_renderer` cell is at least `80%`.
5. Each fit ontology's `all_axes` cell is at least `75%`.
6. The completely withheld ontology's eight-cell macro is at least `80%`, and
   its `all_axes` cell is at least `70%`.
7. Meaning-preserving invariant agreement is at least `99%`.
8. Order/noncommuting twins and changed-answer semantic twins are each at least
   `95%` correctly separated.
9. Ambiguity and contradiction disposition accuracy is at least `95%`.
10. WORLD- and COMMAND-intervention autonomous answers are each at least `90%`
    exact and equal the independently replayed counterfactual target.
11. Query-twin joint accuracy from one sealed packet is at least `90%`.
12. Post-seal deletion, poisoning, and source replacement invariance is exactly
    `100%`.
13. Learned initial-packet exactness is at least `95%` in every ontology.
14. `gold-terminal >= 99.5%` and `gold-initial >= 99%`.

ABSTAIN and REJECT rows are included in joint denominators and also reported
separately. No binary target balancing denominator includes them.

### 9.2 Matched-control margins

For each of A1-A4, A0 must exceed the control by:

- at least `10.00` percentage points in overall scored
  `causal_rectangle_exact`; and
- at least `5.00` points in every exact
  `(fold,ontology,stratum)` cell.

The direction and margin must hold separately for all five seeds, not only for
the seed mean. Equality at the boundary passes the point gate; the simultaneous
statistical lower bound below must be strictly greater than the boundary.

## 10. Statistical protocol and multiplicity

The scored population is finite, but the transfer claim concerns the
deterministic generator population. Dependence among query slices,
presentations, renderers, and rows is handled by resampling only whole
semantic cores.

For arm `a`, seed `s`, and semantic core `g`, let `V[g]` be its emitted view
count and define:

```text
Y[a,s,g] =
  number of exact causal rectangles across all views of g / (4 * V[g])
```

For each control, all differences are paired on the same semantic core, fold,
and seed.

A cell effect is the arithmetic mean paired A0-control difference across its
five seeds and all semantic cores in the cell: 32 cores for every
non-`all_axes` cell and 24 cores for `all_axes`. The overall effect is the
unweighted macro mean of the 72 `(fold,ontology,stratum)` cell effects; it is
not a pooled row, rectangle, or presentation mean. The withheld-ontology
eight-cell macro in Section 9 is likewise an unweighted mean of its eight
stratum accuracies.

The confirmatory comparison family contains exactly:

- four A0-control overall effects; and
- four A0-control effects in each of `3 folds x 3 ontologies x 8 strata`.

This is `4 x (1 + 72) = 292` one-sided superiority endpoints. No endpoint may be
added after development.

Use exactly `100,000` hierarchical paired bootstrap replicates. The bootstrap
seed stream is counter-mode SHA-256 from:

```text
SHA256(
  "R12-ETTR-IL-v2|bootstrap|" || split_plaintext_sha256 ||
  "|" || decimal(replicate_index)
)
```

Each replicate:

1. resamples the five model seeds with replacement, once globally and shared
   by every arm, fold, cell, and endpoint;
2. keeps the three leave-one-ontology-out folds fixed;
3. independently resamples semantic cores with replacement inside each
   `(fold,ontology,stratum)` cell, drawing 32 for non-`all_axes` and 24 for
   `all_axes`;
4. carries every view, all four causal slices, and every arm together; and
5. recomputes all 292 paired effects.

Let `theta_hat[e]` be the observed effect and `theta_star[b,e]` its bootstrap
value. For replicate `b`, compute:

```text
M[b] = max_e(theta_hat[e] - theta_star[b,e])
```

Let `q_0.95` be element `95,000` after sorting the 100,000 `M` values in
nondecreasing order using one-based indexing. The simultaneous one-sided 95%
lower bound is:

```text
LCB[e] = theta_hat[e] - q_0.95
```

This single-step max-deviation construction controls the familywise error rate
at `0.05` across all 292 comparisons. Unadjusted row, causal-rectangle, cell,
seed, or control p-values are descriptive and cannot promote. There is no
separate favorable-subgroup family and no post hoc pooling.

Development applies the same computation only as a screening gate and cannot
create a claim. Confirmation is the sole claim-bearing statistical analysis,
so no alpha is spent or recycled across the two stages. Confirmation promotion
requires every overall `LCB > 0.10` and every cell `LCB > 0.05`, in addition to
all point-estimate gates. The strict inequality is intentional.

Absolute accuracy thresholds are noncompensatory finite-board criteria, not
null-hypothesis tests. Because the global decision requires their conjunction,
none is rescued by averaging or multiplicity adjustment.

## 11. Sealed confirmation

Before any fit, the independent confirmation plaintext and manifest must be
materialized by the external data custodian, then encrypted with
XChaCha20-Poly1305 using a fresh 32-byte key and fresh 24-byte nonce from the
custodian CSPRNG. Associated data is the canonical ASCII manifest containing
the protocol ID, source inventory, split-spec hash, plaintext SHA-256, row and
cell counts, tokenizer hash, and development exclusions. The ciphertext,
nonce, associated-data SHA-256, and ciphertext SHA-256 are signed with the
custodian's frozen Ed25519 key. The decryption key is absent from every
training and development environment.

`OPEN_CONFIRMATION` must bind:

- all 75 immutable update-6000 checkpoint hashes;
- all development score and statistical artifact hashes;
- the exact evaluator/model/source inventory;
- the signed ciphertext identity;
- the decoder and autonomous readout identity; and
- confirmation access count zero.

One evaluator invocation atomically claims an `O_EXCL` opening ledger, verifies
the signature and associated data, decrypts in private memory, verifies the
plaintext hash, evaluates every arm/fold/seed and both ceilings, computes the
frozen statistics, writes one no-replace result, zeroizes the key/plaintext,
and seals access count one. A crash after the opening claim is an invalid
confirmation, not authority for a second opening. No logits, row-level answers,
or plaintext leave the evaluator.

Confirmation cannot change a checkpoint, seed, threshold, arm, donor,
bootstrap stream, decoder, or cell. Any rescore, partial arm opening, or second
decryption invalidates the protocol.

## 12. Promotion, rejection, and localization

### 12.1 Promotion

Promotion requires:

1. every custody, isolation, hash, source-deletion, parameter, optimizer,
   update, token, charged-FLOP, and static-FLOP receipt is exact;
2. A4 qualifies in every seed and fold;
3. all development absolute, margin, and simultaneous-bound gates pass;
4. one authorized confirmation opening then passes the identical gates;
5. all five seeds and all three folds pass without exclusion; and
6. the three previously frozen score-only boards are evaluated once after
   confirmation with unchanged checkpoints and pass the exact gates below.

For every A0 seed/fold checkpoint, the 2,688-row seven-variant matrix requires:

- at least `95%` exactness on identifiable cases in each ontology;
- exactly `100%` correct ABSTAIN on ambiguous cases, REJECT on contradictory
  cases, and coherent-alternate behavior;
- exactly `100%` aligned alpha-reorder, alias-split, and relation-reification
  invariance;
- at least `95%` execution-semantic-twin and noncongruent-twin separation; and
- at least a `20.00`-point exactness advantage over each qualified matrix
  control.

The 48-row direct WORLD-COMMAND-QUERY board requires at least `46/48` exact
autonomous answers overall, at least `15/16` in each ontology, `48/48`
query-twin target separation, and `48/48` correct WORLD/COMMAND
counterfactual-donor outputs. The 48-case hybrid board requires at least
`41/48` exact autonomous outputs overall and at least `13/16` in each of its
three coupling families. Physical deletion and post-seal poisoning invariance
must be exact on all three boards. These finite-board gates are a conjunction;
they are not an additional statistical family.

The sole positive decision string is:

```text
ettr_isolated_synthetic_learnability_systematic_transfer_and_typed_state_advantage_confirmed
```

### 12.2 Invalid evidence

Any custody, leakage, source, target, schedule, parameter, optimizer-family,
update, token, FLOP, seed, fold, derangement, confirmation, or multiple-
comparison mismatch yields:

```text
ettr_isolated_learnability_v2_invalid_no_capability_claim
```

Invalid evidence does not localize an architecture failure. The defect must be
repaired under a new protocol and untouched board.

### 12.3 Scientific rejection

With valid custody, failure of any development gate closes v2 without opening
confirmation. Failure of any confirmation gate rejects v2. More updates,
threshold relaxation, selective seeds, rescoring, and reuse of the opened
confirmation board are forbidden.

All applicable localization labels are reported. One primary label is chosen
by the following deterministic precedence:

1. `control_qualification`: A4 misses its fit or `seen_id` qualification.
2. `query_reader`: gold-terminal is below `99.5%`.
3. `reactor_transaction_interface`: gold-terminal passes but gold-initial is
   below `99%`.
4. `world_compiler`: both ceilings pass but learned initial-packet exactness is
   below `95%` in any seed/fold.
5. `recurrent_state_carry`: A0 fails the A1 point margin or simultaneous bound.
6. `world_command_binding`: A0 fails the A2 point margin or simultaneous bound.
7. `query_bypass`: A0 fails the A3 point margin or simultaneous bound.
8. `typed_sparse_state_advantage`: A0 fails the A4 margin while both A0 and A4
   pass all absolute transfer gates.
9. `systematic_rule_composition_renderer_transfer`: fit passes but any fit-
   ontology single-axis, two-axis, all-axes, twin, or invariance gate fails.
10. `withheld_ontology_transfer`: any withheld-ontology macro or all-axes gate
    fails.
11. `architecture_native_causal_use`: factual accuracy passes but WORLD,
    COMMAND, query-twin, or independent counterfactual effects fail.
12. `seed_stability`: exactly four or fewer of five seeds pass any otherwise
    satisfied gate.
13. `confirmation_nonreplication`: development passes but confirmation reverses
    a required direction, misses a threshold, or misses a simultaneous bound.

The rejection string is:

```text
ettr_isolated_learnability_v2_rejected_at_<primary_localization>
```

Passing absolute learnability thresholds while A4 matches A0 may be described
only as parameter-matched dense learnability. It is not an ETTR typed-sparse
architecture promotion under this protocol.

## 13. No authorization

This specification freezes design choices only. Training remains unauthorized
until a separate v2 source and data manifest resolves the remaining generator,
AST/renderer, command-depth, materialization, overlap, and leakage-audit
requirements and binds literal artifact hashes. No code, job, dataset, score,
or confirmation access is created by this document.
