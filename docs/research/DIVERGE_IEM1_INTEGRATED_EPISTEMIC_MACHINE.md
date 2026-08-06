# DIVERGE-IEM1: Integrated Epistemic Machine

Status: frozen before implementation, data materialization, or neural result on
2026-08-06.

## 1. Decision

TOL3, TFS1, and NVE1 now qualify the separate source, factorized-state, and
natural-evidence interfaces. Their composition is exact, but it is assembled
from separately trained checkpoints and an exact query parser. DIVERGE-IEM1 is
the one authorized integration gate:

> Train one model-owned semantic machine, in one optimizer and one checkpoint,
> to ground source operations, bind delayed evidence, dispatch a tied recurrent
> state algebra, bind a late natural query, and drive one coherent factorized
> commitment.

IEM1 is not continuation pretraining, a public benchmark, or an unrestricted
reasoning claim. NVE1 receives no variants. The protected TOL3, TFS1, and NVE1
results remain immutable controls.

## 2. Architecture

### Shared whole-mention encoder

Start from the exact NVE1 byte embedding, two-layer bidirectional GRU, and
normalization at checkpoint SHA-256
`1610815471c695b0d2d198922dd99369e1f45a5dabc1b1c5d8e986b30fd200ff`.
All weights remain trainable. The same position-free encoder serves four
interfaces:

1. local source-operation and comparator phrases;
2. complete natural evidence sentences with two numeric mentions and two
   source-owned symbol quotient groups;
3. complete natural queries with two source-owned symbol quotient groups; and
4. tied operation/comparator dispatch into the recurrent state algebra.

Evidence retains NVE1's hard `STEP/VALUE` and `TARGET/DISTRACTOR`
permutations. Query binding adds one hard `TARGET/DISTRACTOR` permutation. No
field is decoded independently and no wrong hard assignment is repaired.

### Learned semantic dispatch

Direct operations use four latent channels and a learned doubly normalized
`4 x 4` transport into the fixed rational primitives `SET`, `ADD`, `SUBTRACT`,
and `MULTIPLY`. Comparators use six latent channels and an analogous `6 x 6`
transport. Source logits and transport parameters are co-trained; only their
composed primitive distribution receives semantic supervision. Hard inference
chooses one latent channel and one transported primitive. `SWAP` and `QUERY`
remain explicit structural operations.

The exact rational arithmetic basis, guarded source-order scheduler, complete
state representation, and factorized support engine are architecture
primitives. They are not hidden assessors and they receive no answer. Their
dispatch and all language-to-state bindings are learned. Incompatible worlds
remain complete and are never averaged fieldwise.

### Sealed execution and readout

The runtime path is:

```text
natural WORLD
  -> learned source grounding and semantic dispatch
  -> sealed binary fault-line packet
  -> shared factorized recurrent execution
  -> learned natural-evidence receipts and monotone refinement
  -> coherent terminal state groups
  -> learned natural QUERY binding
  -> exact value read when every survivor agrees, otherwise ABSTAIN
```

After source compilation, raw WORLD bytes and source hidden state are absent
from evidence refinement and query readout. The query may select a source-owned
register but may not alter support or execution.

## 3. Training and data

The one fit uses seed `2026080614`, 1,000 AdamW updates, batch 256 per sampled
natural interface, learning rate `1e-3` with cosine decay, and one H100 when
available. Every update contains:

- one balanced pass over the deduplicated TOL3 operation/comparator examples;
- 256 NVE1 natural-evidence examples;
- 256 deterministic natural-query examples; and
- a balanced semantic-dispatch loss over all direct operations and
  comparators.

Evidence and query losses are class-balanced hard-role cross entropy. Source
loss is class-balanced primitive cross entropy after latent transport.
Transport receives an exact row/column balance penalty; no identity
initialization or gold hard mapping is supplied. Loss weights, update count,
seed, optimizer, and model width are fixed before confirmation.

The confirmation board has exactly 256 fresh TFS1 programs at seed
`2026080615`, 12 binary fault lines and 4,096 coherent worlds per program,
3,072 natural evidence sentences, and three natural queries per program.
Evidence and query renderers are absent from training. Full sentences,
program identities, and query strings have zero exact train overlap. Register
names, values, depths, operation-pair order, guards, swaps, and complete
compositions are held by episode rather than leaked as component IDs.

## 4. Matched controls

- immutable separate TOL3 + NVE1 + typed-query composition as the ceiling;
- premature highest-support top-1;
- equal-memory complete particles;
- no-evidence factorized support;
- IEM1 with semantic transport shifted;
- IEM1 with evidence role assignments swapped;
- IEM1 with query role assignments swapped;
- shuffled complete evidence sets;
- declaration-state reset;
- packet/query provenance swap; and
- post-seal WORLD/evidence poisoning.

The protected NVE1 confirmation board and TOL3 fresh confirmation board are
also regression controls for catastrophic forgetting.

## 5. Frozen pass/kill gate

All conditions are conjunctive:

1. at least 1,000/1,024 exact TOL3 confirmation programs and at least 250/256
   exact NVE1 protected recoveries after joint training;
2. at least 250/256 fresh IEM1 source programs compile exactly, with both
   options retained at every accepted fault line and no gold-support deletion;
3. at least 3,041/3,072 fresh evidence receipts and 752/768 fresh natural query
   bindings are exact, with every accepted receipt/query valid;
4. the separate immutable ceiling is 256/256 and IEM1 reaches at least
   245/256 exact sensitive answers, at least 95% exact conditional on an
   initially wrong top-1, and exact extensional parity on every answered row;
5. no-evidence support abstains on at least 245/256 sensitive queries while
   invariant queries answer and partial-evidence underdetermined queries
   abstain on all accepted rows;
6. top-1 and equal-memory particles trail IEM1 by at least 50 points;
7. semantic-transport shift, evidence-role swap, query-role swap, shuffled
   evidence, and state reset each reduce exactness by at least 50 points;
8. all packet/query swaps reject, post-seal poisoning is bit-invariant, and
   zero invalid receipt, invalid query, false commitment, malformed packet,
   gold deletion, or overflow is accepted; and
9. parameters, source bytes, update examples, checkpoint/data hashes, wall
   time, peak memory, canonical bytes, and logical/unique applications are
   reported.

A component miss closes IEM1 without a width, duration, seed, transport,
renderer, or loss variant. A composition miss closes this shared-encoder
integration. A pass qualifies one broader free-form natural-program board or
the architecture-training phase; it does not itself authorize continuation
pretraining.

## 6. Claim boundary

The candidate contribution is the integrated conjunction: whole-mention
language grounding, learned semantic dispatch, coherent factorized recurrent
state, evidence-driven monotone refinement, and late query binding in one
trainable model. Byte GRUs, latent permutations, exact ALUs, version spaces,
and abstaining marginal readout are not individually novel. Even a complete
IEM1 pass remains controlled-language evidence until it transfers to broader
natural programs and then improves real reasoning after architecture-aware
training.
