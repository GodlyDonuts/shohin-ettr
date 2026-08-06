# DIVERGE-WRA1 Whole-Record Assignment Compiler

**Status:** neural seed failed; WRA1 closed without variants

**Decision date:** 2026-08-05

## 1. Causal hypothesis

DIVERGE-SC1 is closed. Its failure is not boundary discovery: the frozen
checkpoint identifies all 1,494 audited record boundaries with zero false
positives or misses. It fails because independently classified semantic roles
and dense pair edges are multiplied into complete records. Pair precision is
only 37.162%, and the resulting Cartesian proposal set is 7.2--9.4 times too
large.

DIVERGE-WRA1 tests a different interface:

> Once the source has been partitioned into model-owned record segments,
> compile each segment through two exchangeable **complete option slots** and
> an exact-one assignment objective. A slot owns its alias, prior, ordered
> program, and source witnesses jointly. The runtime emits exactly two coherent
> options per record or fails closed; it never thresholds an `O(n^2)` pair
> graph or builds a Cartesian record proposal set.

This is a new source-interface family, not an SC1 threshold, seed, width,
duration, loss, or candidate-cap repair. The factorized DIVERGE packet,
conflict verifier, guarded execution, and late-query semantics remain
unchanged.

## 2. Frozen architecture

### Source pass and segmentation

1. Load the immutable failed-SC1 checkpoint only as a frozen source encoder and
   boundary detector.
2. Encode the complete raw source once through frozen SmolLM2 layer 17 and the
   frozen 192-wide source encoder.
3. Threshold the already-frozen boundary logits at zero.
4. Pair positive gaps in source order as `(start, end)` record segments.
5. Fail closed on an odd boundary count, empty/overlapping segments, more than
   nine records, or any segment wider than 108 words.

No gold segment, record list, renderer ID, ontology ID, query, answer,
execution state, or delayed evidence enters inference.

### Complete record object

For each predicted segment, a shared two-layer slot decoder receives:

- the frozen source states inside the segment;
- one segment summary; and
- two learned exchangeable option-slot seeds.

The two slots communicate by self-attention and attend to the segment source
states. Each slot emits one complete option object:

```text
alias_start_pointer
alias_length in {1,2,3,4}
prior_class in {favored,reserve}
program_class in {ADD->SWAP01, SWAP01->ADD, SWAP23, SWAP34}
prior_source_pointer
action_1_source_pointer
action_2_source_pointer_or_HALT
```

A separate segment head emits `fault-line` versus `background`. Training uses
the minimum cost over the two legal slot-to-gold-option permutations. It also
emits one `record_cue_source_pointer`, so the record-kind decision has an
explicit source-owned witness rather than an untraceable pooled label.
Inference uses one argmax object per slot, checks all pointer/span constraints,
rejects shared physical fields or overlapping aliases, and canonicalizes the
two options by source position. There is no retry, beam, top-k repair,
threshold sweep, or answer-guided choice.

The sealed packet retains the segment span, record cue address, and physical
source addresses and commitments for every selected option witness, then
deletes raw source bytes, source residuals, and source KV state. Exact-byte
nominal equality is computed only among the selected alias spans and never
fuses physical occurrences.

## 3. CPU mechanics gate

Before neural training, an independent CPU reference must establish:

1. 100% extensional parity for valid two-slot objects across at least 1,000
   generated records;
2. 100% reconstruction from calibrated complete-slot scores;
3. exact invariance to swapping the two exchangeable slot IDs;
4. at least a 20-point failure under fieldwise lineage shuffling;
5. duplicate/overlapping alias or witness pointers always fail closed;
6. post-seal source poisoning changes no packet semantics;
7. exactly two option objects and one record object per accepted segment; and
8. linear object accounting: two slot objects per predicted record, with no
   hidden pair matrix or Cartesian proposal materialization.

The frozen 1,000-episode gate passes all eight conditions. Exact and reference
reconstruction, extensional parity, slot-swap invariance, invalid-object
rejection, source-poison invariance, and linear accounting are all `100%`;
fieldwise lineage shuffle is `0%`, and overflow is zero. The canonical payload
digest is
`3cc986b32fcaf97be89d1df246da6ebed7e03a7a447e00ddf71c44626deb7ea1`;
the stored JSON SHA-256 is
`ad0c56a0dca0dc53f0cc514d3dba3c9b1d20a2d2c4204b15fb172ed9d6e4ea2f`.

## 4. Neural gate

The parent, tokenizer, source layer, source encoder, boundary weights, update
budget, batch size, and optimizer family remain fixed. Only the record summary,
two-slot decoder, and complete-object heads are trainable.

- 1,200 updates, batch 8, exactly 9,600 charged raw-source episodes;
- new disjoint train/evaluation seeds;
- 256 episodes each for train, lexical shift, renderer shift, and composition
  shift;
- one full seed first; four additional seeds only after the first seed clears
  every absolute floor; and
- report complete/trainable parameters, source words, wall time, peak memory,
  segment/slot counts, and fail-closed receipts.

Frozen first-seed pass conditions:

- learned segmentation exactness `>=99%` in every cohort;
- gold-support recall and exact sealed packets `>=95%` in every cohort;
- zero accepted duplicate/overlapping fields and zero false nominal merges;
- zero overflow on every cohort;
- post-seal source-poison invariance exactly 100%;
- at least 20 points exact collapse under fieldwise lineage shuffle; and
- at least 15 points exact advantage over the closed SC1 learned packet on
  every shifted cohort.

Five-seed promotion requires every seed to beat SC1 and at least four of five
to clear all absolute floors. Failure of seed one closes WRA1 without a
threshold, width, duration, seed, source-layer, or loss variant.

## 5. Claim boundary

Set prediction, slot attention, Hungarian matching, pointer networks, segmental
parsing, exact-one constraints, and source copying are established ideas. WRA1
is not a novelty claim by itself. It is a bounded attempt to supply DIVERGE's
potentially novel source-sealed factorized epistemic packet with a compiler
whose output cardinality and coherence are architectural invariants rather
than consequences of locally calibrated edges.

A pass authorizes one unchanged DIVERGE delayed-recovery evaluation in the
already-supported `>=8`-world regime. It does not reopen DIVERGE's failed broad
resource gate, authorize long pretraining, or establish general reasoning.

## 6. Frozen neural result

Newton job `742579` completed the one authorized full seed on `evc35` in
`00:06:13`. The trainable whole-record compiler has `1,411,405` parameters
inside a `137,030,233`-parameter frozen-parent system. It consumed exactly
`9,600` raw-source episodes in `336.189` training seconds (`28.555`
episodes/s) and peaked at `765,621,248` allocated GPU bytes. Optimization was
finite and source segmentation remained exactly `100%` throughout; final
minibatch loss was `2.1227` after starting at `39.5029`.

The autonomous result fails every semantic promotion floor:

| Cohort | Segmentation | Support recall | Exact packet | Fail closed |
|---|---:|---:|---:|---:|
| train | 100% | 2.344% | 0% | 69.531% |
| lexical shift | 100% | 0% | 0% | 97.266% |
| renderer shift | 100% | 0% | 0% | 98.438% |
| composition shift | 100% | 0% | 0% | 97.656% |

Overflow remains zero and source-poison invariance remains 100%, showing that
the hard decoder fails closed as designed. Those safety properties cannot
compensate for missing gold support. The first-pass support-loss condition is
fatal by the frozen contract; the remaining four seeds are not run and WRA1
receives no threshold, width, duration, source-layer, loss, or decoding
variant.

The immutable report/checkpoint SHA-256 values are
`4bfa0400815df77e00ec7f45c16dc7ca84b9f0dbe5181b4b3801a45d713d31c5` /
`38fbf931af0b1d0fc75c058948aed467593606877b035cf1a6e2679d8e3ef834`.
Runtime commit is `4f07bdf`; archive and manifest SHA-256 values are
`a75b7c73fb6fd4e90cd56fa1f406aa66b932640d69d727429404375ee2f6e2d3` /
`bb6a11878b367b3f722910cbb5eddfe94512b48fc7999235ece23c5a1f43f216`.

**Decision:** `close_wra1_and_do_not_promote_learned_diverge`.
