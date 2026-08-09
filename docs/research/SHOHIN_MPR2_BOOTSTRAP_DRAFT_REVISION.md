# MPR2: Bootstrap Draft-Owner Revision

Status: frozen prospective development gate, 2026-08-09. Holdout and larger-MoE
transfer remain sealed.

## Hypothesis

MPR1 showed that a simple all-layer post-MoE residual improves pinned OLMoE,
but its original weak draft is causally harmful: aligned scored 233/1,289 while
both shuffled and draft-hidden controls scored 247. MPR2 changes only the
quality of the model-owned temporal input. The successful MPR1 source-only
model becomes a first-pass **draft owner**; a fresh, identically sized residual
then learns to revise that owner's exact output.

This is not a rank, layer, seed, duration, prompt, or threshold retry. It tests
whether temporal revision becomes useful when the preceding model-owned state
is stronger.

## Frozen system

- host: `allenai/OLMoE-1B-7B-0125-Instruct` at
  `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
- draft owner: exact MPR1 hidden/source-only checkpoint at update 256, SHA-256
  `6cbe9fdcb309513f8ff17e4aa9bc18549db2ccf74681e608cffac0213584610d`;
- owner geometry: all 16 sparse layers, rank-18 shared post-MoE residual,
  1,179,648 trainables;
- owner generation: greedy, no thinking, at most 768 new tokens, source plus
  the original revision envelope with the old draft causally unavailable to
  the full model;
- reviser geometry: the same all-16-layer rank-18 residual, initialized fresh;
- fit: 256 AdamW updates, LR `2e-5`, batch 1, accumulation 8, seed
  `2026080901`, data seed `2026080814`, and context 4,096.

One canonical owner draft is generated for every unique training source. The
existing development owner candidates are reused only after exact checkpoint,
data, runtime, and candidate hashes are verified; they scored 247/1,289 with
zero 768-token exhaustion.

## Matched arms

Every admitted training presentation has the same source, verified target,
update budget, target-token charge, and model geometry.

1. **Aligned:** the exact trained-owner draft for that source.
2. **Shuffled:** another source's same-task nearest-token-length trained-owner
   draft.
3. **Hidden/source-only:** the aligned token IDs and positions, but every draft
   token key is causally unavailable throughout the full model.
4. **Draft-owner direct:** the immutable first-pass owner output itself; this
   is the practical unchanged temporal baseline.

All train and development rows retain complete source, complete draft, and
complete target at 4,096. The holdout is neither read nor generated.

## Frozen development gate

On the exact 1,289-row source-disjoint development board, every condition is
required:

1. aligned revision `>=286`, exactly 39 answers (+3.0 points) over the trained
   draft owner at 247;
2. aligned revision at least 13 answers (+1.0 point) above shuffled;
3. aligned revision at least 13 answers above hidden/source-only;
4. domains do not regress below the draft owner: MATH `57`, logic/science
   `182`, executable code `8`;
5. conservative possible-semantic repairs minus strict breaks relative to the
   draft owner are at least 13;
6. exact protected hashes, zero frozen router/expert trainables, exact
   parameter/update/token/FLOP receipts, and complete 4,096-token custody.

A conjunctive development pass opens exactly one sealed holdout with the same
margins and nonnegative domain deltas. Only a development-plus-holdout pass
authorizes staged larger-MoE transfer. Any development miss closes exact MPR2
without geometry, duration, seed, renderer, prompt, threshold, or decoding
variants.

## Interpretation

- **Pass:** meaningful causal temporal revision is operational on small
  OLMoE; proceed to one sealed holdout and then staged larger-MoE transfer.
- **Aligned improves but does not beat controls:** stronger generic supervised
  adaptation remains useful, but temporal draft use is not established.
- **Failure:** close this bootstrap-owner route and do not scale it.

