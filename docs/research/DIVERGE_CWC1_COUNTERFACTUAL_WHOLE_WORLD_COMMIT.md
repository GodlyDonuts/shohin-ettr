# DIVERGE-CWC1: Counterfactual Whole-World Commit

Status: frozen before data materialization or neural scoring.

## Failure boundary

EWC1 is exact on normal WORLD compilation and exact under register, alias, and
entity actions, but retains 27.832% joint accuracy after language context is
deleted. Its operation stream remains 84.888% exact. A single executable
program therefore leaves enough punctuation and position geometry for a
nonsemantic shortcut.

CWC1 changes the problem topology rather than retrying EWC1. Every source
contains two complete executable candidate worlds. Their order, initial
states, programs, aliases, and opaque labels are independently randomized.
Only a natural directive identifies which complete world is valid. The model
must commit one coherent candidate; fields from incompatible candidates are
never averaged.

## Hypothesis

A shared byte encoder can learn source-dependent whole-world selection when
position supplies no stable answer and an exact counterfactual role action is
part of the trainable forward path. If the directive is removed, selection
must return to chance even though both candidate programs remain visible.

For raw candidate scores `r(x)` and the source intervention `g` that swaps
only the two candidate labels inside the directive, the treatment uses

```text
e(x) = 0.5 * (r(x) + flip(r(gx)))
```

The complete candidate blocks are unchanged by `g`. Applying the partner
view a second time must flip the committed candidate exactly. This is an
identity projection over whole candidate lineages, not fieldwise latent
mixing.

## Frozen model and data

- 470,785 trainable parameters;
- two-layer, width-192 bidirectional byte GRU;
- one shared candidate block projection, label projection, global projection,
  and scalar score head;
- maximum 1,536 source bytes;
- 50,000 training rows, 4,096 development rows, and 4,096 confirmation rows;
- depths 3--20 with eight opaque aliases, two opaque registers, and two
  opaque candidate labels per row;
- zero source, identity, alias, register, or candidate-label overlap between
  splits;
- exact 50/50 target position in every split and target imbalance at most one
  row inside every renderer composition;
- eight positive and eight negative semantic directive renderers;
- renderer *compositions* partitioned into train, development, and
  confirmation by `(2 * positive + 3 * negative) mod 5`;
- seeds `2026080731`, `2026080732`, and `2026080733`.

All three files are generated and audited before model scoring. Confirmation
remains inaccessible unless the development report passes every frozen gate.

## Matched arms

All arms use the same initialization seed, parameters, rows, sample sequence,
1,000 updates, batch 128, AdamW, cosine schedule from `3e-3`, clipping at 1.0,
and exactly two encoder forwards per update.

1. **Involution treatment:** normal plus counterfactual partner projected into
   one equivariant whole-world decision.
2. **Duplicate-forward control:** the normal source is evaluated twice.
3. **Ordinary augmentation control:** normal and counterfactual sources receive
   separate cross-entropy losses with mapped labels.

Ordinary augmentation is the strongest standard alternative. If it matches
the treatment, CWC1 may qualify a practical semantic selector but cannot
claim a score advantage for the projection.

## Development interventions

- normal unseen renderer compositions;
- mapped directive counterfactual;
- complete candidate-block order swap;
- consistent unseen renaming of labels, aliases, and registers;
- deletion of the complete directive while retaining both candidate worlds;
- exact projection algebra;
- matched parameter, initialization, data, update, batch, learning-rate,
  source-byte, and forward-count receipts.

## Conjunctive gate

Before confirmation can open, the involution arm must satisfy all conditions:

- at least 99% training fit;
- at least 99% normal development exactness;
- at least 99% mapped-counterfactual exactness;
- at least 95% on every held-out renderer composition;
- at least 99% entity-rename exactness;
- at least 99% complete-block-swap exactness;
- directive-deletion accuracy between 49% and 51%;
- directive-deletion mean signed margin at most `1e-6` in magnitude;
- exact zero projection residual;
- complete matched receipts across all three arms;
- both controls fit at least 99%;
- treatment normal accuracy no more than one point below the strongest
  standard control.

Confirmation applies the same treatment-only semantic conditions on the
third renderer-composition partition. A miss closes CWC1 without width,
duration, seed, threshold, phrase, loss, or renderer variants.

## Claim boundary and successor

A confirmed pass qualifies a learned, source-dependent, whole-world selector
on this controlled natural interface. It does not establish unrestricted
language grounding or end-to-end reasoning. The one authorized successor is
to select one complete candidate WORLD, run the frozen EWC structural
extractor only inside that selected candidate, and feed the resulting typed
WORLD into unchanged confirmed NPL2. EWC remains a structural extractor, not
a semantic owner.

A failure closes this selector design and requires a structurally different
WORLD mechanism. No continuation pretraining is authorized by CWC1 alone.
