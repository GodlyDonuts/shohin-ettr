# DIVERGE-EIC1: Equivariant Identity Commit

Status: frozen after CGL1 and its sole orbit attribution closed, before EIC1
training or model result.

## Capability hypothesis

CGL1 proves that terminal-outcome labels causally control the fitted decision,
but a standard candidate-entailment scorer can still represent a different
role convention for each renderer. Applying an orbit product only after CGL1
training recovers SmolLM2 (`768/768`) but not Shohin (`512/768`). The missing
operation is therefore not merely another inference sample. The identity
equivariance must be the trainable model's forward computation.

EIC1 projects every two-candidate score into the sign representation of the
candidate-swap group before the loss or commit can observe it. For raw
candidate scores `r(x)` and the exact mention transposition `g`, EIC1 uses

```text
e(x) = 0.5 * (r(x) + g^-1 r(gx)).
```

The same shared LoRA owner produces both terms. The model can fit only evidence
that survives this projection in the correct physical-identity frame. It
cannot commit one renderer-local convention on `x` and a different convention
on `gx`. This is a changed computation boundary, not a CGL1 seed, width,
duration, prompt, threshold, or post-hoc loss retry.

## Matched arms

Run four independent arms with the immutable CGL1 public/supervisor bytes,
order, seed, optimizer, one-pair epoch, LoRA geometry, parent, and evaluator:

1. Shohin EIC1 involution projection.
2. Shohin duplicate-forward control, which evaluates the normal view twice.
3. SmolLM2 EIC1 involution projection.
4. SmolLM2 duplicate-forward control.

The duplicate arm matches two backbone forwards and all trainable parameters
without receiving the transposed view or candidate mapping. Frozen CGL1 is an
additional historical control. The SmolLM2 treatment is a capacity ceiling;
Shohin is the decisive protected-backbone arm.

## Development gate

The already-open CGL1 development board rejects EIC1 unless the Shohin
treatment satisfies every condition:

- normal assignment at least `765/768`;
- every mode at least `254/256` and every renderer at least `127/128`;
- mapped mention-swap assignment at least `765/768`;
- context deletion loses at least 250 exact assignments;
- entity-renamed prompts, scores, and predictions are bit-identical;
- frozen-parent tensors are bit-identical; and
- mapped-swap accuracy exceeds the equal-FLOP duplicate control by at least
  200 assignments without losing more than three normal assignments.

The evaluator reports the exact projection identity residual. A development
miss closes EIC1 without seed, width, duration, layer, loss, prompt, renderer,
or optimizer variants. A Shohin pass admits exactly one independently built,
source-disjoint EIC1 confirmation board. SmolLM2-only success records a
capacity floor and does not promote Shohin.

The confirmation generator is frozen before training at seed `2026080713`.
It builds 256 new exact programs from a 32-symbol opaque entity bank, six new
query families with both clause orders, 768 balanced transactions, and
1,048,576 represented worlds. It must report zero source, query, identity, and
entity overlap against the CGL1 training corpus, the opened development board,
the prior PQI board, and the still-unopened CGL1 confirmation board. Its bytes
may be staged read-only, but no EIC1 model may access them unless the Shohin
development treatment and its matched-control comparison both pass.

## Claim boundary

EIC1 is an exact typed identity commit mechanism, not open-domain reasoning.
It may qualify a semantic transaction owner for later DIVERGE execution and
verified branch-local plasticity. It does not authorize continuation
pretraining, access the unopened CGL1 confirmation board, or rescue CGL1.
