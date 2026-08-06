# DIVERGE-NTA2: Constrained Role Projection

Status: frozen before model result on 2026-08-06.

## Hypothesis

NTA1 transfers operation semantics perfectly but independent byte-role argmax
violates the known transaction topology. NTA2 makes one zero-update change:
project source bytes through a finite-state signed-number grammar with exactly
three ordered fields (`LHS`, `argument`, `RHS`), forcing CLS and separators to
`OTHER`. The unchanged learned FTA1 head still selects the operation.

This is a structural decoder, not a new model fit. It is intentionally narrow:
the same 279 rows, 963 transactions, checkpoint, operations, algebra, and
controls are reused. NTA1 remains the raw-argmax negative.

## Frozen gate

- operation, projected roles, and valid packets each >=95%;
- >=250/279 exact first-error selections, terminals, and trajectories;
- zero invalid rows;
- every error operation and depth reaches >=80%;
- trust-source and ignored-conflict drops >=200;
- initial packet swap and operation shift drops >=150.

A pass promotes the projection only to one context-rich natural trace gate
where extra numeric spans make field choice nontrivial. It is not a broad
reasoning claim. A failure closes this zero-update projection before supervised
adaptation.
