# DIVERGE-NTA2: Constrained Role Projection

Status: passed the one frozen zero-update gate on 2026-08-06.

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

## Result

NTA2 passes every condition without changing a weight:

- 963/963 learned operation classes remain exact;
- finite-state projection produces 963/963 exact legal role paths and packets;
- all 279 first-error selections, terminals, and complete trajectories are
  exact across all 963 transitions;
- every operation/depth slice is exact and no packet is invalid;
- trust-source and ignored-conflict controls score 0/279, initial packet swap
  scores 2/279, and operation shift scores 0/279.

Raw independent role argmax remains 0/963, preserving NTA1 as the control. The
result therefore belongs to the conjunction of learned operation recognition
and explicit field topology, not to a post-hoc model update.

Evaluation/gate SHA-256 values are
`0d9c0755bc66bf8b3279acebde627080745289534f13ab2b9b7d8e1bc6bea550` /
`39c3fb35e59c6b4383925dbf9183c0a270eeda3946ea9c7b2de34149c364e531`.
One full-document scanner gate is authorized; broader reasoning is not.
