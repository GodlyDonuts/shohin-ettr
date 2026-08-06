# DIVERGE-FTA1: Length-Equivariant Finite-State Source Transducer

Status: implementation and local QA complete; frozen before any H100 neural
result on 2026-08-06.

## 1. Capability hypothesis

ATS1's algebraic packet executes every valid compiled trajectory exactly, but
its absolute-position Transformer compiler loses semantic side identity when
numeric and symbolic spans become longer than training. The copied surfaces
remain largely correct; the model swaps LHS/RHS or argument roles.

FTA1 changes only the source computation:

> A position-free bidirectional finite-state transducer, tied across source
> length, can assign byte roles relative to delimiters and local syntax rather
> than memorized absolute offsets, preserving exact spans under length shift.

This is not another ATS1 width, seed, duration, or loss variant. It removes
absolute position parameters and replaces global Transformer attention with a
two-layer bidirectional GRU whose recurrent transition is shared at every byte.
ATS1 is the protected absolute-position control.

## 2. Frozen mechanism and budget

- unchanged source bytes, role vocabulary, operation vocabulary, boards,
  typed CRT/symbol packet, transaction layer, evaluator, and controls;
- width 192, two bidirectional recurrent layers, 96 hidden units per direction;
- no absolute or relative position embeddings;
- active lengths are packed, so padding never enters either direction;
- operation class is read from the active recurrent states; roles are decoded
  per byte;
- one seed `2026080607`, 1,600 updates, batch 512, AdamW `3e-4` cosine;
- no pretrained language-model parameters.

## 3. Frozen gate

The exact ATS1 component thresholds are reused without change:

- operation and copied arguments >=99%;
- copied LHS state >=95%;
- 432/480 exact OOD terminals;
- 136/160 terminals and 128/160 complete trajectories in every family;
- zero invalid packets and RHS poison invariance;
- >=240-answer drops under packet swap and operation shift.

Parameter count, source bytes, H100 time, peak memory, and every ATS1 score are
reported beside FTA1. A pass authorizes one autonomous CRP1 composition gate.
A failure closes this local rendered-step compiler without recurrent width,
direction, layer, seed, duration, optimizer, role, or loss variants.
