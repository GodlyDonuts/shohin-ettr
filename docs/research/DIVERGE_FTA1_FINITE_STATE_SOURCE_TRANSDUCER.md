# DIVERGE-FTA1: Length-Equivariant Finite-State Source Transducer

Status: passed the one frozen component gate on 2026-08-06.

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

## 4. Result

Immutable capsule `diverge_fta1_8eeb136_r1` has archive SHA-256
`3f415cd7b104567b98b82eebd3a8078a6fb2849434c2a49bbab3adbe997e35ad`.
Smoke `743581` completed both BF16/CUDA updates. Scientific job `743585`
completed exactly 1,600 updates in 97.040 seconds with 400,724 trainable
parameters, 32,262,081 source bytes, 332,461 source bytes/s, and 357,808,640
peak allocated CUDA bytes. The 96-row development probe was completely exact.

Untouched evaluator `743590` then passed every frozen held-length condition:

| Metric | FTA1 | ATS1 control |
|---|---:|---:|
| valid/compiler-exact segments | 3,854/3,854 | 3,093/3,854 |
| exact terminals | **480/480** | 272/480 |
| exact complete trajectories | **480/480** | 272/480 |
| scalar/register/symbolic terminals | **160/160/160** | 100/128/44 |
| initial-packet swap terminals | 3/480 | 2/480 |
| operation-shift terminals | 1/480 | 1/480 |

Removing absolute positions and tying the recurrent source update therefore
repairs the exact observed length-transfer failure with fewer parameters than
ATS1. This is causal evidence for delimiter-relative finite-state compilation,
not an open-domain reasoning result.

Training report/checkpoint SHA-256 values are
`06536c2f63e7eed1c464217d1314ef751510b398455b4cb0caacbf7bd99f8929` /
`9321b78372d9926930d4de073d70e82c94e8360a69e09be695bab91b2e479f2d`.
Evaluation/gate SHA-256 values are
`7b7f3a6830a9f777464325321b9be1dde271ff602de012483ec77a77d931678c` /
`dd5f0d261fdf88f18bbcf7aee6574d46f5d48a5d31e438bfcf0cc8181f3dfa72`.
The pass authorizes exactly one autonomous contradiction-guided composition
gate on the unchanged CRP1 OOD board.
