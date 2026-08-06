# DIVERGE-ATS1: Source-Sealed Algebraic Typed State

Status: frozen before implementation or neural results on 2026-08-06.

## 1. Decision boundary

RSM1 selected the gold CRP1 boundary on all 480 OOD rows but decoded zero
exact initial states. Its flat 24-byte state then produced only 2/2,662 exact
transitions. The failure is therefore not evidence against recurrent
execution from a valid state; it rejects regeneration of an exact state from
six continuous packet vectors into an untyped character tape.

ATS1 changes both parts of that interface without changing CRP1 localization:

> Preserve source characters through a learned role-copy compiler, seal them
> into a typed algebraic packet, and compose transactions in that packet rather
> than regenerating intermediate text.

ATS1 is a bounded internal-machine experiment. Its residue arithmetic and
symbol-tape kernels are architectural primitives, analogous to adding an ALU
inside the model. It does not establish unrestricted mathematical reasoning,
language understanding, or novelty.

## 2. Source boundary

For a selected CRP1 candidate `e > 0`, the compiler receives the unchanged
bytes of draft step `e` and every later draft step. It does not receive the
gold state, program object, answer, correct trace, or claimed semantic fields.
It predicts one role for every source byte and one complete operation class per
step. Selected bytes remain in source order and become the only values written
to the packet. The claimed successor on each draft step is compiled for audit
but is never used during replay.

The role vocabulary is fixed before training:

```
OTHER, LHS_A, LHS_B, LHS_SYMBOL,
RHS_A, RHS_B, RHS_SYMBOL, ARG1, ARG2
```

The operation vocabulary is the union of the three frozen CRP1 families:
scalar add/subtract/multiply, five two-register transactions, and symbolic
reverse/rotate/swap. No operation ID or argument is supplied at candidate
inference. A malformed, noncontiguous, duplicated, or type-incompatible copy
fails closed.

After compilation, source bytes, source hidden states, and claimed successors
are deleted. Replay receives only the sealed typed state, operation IDs, and
copied arguments.

## 3. Algebraic packet

Numeric values are represented by residues modulo `(17, 19, 23, 29, 31)`.
Their product is 6,678,671, giving an unambiguous signed interval much larger
than the complete frozen board. Decimal source bytes are folded directly into
each residue; no vocabulary-sized numeric class or autoregressive digit
decoder exists.

Scalar and register transactions apply componentwise modular arithmetic.
Symbolic values use a hard source-copied character tape plus length, and
transactions are whole-tape permutations. One tied transaction layer is used
at every replay depth. Intermediate states remain typed packets and are never
converted to language.

The independent assessor reconstructs signed integers by CRT only for scoring.
CRT reconstruction, correct program objects, and correct successors are absent
from the candidate runtime.

## 4. Frozen data and budget

- CRP1 train/development/evaluation boards and hashes are unchanged.
- Train depth is 4--6; OOD depth is 7--9.
- Both correct and wrong train-step renderings may teach source roles, but only
  wrong OOD traces enter the forced replay gate.
- One seed: `2026080606`.
- Byte width 128, two Transformer encoder layers, four heads, no pretrained
  language-model weights.
- At most 1,600 optimizer updates and 512 source steps per update.
- AdamW, peak learning rate `3e-4`, fixed cosine schedule.
- No held renderer, OOD row, answer, or threshold enters optimization.

## 5. Frozen component gate

The CPU reference must first show 100% extensional parity on every state and
transition in all 5,760 CRP1 identities.

Forced gold-boundary OOD promotion is conjunctive:

- complete operation class exact >=99%;
- complete copied LHS state exact >=95%;
- copied operation arguments exact >=99%;
- exact terminal typed state >=432/480;
- exact terminal typed state >=136/160 in every family;
- exact complete typed trajectory >=128/160 in every family;
- zero accepted malformed packets;
- claimed RHS deletion leaves every replay result unchanged;
- initial-packet swap and within-family operation shift each reduce terminal
  exactness by at least 240/480; and
- all source-role parameters are finite and the frozen CRP1 checkpoint is not
  modified.

If the component gate fails, close ATS1 without a seed, width, depth, role,
modulus, optimizer, duration, or loss repair. Do not launch autonomous CRP1
composition.

## 6. Ordered autonomous gate

Only a component pass authorizes one autonomous composition with the immutable
guarded CRP1 selector. On wrong traces it must beat the preserved guarded
language path (`213/480`) by at least 27 exact answers and recover at least 80%
of rows whose selected boundary is no later than the true first error. On
correct twins it must preserve at least 432/480. Swapping sealed typed packets
must cause the predicted large collapse.

A pass would establish a bounded source-sealed model-owned transaction
mechanism. It would not by itself establish broad reasoning or justify long
pretraining.

## 7. Pre-neural qualification

The independent CPU reference passes exact extensional parity before any
neural result. Across 4,800 train, 480 development, and 480 evaluation
identities it executes 24,044 / 2,368 / 3,854 transitions and reaches every
terminal state exactly. The complete trajectory digest is
`4b7e06df1f7cd9237e7309fb74e633d9fac2c7f60583692c140f5d8ab3ca5eeb`.
The audit report SHA-256 is
`eaadb1793c3a4e40d5c7e81ec4d3f4fce75bde19b727230d67284faaaf8b1314`.

Local source-role, runtime, gate, Python compilation, Ruff, Bash syntax, and
two-update CPU smoke checks pass. No H100 capability result exists yet.
