# LAM1: Learned Arithmetic Microcode

Status: frozen prospective mechanics gate

Date: 2026-08-10

## Hypothesis

Shohin's strongest current development compiler converts raw source bytes into
an exact stack program, but prior learned ledgers tried to regenerate complete
intermediate values and collapsed. LAM1 tests a different architecture:
learn only the finite local transition laws of decimal arithmetic, then reuse
those learned laws recurrently across arbitrary digit positions and program
steps.

The candidate owns three trainable transition tensors:

- `ADD(a, b, carry) -> (digit, carry)`;
- `SUB(a, b, borrow) -> (digit, borrow)`; and
- `MUL(a, b, carry) -> (digit, carry)`.

All legal local states are supervised exactly once per optimization step. The
same learned tables execute signed arbitrary-precision integer operations,
unreduced rational addition/subtraction/multiplication/division, and a typed
postfix stack. Candidate execution manipulates only digit arrays, signs,
stack addresses, and argmax table outputs. Python integer/Fraction arithmetic
exists only in the independent assessor.

The tensors contain exactly 108,000 trainable logits. There is no answer
memorization, pretrained language model, verifier, solver call, source family
label, or whole-expression target in microcode training.

## Frozen mechanics gate

Train from all-zero logits for exactly 32 full-table SGD updates, learning
rate 1.0, seed `2026081041`. Evaluate immutable BTT1 programs from the exact
75,935-row train and 3,917-row source-disjoint development splits.

All conditions are conjunctive:

1. all 1,400 legal local transitions are exact;
2. normal terminal rational state is exact on 75,935/75,935 train programs;
3. normal terminal rational state is exact on 3,917/3,917 development programs;
4. normal execution has zero invalid states and zero 256-digit overflow;
5. resetting carry/borrow at every recurrent digit loses at least 20 points on
   multi-digit development programs; and
6. the fixed opcode permutation loses at least 50 points overall.

Failure closes exact LAM1 without table width, optimizer, duration, seed,
digit-base, or overflow variants. A pass opens one composition with the frozen
BTT1 checkpoint and unchanged width-64 WGP development projection. It does
not reopen WGP1 confirmation and does not authorize a holdout claim.

## Claim boundary

A mechanics pass establishes that learned finite micro-operations can compose
exactly over much longer arithmetic trajectories than those seen by any one
transition parameter. It does not establish raw-language generalization,
novel arithmetic, broad reasoning, or holdout capability. Only the subsequent
frozen compiler composition can establish a complete development system.
