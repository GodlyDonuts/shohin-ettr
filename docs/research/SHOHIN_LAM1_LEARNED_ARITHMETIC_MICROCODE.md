# LAM1: Learned Arithmetic Microcode

Status: mechanics and frozen development composition passed

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

## Frozen composition gate

The only authorized composition uses BTT checkpoint SHA-256
`2283c86b1a640c9d5c02ffbc70b4646a0a1ad3538c4e1d4ab4ea3b164363e278`,
LAM checkpoint SHA-256
`baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4`,
the immutable 3,917-row BTT development split SHA-256
`de16d5dc8dd3676f4c3f4a69c306ec19a0ea8a42e9750bf93a2aaf16f4be0bf8`,
and width-64 weighted grammar projection. No checkpoint, projection, source,
or learned transition is changed.

All conditions are conjunctive:

1. normal source-to-terminal exactness is `3917/3917`;
2. normal compilation and execution have zero invalid states and projection
   has zero exhaustion;
3. same-family/depth source shuffle and zero-byte input each score at most
   25%;
4. resetting carry/borrow loses at least 20 absolute points; and
5. fixed opcode permutation loses at least 50 absolute points.

This is a single development evaluation. Failure closes this exact
composition without threshold, beam, seed, source, table, or checkpoint
variants. Passing qualifies a complete development architecture, but it does
not create held-out evidence or reopen the closed WGP1 confirmation.

## Mechanics result

CPU job `749769` passed every frozen mechanics condition. Learned transitions
were `1400/1400`; normal execution was `75935/75935` on train and `3917/3917`
on development with zero invalid states. Carry reset scored `367/3917 =
9.3694%`; opcode permutation scored `5/3917 = 0.12765%`. Scientific elapsed
time was 55.42 seconds. Report SHA-256 is
`b9a82f97065dce5750a02660a80fa71d4af2a4f14c45a39a5f35c5e0135e63c8`.

## Composition result

H100 job `749784` completed the only authorized composition in 192.24
scientific seconds. The normal path was exact on `3917/3917` programs with
zero compiler invalidity, execution invalidity, or grammar-search exhaustion.
The controls scored:

- source shuffled: `7/3917 = 0.1787%`;
- zero bytes: `14/3917 = 0.3574%`;
- carry reset: `367/3917 = 9.3694%`; and
- opcode permutation: `5/3917 = 0.12765%`.

Every frozen gate passed. The atomic result is
`docs/research/SHOHIN_LAM1_COMPOSITION_RESULT.json`, SHA-256
`60698d0b00bc0fc82d38c54173a956e5e3b74d191960e4f1b1de66c7a549e4cf`.
This qualifies the complete raw-byte compiler, constrained program projection,
and learned arithmetic executor on source-disjoint development data. It is not
a held-out confirmation because WGP1 confirmation closed at source admission.

## Claim boundary

A mechanics pass establishes that learned finite micro-operations can compose
exactly over much longer arithmetic trajectories than those seen by any one
transition parameter. It does not establish raw-language generalization,
novel arithmetic, broad reasoning, or holdout capability. Only the subsequent
frozen compiler composition can establish a complete development system.
