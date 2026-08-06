# DIVERGE-NTA1: Natural Arithmetic Transfer

Status: frozen before board construction or model result on 2026-08-06.

## Question

FTA1-AC1 is exact on a generated typed grammar. NTA1 asks the first external
transfer question: does the unchanged finite-state compiler recognize and
falsify arithmetic transactions copied from an independently answer-verified
reasoning corpus when the synthetic `Step N:` prefix is absent?

## Board

The only source is the frozen V10 verified corpus at SHA-256
`2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`.
The deterministic builder admits exactly 279 `reasoning_gym_trace` rows with
two to five chained integer `+`, `-`, or `*` transactions, nonnegative explicit
arguments, exact intermediate arithmetic, exact final answer, and CRT-safe
values. It preserves each original equation substring, injects one deterministic
wrong result, and recomputes the suffix from that corrupted state. The FTA1
checkpoint receives no update.

This is corpus-derived arithmetic, not broad natural language. It tests a real
renderer/source shift while keeping the transaction algebra fixed.

## Frozen gate

- operation >=90%, whole-role and valid-packet rates >=80%;
- >=200/279 exact first-error selections, terminals, and trajectories;
- <=28 invalid rows;
- every error operation and depth 2--5 reaches >=60% terminals;
- trusting source or ignoring the conflict costs >=150 answers;
- initial packet swap and operation shift each cost >=120 answers.

The gate is zero-shot and one-pass. A pass authorizes one source-disjoint
supervised natural-compiler gate; it does not authorize a general reasoning
claim. A failure is localized before any natural-interface training.
