# DIVERGE-NTA1: Natural Arithmetic Transfer

Status: closed after the one frozen zero-shot gate on 2026-08-06.

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

## Result

The builder emits exactly 279 rows / 963 transactions at board SHA-256
`d71f1875ca967e2ff84cf0ff9e9940794643685c42a6298af90151b1a958d5b3`.
Depth counts are 78/60/78/63 for depths 2/3/4/5, and every wrong terminal
differs from the verified answer.

Zero-shot operation transfer is perfect: 963/963 operation classes. Unconstrained
role argmax nevertheless yields 0/963 valid packets and therefore zero
autonomous selections or answers. Token audit makes the boundary precise:

- the source-free CLS position is predicted as `LHS_A` on all 963 segments;
- all 2,748 LHS bytes and all 2,939 RHS bytes are otherwise exact;
- 2,265/2,354 argument bytes are exact;
- the remaining 86 errors assign leading digits of long subtraction arguments
  to `RHS_A`, violating contiguous field structure.

Thus semantic operation recognition transfers, but independent token argmax
does not preserve a valid finite-state field sequence after the renderer shift.
Evaluation/gate SHA-256 values are
`c0c270778cb643316776e84b3512d7ef738d93272a20b9f57f4e8d4b9e4e4750` /
`ea86c3bde7a4c83b8333aafec78a5a6285c493501273d6b1db9750a5d0fb4e09`.

The ordered successor is one zero-update constrained finite-state decoder that
forces CLS/punctuation to `OTHER` and finds the highest-scoring legal
`LHS -> argument -> RHS` field path. NTA1 is not rescored.
