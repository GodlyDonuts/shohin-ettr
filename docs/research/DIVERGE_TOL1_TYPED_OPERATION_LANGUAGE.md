# DIVERGE-TOL1: Typed Operation Language Gate

Status: completed and failed the one frozen OOD gate on 2026-08-05.

## Capability hypothesis

The working FTA1/NTA3 subsystem recognizes closed arithmetic transactions but
does not compile variable binding, exact rational values, predicates, or
stateful programs. TOL1 tests whether one length-equivariant finite-state
source encoder plus a parameter-free typed assignment layer can compile a
broader operation language and execute held-out programs exactly.

The target machine stores exact rational registers and supports:

- `SET`, `ADD`, `SUBTRACT`, and `MULTIPLY` with literal or register operands;
- `SWAP` between two registers;
- six comparisons and one guarded true/false update; and
- a late `QUERY` of one register.

The compiler predicts a clause opcode, predicate and branch opcodes, and role
scores over source spans. A maximum-weight assignment may enforce only the
declared type and arity schema. It may not inspect answers, execute candidate
programs, or use generator metadata. Register identity is the quotient of
exact repeated source names. All execution uses `fractions.Fraction`.

## Frozen splits

- Training: 24,000 generated programs, depths 4--8, training renderer and
  register-name banks.
- Development: 512 fresh programs from the training support.
- OOD: 1,024 programs, depths 9--14, disjoint register names, held-out syntax
  recombinations, and reserved operation adjacencies.
- Every split is generated independently from a fixed seed and written with a
  content hash. The assessor re-executes serialized typed programs and rejects
  malformed or non-finite records.

## Frozen comparison

The treatment uses maximum-weight typed role assignment. The protected control
uses the same checkpoint and logits but independent role argmax with no legal
assignment repair. This isolates the structured decoder without changing
parameters or training FLOPs. Operation-shift, binding-derangement, state-reset,
and query-only interventions test causal dependence on the compiled program.

## Pass / kill gate

On the OOD split, all of the following are required:

- at least 97% exact clause opcodes and 95% exact typed instructions;
- at least 80% exact complete programs and 85% exact terminal answers;
- at least 80% answer accuracy on programs containing each of rational
  operands, register operands, swaps, and guarded branches;
- treatment exceeds unconstrained decoding by at least 20 answer points;
- operation shift, binding derangement, and state reset each lose at least 50
  answer points, and query-only remains below 20%; and
- zero executor acceptance of malformed packets.

A failure closes this finite-state compiler at the failed boundary. It does
not authorize width, seed, duration, or renderer tuning. A pass authorizes one
full-document integration gate and later replacement of exact-name quotienting
with learned alias binding. Neither outcome is a claim of general reasoning.

## Result

The board contains 24,000 / 512 / 1,024 train, development, and OOD
programs, or 263,972 / 5,633 / 16,888 clauses. Train, development, and OOD
SHA-256 values are
`d8b4af0744d3c4232c1de91989a7b6fd4dd3168f45e35c98f9495add1b52b8ba`,
`790e40984e042f3983d00724089370b4caa2063cab0d6811ab215d80b854a108`,
and `f8f94746c4f5e1cef06204fc39bd1cfe4630ab81b1a15f0f6c6624a309189f70`.

H100 job `743668` completed 2,000 updates over 30,896,564 sampled source
bytes in 96.879 seconds. The compiler has 515,362 trainable parameters and
used 621,024,768 peak allocated GPU bytes. It reaches 512/512 exact
development programs and answers. Checkpoint SHA-256 is
`6de732edf7158bfacc4a1627b7f2b8fbcb4274482b91fef8c524ee114af7e1b0`.

The unopened OOD evaluation `743681` fails:

- clause opcode: 16,609/16,888 = 98.348%;
- exact typed instruction: 11,565/16,888 = 68.481%;
- exact complete program: 0/1,024;
- exact answer: 172/1,024 = 16.797%; and
- raw independent-role answer: 0/1,024.

Evaluation and gate SHA-256 values are
`d6ab92ca5400d0bc26d867e1a6d95f24b3d6a4985e88254f053567953e7d6c98`
and `a3f5e4cb868e6bb709d9e1350fdb70d12a5be5c63793c77424fff5ff567b3517`.

## Failure localization

The operation concept largely transfers; binding does not. Exact instruction
rates by opcode are SET 5,502/5,502, ADD 1,247/1,402, SUBTRACT 1,367/1,478,
MULTIPLY 2,349/2,515, QUERY 702/1,024, GUARD 403/2,506, and ordered SWAP
0/2,461. Nearly every correctly classified OOD swap reverses its two operands,
which is execution-equivalent but exposes an over-strict ordered packet hash.
It does not explain the low answer rate. The decisive errors are guards:
clause-global pooling selects ordinary words as registers, confuses the moved
true branch with the predicate, and reaches only 909/2,506 exact comparator
plus true/false action tuples.

TOL1 is closed. No width, seed, duration, or renderer repair is authorized.
The justified successor is a document-level anchor-relational compiler that
builds one declared-register quotient, restricts all register pointers to that
table, predicts action/comparison anchors locally, decodes guard regions
independently, and canonicalizes symmetric bytecode. It requires a fresh
confirmation renderer after the opened TOL1 OOD board is used for interface
development.
