# DIVERGE-TOL1: Typed Operation Language Gate

Status: frozen before board generation or neural result on 2026-08-05.

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
