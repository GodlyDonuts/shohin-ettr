# R12 Atomic Typed-Edit Algebra Preregistration

## Status

Preregistered before inspecting the sealed sparse-residual result. That result
failed both required state-isolated axes. The atomic architecture is now
implemented and locally qualified before its first H100 launch. Fifty-five
focused and related tests pass with clean Ruff, byte compilation, Bash syntax,
and diff checks. The trainer fails before optimization unless canonical edits
exactly reconstruct the first real training batch's target terminal packet.

## Diagnosis

Copy-biased residual transport separates preservation from rewriting, but its
value, type, active, root, relation, and status gates can still make mutually
inconsistent local decisions. A causal intervention is not a bag of changed
fields. It is a small typed edit whose opcode, address, payload, and graph
effects must agree. Independent gates can reduce aggregate Brier loss while
never producing the exact intervention-specific state consumed by the reader.

## Architecture

Compile COMMAND plus the initial typed state into one parallel set of atomic
edits, then apply those edits with a fixed differentiable state algebra. The
neural module predicts:

- one categorical node action per slot: `KEEP`, `ALLOCATE`, `WRITE`, or
  `CLEAR`;
- value and type payloads, consumed only by actions that legally use them;
- one categorical relation action per typed edge: `KEEP`, `LINK`, or
  `UNLINK`;
- one categorical root action: `KEEP`, `CLEAR`, or `SET(slot)`;
- one categorical disposition action: `KEEP`, `COMMIT`, `HALT`, or `REJECT`.

The executor applies all non-conflicting edits atomically. `KEEP` is exact
identity, `ALLOCATE` writes active/type/value together, `WRITE` changes only a
legal active value, `CLEAR` removes the node and every incident relation, and
relation actions are masked by post-node active endpoints. Root and status are
single categorical decisions. This removes serial exposure while preserving
the transaction algebra that independent field gates discarded.

Hard inference uses deterministic argmax actions and emits one valid terminal
packet. Soft training uses the same algebra with bounded straight-through
categoricals. The runtime receives autonomous initial state and COMMAND bytes
only. It receives no QUERY, answer, target, oracle program, host solver,
candidate selector, or best-of-K evaluator.

## Objective

Retain the full-state Brier anchor and complete 2x2 WORLD/COMMAND terminal-
delta loss. Add action supervision derived deterministically from the initial
and target terminal packets; this is a canonical state difference, not an
oracle transaction trace. Ambiguous edits are rejected during materialization
or assigned a deterministic minimal form before the immutable train split is
sealed. Loss is balanced by action class and intervention axis.

The first gate uses the same architecture seed 31, data seed 11, 1,000
updates, `3e-4` learning rate, clip `1.0`, 32 evaluation batches, protected
checkpoint, source-deleted reader, and held-out ordering as the sparse-
residual arm. This changes the edit representation and executor, not the data
population or evaluation.

## Promotion And Rejection

Promotion requires nonzero strict WORLD and strict COMMAND under the fully
autonomous path and corroborating movement under oracle-program/autonomous-
state isolation, without material factual collapse. It must then reproduce on
a fresh data ordering and architecture seed. Action accuracy, aggregate field
accuracy, training loss, and soft margins cannot promote it.

Reject the mechanism if either causal axis remains invariant under the state-
isolated gate or if gains depend on one ordering. A negative result would show
that the missing object is not merely coherent editing; the next intervention
must alter how COMMAND semantics are bound to state identities before edit
compilation.

Decision:
`compile_one_coherent_typed_state_difference_and_apply_it_atomically_before_scaling_any_dense_terminal_decoder`.
