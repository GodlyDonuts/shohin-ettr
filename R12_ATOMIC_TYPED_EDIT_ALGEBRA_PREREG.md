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

The primary action-loss weight is `1.0`. Before inspecting any sealed result,
an equal-budget `4.0` dose arm is admitted because live training telemetry
localizes most residual action loss to the 256-way value payload while node,
relation, root, and disposition actions are already low. The primary remains
the claim-bearing architecture gate. The dose cannot promote on action NLL,
state loss, or interface fields; it must independently move both hard WORLD
and COMMAND under the unchanged evaluator.

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

## Result

H100 primary `725510`, V100 replication `725511`, and H100 action-weight-4
dose `725513` completed all 1,000 updates cleanly. Canonical edits exactly
reconstructed the first real target packet before every run began. The H100
primary learns low final-batch action NLL (`0.253`; value payload `1.076`) and
reaches 54.49% factual top-1, but strict WORLD/COMMAND and both margin-1 rates
remain zero. The oracle-program/autonomous-state cross has DID exactly zero on
both axes. Report SHA-256 is
`bd9e55a2dfabf632209efe7431d567f01d88add14f9ecccc558a81b40392fd9d`.

The V100 result independently returns the same isolated zeros; report SHA-256
is `b1c078e22eb2095094494a33288202ef6fa1b72404291b94805ce3185b3ef0c4`.
Increasing action weight to 4 does not repair the gate. It reaches 54.69%
factual but lowers value accuracy to 49.43%; every autonomous and isolated
WORLD/COMMAND strict, margin-1, and DID metric is zero. Report SHA-256 is
`051385f1fb3dbc7ca1111398d2f1322cfb464c4e6cebdc00ca61547d3d69406a`.

Coherent edit mechanics are therefore insufficient when the initial packet is
lossy. Training targets derived from an autonomous WORLD state include repairs
for information the frozen WORLD compiler already discarded; COMMAND cannot
identify those repairs. The next bounded diagnostic trains the same atomic
editor from oracle initial packets to isolate COMMAND binding. It is not a
reasoning claim. A positive result would require a jointly trainable WORLD
compiler plus atomic COMMAND editor; a negative result would localize failure
inside COMMAND-to-edit binding itself.

Disposition:
`retain_fixed_atomic_algebra_but_stop_asking_command_to_recover_world_information_that_the_initial_packet_erased`.
