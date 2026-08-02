# R12 Sparse Residual Terminal Transport Preregistration

## Status

Implemented and locally qualified on 2026-08-01. No accelerator result exists
at the time of preregistration.

## Diagnosis

One-pass absolute terminal-state prediction learns common fields but erases
causal differences. Explicit rectangle-delta loss does not repair it: four
matched H100/V100 arms remain oracle-program state-invariant, and stronger
delta weight damages factual state. The architecture is asking one dense
decoder to solve two statistically opposed tasks: preserve the large invariant
background and rewrite a small intervention-specific subset.

## Architecture

`ParallelTerminalStateCompiler(residual_edits=True)` separates those tasks.
The initial typed state is an identity highway and also directly seeds every
terminal slot query. Learned COMMAND-conditioned proposal heads emit candidate
value, type, active, root, relation, and status fields. Separate edit gates
choose, per slot or relation edge, how much of the initial field to preserve
and how much proposal to apply. Relation edit gates use an independent
factorized pair scorer. Gates start copy-biased, so the optimizer must earn a
rewrite instead of reconstructing every invariant coordinate from scratch.

Hard inference still produces one deterministic typed terminal packet with
one-or-none root, active-masked categorical fields, active-endpoint relations,
and the exact edge cap. There is no transaction schedule or claimed trace.
The runtime receives only the autonomous initial typed state and candidate-
visible COMMAND residuals. It receives no QUERY, answer, target, oracle
program, host solver, verifier, or candidate selector.

The residual path adds 1,051,670 parameters. The terminal compiler has exactly
19,572,019 trainable parameters and the complete replacement system has
175,510,913 parameters, below Shohin's 200M cap.

## Objective And Gate

The first arm uses architecture seed 31, data seed 11, autonomous initial
state, position zero, 1,000 updates, LR `3e-4`, clip `1.0`, causal-delta weight
`1.0`, and the unchanged 32-batch/512-row source-deleted evaluator. The full-
state Brier anchor and complete WORLD/COMMAND rectangle-delta loss remain
unchanged from the closed objective control. This isolates the residual edit
architecture.

Promotion requires nonzero strict WORLD and strict COMMAND in the fully
autonomous arm and corroborating movement in the oracle-program/autonomous-
state cross-check, without a large factual collapse. Any gain must reproduce
on a fresh ordering and second architecture seed. Aggregate field accuracy,
gate values, or soft deltas alone cannot promote the mechanism.

## Rejection Rule

Reject sparse residual terminal transport if state-isolated WORLD or COMMAND
remains invariant, if only one causal factor moves, or if a gain disappears on
fresh population/seed replication. The next successor would represent the
edit itself as a coherent typed object and apply it through a fixed state
algebra, rather than mixing absolute fields.

Decision:
`make_identity_transport_default_and_force_command_to_purchase_sparse_typed_edits_then_gate_state_causality`.
