# R12 Causal-Delta Terminal-State Preregistration

## Status

Implemented and locally qualified on 2026-08-01. Both preregistered doses are
complete on H100 and independently replicated on V100. The result is negative.

## Diagnosis

The direct terminal-state quotient reaches high aggregate terminal-field and
factual accuracy on both H100 and V100, but predicts intervention-invariant
states. WORLD and COMMAND strict gates, margin-1 rates, and DID remain zero.
The original state objective balances positive and negative classes within
each field, but it does not balance the sparse *differences between episodes*.
The optimizer can therefore fit the dominant terminal-state manifold while
discarding the small set of coordinates that carry each causal intervention.

## Treatment

Keep the 18,520,349-parameter `ParallelTerminalStateCompiler`, hard state
constraints, frozen Shohin backbone, frozen algebraic query stack, and exact
source-deleted evaluator unchanged. Add rectangle-level terminal-state delta
credit during training only.

Every batch is already an immutable set of complete 2x2 WORLD x COMMAND
rectangles. For each WORLD edge and COMMAND edge, the objective computes the
predicted terminal-state difference and exact target difference. Loss is
assigned only to support-valid coordinates whose target actually changes.
Binary fields use squared delta error; value and type use vector Brier delta
error. Each present semantic field and intervention axis receives equal
weight, independent of raw coordinate sparsity. The original full-state Brier
loss remains as an anchor.

This objective sees rectangle membership and terminal packet targets. It does
not see QUERY bytes, query targets, answer labels, oracle programs, host
execution, candidate scores, or best-of-K selection. Inference remains one
deterministic hard terminal state from initial state plus COMMAND.

## Matched Gate

Two preregistered 500-update arms use identical architecture seed 31, data seed
11, stream position zero, autonomous initial state, LR `3e-4`, clip `1.0`, and
32 evaluation batches:

- primary: full-state loss + `4.0 * causal_delta_loss`;
- dose control: full-state loss + `1.0 * causal_delta_loss`.

The exact protected checkpoint, release, algebraic reader, evaluation ordering,
and 200M system cap remain unchanged. Changed-coordinate counts and per-axis
delta losses are logged on every metric interval.

## Promotion Rule

Training loss, aggregate packet fields, exact packet count, and factual top-1
are diagnostics. The mechanism advances only if a fully autonomous arm moves
both strict WORLD and strict COMMAND above the zero matched baseline without a
large factual collapse. A margin-only result may justify one bounded extension
but is not a win. Promotion requires fresh-ordering and second-seed replication.

## Rejection Rule

Reject causal-delta credit as sufficient if both doses remain strict-zero, if
only one causal axis moves, if gains require oracle initial state, or if a gain
fails fresh population or seed replication. The next architecture in that case
must transport a sparse coherent edit object explicitly rather than asking one
dense terminal state to represent both invariant background and intervention.

Decision:
`train_terminal_state_differences_as_first_class_causal_objects_and_require_joint_world_plus_command_transfer`.

## Result

All four 500-update runs completed cleanly. Weight `1.0` reaches 60.94%
factual top-1 on H100 and V100; weight `4.0` reaches 48.24% on H100 and
60.94% on V100. Every run remains strict WORLD `0%` and strict COMMAND `0%`.
More importantly, the oracle-program/autonomous-state cross-check is invariant
for every arm: WORLD and COMMAND strict, margin-1, and DID are all exactly
zero. The H100 weight-4 fully autonomous stack shows a 5% WORLD margin-1 and
DID `0.252`, but it vanishes on V100 and under oracle-program isolation while
factual accuracy collapses; it is therefore a composition artifact, not state
reasoning.

Report SHA-256 values are:

- H100 weight 1: `5effcde6064ad349408e6d9a9b6466b47d002adc851ceacd73a6894f5e677225`;
- H100 weight 4: `9ebb1af5437cfdda25329e254847a36ba37c1367206a29524a578951f3ea1bc8`;
- V100 weight 1: `99e22ebda447dce748b1017bd11919a819a8adbb3718e4fea01becd6faac171e`;
- V100 weight 4: `90755ca4cab9a42dbf51512eda2d1e91a31309f6472ab769c61f1e66fd32f4fd`.

This closes loss-only repair. The dense absolute-state architecture can absorb
causal-delta gradients by degrading the common state estimate without learning
a stable intervention-conditioned edit. The next mechanism must make identity
transport and sparse editing distinct architectural operations.

Disposition:
`reject_loss_only_causal_delta_repair_build_explicit_identity_plus_sparse_edit_transport`.
