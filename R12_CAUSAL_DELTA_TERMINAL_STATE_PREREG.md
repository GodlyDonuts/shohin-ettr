# R12 Causal-Delta Terminal-State Preregistration

## Status

Implemented and locally qualified on 2026-08-01. No accelerator result exists
at the time of preregistration.

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
