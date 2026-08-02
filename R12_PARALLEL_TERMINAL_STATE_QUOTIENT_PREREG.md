# R12 Parallel Terminal-State Quotient Preregistration

## Status

Implemented and locally qualified on 2026-08-01. No H100 result existed when
this preregistration was written.

## Causal diagnosis

The strongest learned COMMAND mechanism is a one-pass parallel schedule
compiler. It can reach nonzero fully autonomous WORLD and COMMAND strict pairs,
but the effect is initialization- and population-sensitive. Better exact
schedule accuracy, grounded pointers, semantic-prefix credit, fieldwise
multi-basin averaging, and training on the deployed initial packet do not make
the result stable. The common defect is supervision of one arbitrary ordered
transaction serialization even though many schedules denote the same semantic
terminal state.

## Treatment

`ParallelTerminalStateCompiler` predicts the query-independent terminal typed
state directly from:

1. the initial typed state produced before COMMAND, and
2. frozen Shohin residuals for the candidate-visible COMMAND bytes.

It emits active slots, one-or-none root, value code, type, sparse typed
relations, committed, and halted. Hard inference enforces all typed packet
constraints and the relation edge cap. The complete 64-step learned policy is
removed from deployment. The model receives no QUERY bytes, answer token,
target packet, oracle program, best-of-K scorer, verifier, or host solver.

The production geometry is width 512, four joint slot layers, eight attention
heads, and relation rank 64. It adds 18,520,349 parameters. Replacing the dead
29,757,217-parameter recurrent reactor keeps the complete system below 200M.

## Objective

The only training target is the exact terminal packet equivalence class.
Binary fields use class-balanced Brier loss; value and type use categorical
Brier loss on target-active slots. Sparse relation loss uses the assessor mask.
The primary arm trains from Shohin's hard detached autonomous WORLD packet,
which matches deployment. An oracle-initial arm is an admissible matched
diagnostic, not a promoted result.

## First gate

- architecture seed: 31
- data seed: 11
- stream position: 0
- updates: 500
- learning rate: `3e-4`
- gradient clip: `1.0`
- evaluation: 32 batches / 512 rows
- inference: one hard terminal state, no candidate selection

The protected Shohin checkpoint, algebraic query compiler/reader, release,
source-deleted evaluator, and public gate definitions remain unchanged.

## Promotion rule

Loss, soft packet loss, per-field terminal accuracy, complete packet accuracy,
factual top-1, and oracle-program arms are diagnostics only. The mechanism is
extended only if the fully autonomous arm improves both strict WORLD and
strict COMMAND, or produces a clearly positive margin-1 precursor without a
large factual collapse. A claim requires simultaneous strict WORLD and COMMAND
improvement on three distinct held-out data orderings and more than one
architecture seed.

## Rejection rule

Reject direct terminal-state quotient transport as currently formulated if:

- it learns terminal labels but strict WORLD and COMMAND remain zero;
- only one causal factor improves;
- the result depends on oracle initial state;
- the result disappears on fresh data orderings or a second architecture seed;
- or it violates the typed hard-state or 200M parameter contracts.

## Interpretation boundary

A positive result would show that Shohin can learn a query-independent semantic
state transition without committing to an arbitrary operation serialization.
It would not by itself prove unrestricted general reasoning beyond the frozen
factorial qualification families.
