# DIVERGE-NLS1 Neural Episode-Law Synthesizer

Status: frozen after EAL2 confirmation and before NLS1 data materialization or
neural scoring.

## Capability hypothesis

EAL2 confirms the full observable-semantic path but still uses exact support
intersection over 25 coefficient rows to induce each episode-local law. NLS1
changes exactly that owner:

> A permutation-invariant neural synthesizer can consume three complete
> before/after transactions from the qualified EAL2 reader, infer each unseen
> episode-local law, and recurrently execute held matrix combinations without
> an exact support solver.

The confirmed 397,250-parameter EAL2 reader and its checkpoint remain frozen.
NLS1 adds one shared value embedding, one demonstration encoder, sum pooling,
and two categorical row heads. Demonstration order cannot identify a law.
The synthesizer exposes the bounded 25-row output vocabulary but contains no
support intersection or oracle transition. The already-qualified typed EAL2
executor remains exact and frozen; NLS1 changes only episode-law induction.

This is a scaffold-removal test, not a novelty claim for set encoders,
hypernetworks, categorical program synthesis, or recurrent execution.

## Frozen data and schedule

Training contains 100,000 typed three-demonstration law episodes under seed
`2026080781`. Matrices come only from the existing EAL2 training partition;
all individual coefficient rows occur in both train and held-out partitions,
while complete 2x2 matrices are disjoint. Development uses 256 fresh EAL2
episodes under seed `2026080782`. Five conditional confirmation boards use
seeds `2026080783`--`2026080787`. Sources, aliases/registers, and episode
identities must be disjoint from EAL2 training/development/confirmation and
between every new board. Every artifact must regenerate byte-for-byte before
training.

Two matched 216,946-parameter synthesizers start from the same state and train
for exactly 500 AdamW updates, batch 2,048, peak learning rate `0.003`:

- treatment receives the true three before/after transactions;
- shuffled-outcome control receives the same before states and labels, but
  after states are shifted between examples inside every identical minibatch.

Both arms receive the same sampled row order, optimizer schedule, update
count, parameter count, and charged examples. NLS1 does not retrain EAL2.

## Frozen development gate

All conditions are conjunctive:

1. inherited EAL2 normal and temporal-counterfactual complete reading are at
   least 99%, and temporal scrub is at most 30%;
2. treatment coefficient-row, terminal-state, and late-query exactness are
   each at least 99%;
3. the treatment terminal-state floor is at least 95% at every held depth
   from 12 through 32;
4. temporal-counterfactual terminal-state exactness is at least 99%;
5. shuffled-outcome-model and after-value-scrub terminal-state exactness are
   each at most 5%;
6. one-example terminal-state exactness is at most 20%, because one transition
   does not identify both coefficients;
7. temporal-scrub terminal-state exactness is at most 10%;
8. initialization, parameters, data, update count, batch, and optimizer
   schedule are matched; checkpoint/report custody is exact; and
9. source deletion and runtime-source audits pass.

A development miss closes NLS1 without width, embedding, update, seed,
learning-rate, renderer, threshold, or duration variants. A pass opens the
five already-built confirmation boards exactly once with the same frozen
reader and synthesizer checkpoints. NLS1 does not authorize continuation
pretraining or an open-domain reasoning claim.
