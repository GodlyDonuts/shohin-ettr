# DIVERGE-NCP1 Natural Command Pointer

Status: frozen after bounded feasibility calibration and before final data
materialization or final neural scoring.

## Capability hypothesis

EAL2 still receives each transfer program as an engineered list of operation
symbols. NCP1 removes that list:

> A content-addressed CTC pointer can read one raw variable-length natural
> command, bind unseen episode-local operation names against an eight-entry
> alias table, preserve operation order and repetition, and drive the frozen
> EAL2 law packet and recurrent executor without typed program symbols.

The 273,794-parameter pointer uses a shared byte embedding, separate
bidirectional GRUs for the command and alias table, normalized dynamic pointer
scores, one learned blank score, and greedy CTC collapse. The output classes
are episode-local alias positions rather than a global operation vocabulary.
Training-only alias spans provide frame supervision plus an auxiliary CTC
loss; inference receives no spans, target length, typed symbols, or alignment.

This is a practical scaffold-removal gate, not a novelty claim for CTC,
pointer networks, dynamic vocabularies, or recurrent encoders. The qualified
EAL2 temporal reader, exact register binder, exact law-support solver, typed
initial state/query, and exact Z/97 executor remain frozen.

## Bounded feasibility calibration

Before freeze, one explicitly non-claim pilot established that the interface
can learn. A 10,000-row treatment reached 2,048/2,048 fixed-sample program
exactness after 1,000 updates; its matched shuffled-table arm stayed 0/2,048.
On the pilot board, normal, reverse-order, and end-to-end state exactness were
99.6826%, while shuffled table, source scrub, and shuffled-table model were
0%. Renamed aliases exposed a generator-distribution mismatch. Correcting
only that pre-freeze generator to use the same random consonant-vowel carrier
as EAL2, then using 1,500 updates, produced 4,096/4,096 exact normal, renamed,
and reversed programs and 4,096/4,096 terminal states at every depth 12--32;
all three negative controls remained 0/4,096.

Those opened pilot boards cannot count as final evidence. Final development
is rotated to fresh seed `2026080798`; conditional confirmation seeds are
`2026080799`--`2026080803` and must be built before final development scoring.

## Frozen data and schedule

Training seed `2026080791` generates 100,000 command records at depths 4--20.
Every record has eight fresh, randomly permuted opaque aliases and one raw
command rendered from held-composition action/connector pairs. Development and
confirmation use fresh EAL2 episodes and commands at depths 12--32. The typed
`symbols` list is removed from every candidate-visible transfer. All final
sources, names, and episode identities must be disjoint from EAL2, NLS1, the
opened pilot boards, and each other, and every final byte must independently
regenerate.

Two matched 273,794-parameter arms start from the same state and train for
exactly 1,500 AdamW updates, batch 128, learning rate 0.001:

- treatment receives the correct episode alias table;
- shuffled-table control receives the same commands, targets, minibatches,
  and span supervision, but its alias table is cyclically shifted.

Both arms receive the same parameters, data, update count, optimizer schedule,
and charged examples. EAL2 is not retrained.

## Frozen development gate

All conditions are conjunctive:

1. inherited EAL2 temporal reading and law commitment are each at least 99%;
2. raw-command program exactness is at least 99%, with at least 95% at every
   depth 12--32;
3. independently renamed aliases and reversed command order are each at least
   99% program exact;
4. source-deleted terminal-state and late-query exactness are each at least
   99%, including at least 99% state exactness after alias renaming;
5. command-source scrub, shuffled alias table, and the independently trained
   shuffled-table model are each at most 5% program exact;
6. initialization, parameters, data, updates, batch, and optimizer schedule
   are matched; checkpoint/report custody is exact; and
7. the candidate runtime contains no exact alias search, typed program list,
   target length, or training alignment at inference, and command bytes are
   absent before execution.

A development miss closes NCP1 without width, layer, update, seed,
learning-rate, renderer, decoder, threshold, or duration variants. A pass
opens the five already-built confirmation boards exactly once with unchanged
weights. NCP1 does not authorize continuation pretraining or an unrestricted
reasoning claim.
