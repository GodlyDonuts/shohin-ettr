# R12 Occurrence-Linked Sticky Schedule Preregistration

## Question

Can a target-free equality circuit over repeated token-native identifiers make
the strongest exact-executor ETTR path compositionally stable?

The current narrow sticky scheduler can learn local transaction fields and has
produced the strongest simultaneous fully autonomous causal result, but that
result is basin-sensitive. Frozen Shohin residuals do not reliably expose that
two opaque codewords in different grammar positions denote the same local
entity. This arm supplies that missing equality relation before any schedule
decision is made.

## Architecture

The treatment changes only COMMAND memory inside
`ParallelAddressedTransactionCompiler`:

1. Recover the leading syntax document from the public token-native grammar and
   delete deterministic transport cover.
2. Embed public call head, arity, integer, renderer, and physical-position
   roles without decoding an ontology.
3. Build an exact within-document equality matrix over bounded local identifier
   codewords.
4. Mean-pool contextual occurrence states across equal identifiers and
   broadcast the shared state back to every occurrence.
5. Compile one sticky transaction schedule and replay it through the unchanged
   exact typed-state algebra.

The compiler receives no QUERY, answer, terminal target, oracle program,
candidate selection, or host solver. A one-to-one renaming of identifier
codewords with fixed contextual memory must leave the schedule unchanged.

## Matched Gate

- protected checkpoint: `ckpt_0300000.pt`
- architecture/data seeds: `31/11`
- initial state: oracle, matching historical control `725460`
- position: `0`
- updates: `1,000`
- learning rate: `3e-4`
- gradient clip: `1.0`
- scheduler: width `384`, layers `3`, heads `8`, ungrounded pointers
- evaluation: unchanged 32-batch source-deleted four-arm board
- exact executor and typed query reader: unchanged

The V100 run is diagnostic. Any promotion claim requires the hash-bound H100
run. The arm advances only if both fully autonomous WORLD and COMMAND strict
causal gates improve over the matched control without factual collapse. Local
schedule accuracy, oracle-state performance, margin-only movement, or a
best-of-K result is insufficient.

The implemented narrow treatment has `10,924,449` trainable scheduler
parameters and a `166,863,343`-parameter complete system, leaving
`33,136,657` parameters below the hard 200M cap.

## Failure Conditions

The lane closes as a standalone repair if it remains causally invariant, moves
only one strict gate, loses broad factual behavior, depends on identifier
ordinals, violates the 200M complete-system cap, or fails independent
source-deleted replay. Any surviving result must be repeated across fresh
architecture and data seeds before it can become the training architecture.
