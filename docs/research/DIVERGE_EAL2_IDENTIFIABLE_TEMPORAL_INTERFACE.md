# DIVERGE-EAL2 Identifiable Temporal Interface

Status: frozen after the EAL1 failure and before EAL2 neural scoring.

## Structural correction

EAL1 asked one reader to infer two independent coordinates for every numeric
mention: temporal identity (`BEFORE`/`AFTER`) and episode-local register
identity (`X`/`Y`). The source explicitly expresses temporal identity, but the
random register names have no intrinsic X/Y meaning and EAL1 did not provide
their episode-local table to the reader. Its 50% raw-role plateau and
approximately 31% complete accuracy are the expected identifiability failure.

EAL2 does not change data, seed, width, duration, optimizer, coefficient
catalog, law state, executor, controls, or thresholds. It assigns the two
coordinates to their qualified owners:

- a 397,250-parameter byte-GRU owns only natural `BEFORE` versus `AFTER`
  semantics;
- exact whole-register scanning owns the disclosed episode-local X/Y table;
- their Cartesian composition yields the four complete transition roles;
- unchanged factorized support intersection induces each unseen operation;
  and
- unchanged sealed recurrent execution answers the depth-12--32 transfers.

This is a structurally different interface, not an EAL1 hyperparameter rescue.
It asks the learned component to infer only information observable in its
input while preserving model ownership of natural temporal semantics.

## Frozen data, training, and controls

EAL2 generates a fresh 100,000-statement/256-episode corpus under seeds
`2026080761` and `2026080762`. Training and development use disjoint renderer
pairs, but every individual before phrase and every individual after phrase
appears in both splits; this holds out compositions without demanding lexical
knowledge absent from a scratch byte reader. Sources, names, and matrix
catalogs must have zero overlap and a second generation must be byte exact.
Training remains 1,000 AdamW updates, batch 256, peak learning rate `0.003`,
cosine decay, and exactly half normal plus half temporal-counterfactual views.
The full charged update budget matches EAL1.

Every EAL1 intervention remains unchanged: temporal counterfactual, temporal
scrub, shuffled episode evidence, unrelated law transplant, reset,
one-example underdetermination, and oracle temporal roles. The sealed packet
still contains no demonstrations. The transitive EAL1/EAL2 candidate runtime
must contain no exact operation table or runtime verifier.

## Conjunctive gate

EAL2 passes only if all frozen EAL1 thresholds hold with temporal-reader
semantics substituted for the impossible four-role reader:

- normal and temporal-counterfactual complete temporal assignment at least
  99%, with at least 95% per renderer;
- temporal-scrub complete assignment at most 30%;
- learned law commit, coefficient-row, terminal-state, and query exactness at
  least 99%, with at least 95% at every held depth;
- oracle temporal roles give 100% law commit, state, and query exactness;
- one-example unique commits exactly zero;
- shuffled-evidence and unrelated-transplant state/query exactness each at
  most 5%;
- reset always abstains; and
- source deletion and runtime-source audits pass.

A miss closes EAL2 without local seed, width, update, learning-rate, renderer,
or duration variants. A pass may open fixed independent confirmation boards
and one composition with qualified Shohin owners. It does not authorize long
continuation pretraining.
