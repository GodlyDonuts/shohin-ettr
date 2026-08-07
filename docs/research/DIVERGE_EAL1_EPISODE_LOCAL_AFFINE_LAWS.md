# DIVERGE-EAL1 Episode-Local Affine Law Induction

Status: frozen before corpus materialization or neural scoring.

## Capability hypothesis

The confirmed MZE1 executor owns recurrent operation semantics, but those
semantics are globally fixed after training. EAL1 tests a stronger capability:
a learned natural-language reader should infer previously unseen operation
laws from episode-local before/after evidence, seal only the induced laws,
delete the demonstrations, and execute longer held-out programs without a
global operation table or runtime verifier.

This is a controlled system-identification and compositional-execution gate,
not a claim of general natural-language reasoning. Exact alias scanning, the
finite field, the bounded coefficient catalog, factorized support
intersection, and typed transfer programs remain engineered priors.

## Frozen architecture

Each episode contains eight fresh opaque operation aliases. Each alias denotes
a fresh invertible `2 x 2` matrix over `Z/97Z`; its two coefficient rows are
drawn from `{-2,-1,0,1,2}^2`. Training and development use disjoint matrix
catalog partitions and disjoint opaque names.

A `397,636`-parameter byte-GRU reader assigns each of four complete integer
mentions to `BEFORE_X`, `BEFORE_Y`, `AFTER_X`, or `AFTER_Y`. Hard inference
selects one complete role permutation. Exact whole-alias scanning binds the
evidence to an episode-local operation. For each operation and output
register, a factorized law packet intersects the 25 coefficient rows
consistent with each observed transition. It commits only when every support
is singleton and otherwise fails closed.

The sealed packet retains aliases, committed rows, evidence hashes, and reader
state hash. It retains no source text or before/after values. Recurrent
execution applies only those committed rows to held-out typed programs. The
candidate runtime imports neither EAL1 data/oracle matrices nor PL1 exact
operation semantics and contains no runtime verifier.

## Frozen data and schedule

- Reader training: 100,000 unique natural transition statements, seed
  `2026080751`.
- Views: exactly half normal and half temporal-counterfactual per update.
- Optimization: 1,000 updates, batch 256, AdamW, peak learning rate `0.003`,
  cosine decay, gradient norm 1.0.
- Development: 256 episodes, seed `2026080752`.
- Episode: eight unseen matrices, three natural demonstrations per operation,
  and sixteen transfer programs.
- Transfer depths: `12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32`.
- Queries: both terminal registers for every transfer.
- Renderer split: training and development hold out before/after phrase-pair
  compositions. Sources, opaque names, and matrix catalogs must have zero
  overlap.

The first axis demonstration leaves exactly five rows possible for every
output. The second identifies the row; the third checks consistency. Thus the
one-example control must remain underdetermined by construction.

## Frozen interventions

- **Temporal counterfactual:** preserve integer values, mention order,
  operation, and registers while swapping only before/after language and role
  labels.
- **Temporal scrub:** preserve integers, order, operation, and registers while
  replacing every temporal clause by semantically neutral wording.
- **Shuffled episode evidence:** use another episode's transitions after an
  alias-only bijection, then execute the untouched target programs.
- **Unrelated law transplant:** rebind another episode's sealed rows to the
  target aliases.
- **Law reset:** execute with no packet and require abstention.
- **One-example:** retain only the first demonstration for each operation and
  require fail-closed underdetermination.
- **Oracle roles:** use assessor role labels with otherwise identical law
  intersection and execution.

## Conjunctive development gate

EAL1 passes only if every condition holds:

- normal complete-role exactness at least 99%;
- temporal-counterfactual complete-role exactness at least 99%;
- normal and counterfactual renderer floors at least 95%;
- temporal-scrub complete-role exactness at most 30%;
- learned episode-law commits and learned coefficient-row exactness at least
  99%;
- learned terminal-state and late-query exactness at least 99%;
- learned minimum depth exactness at least 95%;
- oracle-role commit, terminal state, and query exactness all 100%;
- one-example unique commits exactly zero;
- shuffled-evidence and unrelated-transplant state and query exactness each at
  most 5%;
- reset abstains on every program;
- sealed packets contain no raw evidence; and
- runtime source contains no exact EAL1/PL1 semantics or verifier.

A failure closes this exact reader/law-state gate without seed, width, update,
learning-rate, renderer, coefficient-range, or duration variants. A pass may
open fixed independent confirmation boards and then one integration with
qualified Shohin owners. It does not authorize continuation pretraining.
