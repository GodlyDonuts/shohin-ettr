# R12 ETTR-IL-v3 Initializer Data Protocol

**Protocol ID:** `R12-ETTR-IL-v3-initializer`

**Status:** frozen before population generation

**Purpose:** initialize and qualify the 67,697,771 newly added ETTR parameters
without changing the frozen 192,779,435-parameter neural architecture.

## 1. Scope and non-claims

This protocol constructs data only. It authorizes no model update, checkpoint
write, continuation pretraining, development-score opening, or reasoning claim.
The protected step-300k Shohin checkpoint remains read-only.

The v2 architecture remains fixed:

- 64 packet slots;
- 16 available runtime slots at positions 32 through 47;
- 64 recurrent transaction positions;
- 256 categorical value codes;
- 256 relation edges;
- WORLD width 192;
- COMMAND width 96;
- QUERY width 48;
- 67,697,771 trainable ETTR parameters; and
- 192,779,435 complete-system parameters.

The v3 corpus is an initializer and systems-training population. A later
claim-bearing experiment must use untouched evaluation data and matched causal
controls.

## 2. Why v2 data generation is superseded

The v2 generator required every selected training core to be a strict Boolean
checkerboard. Exhaustive CPU probing found that most required rewrite theories
have no such core in the frozen 46-world, two-command domain. Fold 0 training
has no usable rewrite depth-1 theory under the frozen ownership hash. Several
resource cells are also empty. The v2 specification additionally refers to a
frozen surplus without assigning a number.

These are data-domain defects, not failures of the frozen neural architecture.
No v2 production row was selected before the defect was found.

V3 therefore:

1. uses broad exact component and joint supervision for initialization;
2. reserves strict checkerboards for counterfactual curriculum cells and
   evaluation;
3. replaces only the pathological rewrite semantic family;
4. expands resource worlds to every capacity-valid marking when needed;
5. preserves all architecture widths and parameter counts; and
6. fixes every quota, surplus, selection, and split rule before generation.

## 3. Quality definition

An admitted core must be:

- **correct:** primary and independent replay executors agree at every step;
- **causally informative:** controlled WORLD, COMMAND, or QUERY changes have
  independently recomputed effects;
- **architecture-native:** packet and transaction targets fit the deployed
  state and trace geometry;
- **diverse:** no dominant law, template, topology, renderer, or length route;
- **nontrivial:** cheap shortcut features cannot determine the target;
- **nonleaking:** split, answer, ontology, and stratum are absent from
  candidate-visible metadata and path structure;
- **replayable:** a fresh process can reproduce the terminal packet from the
  frozen source and trace; and
- **immutable:** selected data and manifests are content addressed.

Teacher language models may propose paraphrases or adversarial mutations. They
never create semantic targets. A teacher-produced surface is admitted only
after exact round-trip or execution-equivalence verification.

## 4. Semantic families

### 4.1 Typed Horn closure

Retain the independently executed Horn family, but admit broad valid episodes
in addition to strict checkerboards. Cover primitive facts, derived facts,
multi-hop closure, irrelevant evidence, contradiction, and source-deleted
queries.

### 4.2 Bounded typed local rewriting

Replace the decreasing-tree rewrite board with a fixed-width local system:

- six alternating typed runtime registers;
- four legal symbols for each register type;
- 4,096 initial worlds per theory;
- six primitive size-preserving local rewrite laws;
- fifteen theories formed by all two-law combinations;
- commands selecting an opaque theory-local law slot, local site, and
  direction;
- applied, blocked, rejected, and terminal outcomes;
- depths 1 through 6;
- at most two runtime-register writes per primitive operation; and
- structural Boolean queries over equality, symbol counts, and guarded local
  patterns.

State size cannot grow with depth. Every primitive world/law/operation is
exhaustively checked by two independent executors before candidate search.

### 4.3 Guarded resource processes

Retain guarded resource semantics and expand the world domain from the old
`3^4` subset to all `4^4 = 256` capacity-valid markings if a quota cell needs
it. Include applied, blocked, skipped, deadlocked, and halting trajectories.
Balance conservation, capacity, guard, cursor, and order-sensitive effects.

## 5. Curriculum stages and exact selected quotas

The selected training population contains exactly 40,000 semantic cores:

| Stage | Cores | Full-objective positions | Purpose |
|---|---:|---:|---|
| compiler grounding | 8,000 | 270,336,000 | WORLD to complete typed packet |
| atomic transactions | 8,000 | 270,336,000 | every opcode and operand head |
| dependent composition | 12,000 | 405,504,000 | depths 2-6 and dependency chains |
| query/counterfactual grounding | 6,000 | 202,752,000 | packet-to-answer and minimal flips |
| closed-loop invariance | 6,000 | 202,752,000 | full source-deleted causal episodes |
| **total** | **40,000** | **1,351,680,000** | no repeated exposure counted |

Each selected core has exactly four admitted semantic views, 16 rows per view,
64 rows total, and 33,792 charged full-objective positions.

Additional selected populations:

- development: 5,000 cores;
- sealed confirmation: 5,000 cores;
- training reserve: 10,000 cores;
- development reserve: 1,250 cores; and
- confirmation reserve: 1,250 cores.

Every stage is balanced across the three semantic families by quotient and
deterministic cyclic remainder. Within a family it is balanced over law,
depth, outcome, operation class, query class, renderer, trace-length bin,
active-slot bin, relation-count bin, and topology class subject to constructive
feasibility.

The train stage totals above are authoritative. Development, confirmation,
and reserve stage totals use the same exact `20/20/30/15/15` proportions,
with any indivisible remainder assigned by the frozen master-seed PRF. One
deterministic stage-by-family transportation matrix jointly satisfies both
the stage totals and each split's balanced family totals; independent
per-row rounding is forbidden because it can produce inconsistent marginals.

## 6. Candidate surplus and selection

For an exact quota `q`, the fully admitted surplus is:

```text
surplus(q) = max(16, ceil(q / 4))
```

Candidate generation continues until the cell reaches at least
`3 * (q + surplus(q))` semantically valid candidates or exhausts its finite
domain. Exhaustion below that threshold blocks publication.

Selection is deterministic and label-independent after admission:

1. assign split ownership from a hash of the canonical graph-isomorphism
   orbit and the v3 master seed;
2. reject cross-split orbit, world, or bound-command reuse;
3. rank by deterministic coverage gain, then the split-key PRF, then canonical
   bytes;
4. select quota without replacement;
5. freeze the required reserve before publication; and
6. never regenerate or replace a selected item after fitting begins.

The coverage-gain function and all bins are part of the generator source hash.
No human or model may inspect labels and then alter a quota, seed, renderer,
or ranking rule.

## 7. Required geometry per core

Every closed-loop or counterfactual core contains:

- two meaning-preserving surface views;
- one minimal WORLD counterfactual;
- one minimal COMMAND counterfactual;
- two distinct query denotations;
- one query counterfactual when representable;
- one invalid, blocked, or no-op near-neighbor;
- exact initial and terminal packet targets;
- exact per-step transaction targets;
- primary and replay execution receipts; and
- trace corruptions for opcode, source, target, relation, value, cursor, and
  premature termination where each corruption is legal to construct.

Compiler and atomic stages may use a reduced pair geometry, but their selected
schedule must include the same four-view expansion and exact source deletion.

Strict 2-by-2 checkerboard balance is mandatory only for designated causal
rectangles. Broad component episodes need balanced target marginals and
controlled near-neighbors but are not discarded merely for lacking XOR
structure.

## 8. Presentation and natural-language bridge

Each latent core is rendered through controlled presentation orbits:

- canonical structured form;
- alpha-renamed and declaration-reordered form;
- relation-reified or alias-split form; and
- verified natural-language form where the family supports lossless
  round-trip parsing.

Surface diversity never substitutes for semantic diversity. Reports count
latent cores, executions, views, rows, stored tokens, and scheduled exposures
separately.

Natural-language surfaces originate from the formal latent program. A
paraphrase is rejected unless a fresh parser reconstructs the same canonical
semantic object or an independent equivalence checker proves identical
execution over the complete bounded witness set.

## 9. Admission gates

Publication fails unless all gates pass:

1. primary/replay agreement for every primitive and selected composition;
2. exact replay of every transaction trace to its terminal packet;
3. every packet fits 64 slots and 256 edges;
4. every trace fits 64 transactions;
5. exact selected quotas and frozen reserves;
6. zero malformed rows or conflicting packet/query targets;
7. zero duplicate core IDs, raw rows, token sequences, semantic worlds,
   bound commands, alpha-normalized ASTs, or graph-isomorphism orbits where
   disjointness is required;
8. zero train/development/confirmation normalized 13-gram overlap;
9. balanced family, law, operation, depth, query, outcome, renderer, packet
   density, and trace-length marginals within the frozen tolerances;
10. meaningful supervision for every deployed opcode, including ALLOC, CLEAR,
    SET_ROOT, WRITE, LINK, UNLINK, and terminal actions;
11. presentation invariance and semantic-twin sensitivity;
12. world-only, command-only, query-only, length-only, bag-of-token, and
    metadata shortcut reports;
13. chance-level answer prediction from forbidden metadata routes;
14. byte-identical independent audit replay;
15. immutable source, environment, tokenizer, generator, shard, and manifest
    hashes; and
16. successful Hugging Face download and local hash verification in a fresh
    directory.

Shortcut reports are diagnostic during the pilot. Production publication
requires either chance-level performance or a preregistered repair followed by
a new protocol version. No post-hoc filtering is permitted under this ID.

## 10. Storage schema

The dataset has physically separated configurations:

- `training`: model-visible WORLD, COMMAND, QUERY, packet, trace, and target
  tensors plus opaque grouping keys;
- `audit`: assessor-only latent semantics, primary/replay receipts, coverage,
  and hash lineage;
- `development`: development rows and their audit records;
- `reserve`: frozen unused replacement candidates; and
- `confirmation-commitments`: hashes and counts only.

Confirmation payloads are stored in a separate private repository or encrypted
archive and are never mounted during fitting.

Shards are deterministic compressed JSONL or Parquet, named by content hash.
The root manifest records every shard's path, bytes, rows, SHA-256, schema,
split, stage, and family. Paths contain no answer, ontology, renderer, depth,
or stratum label in the model-visible configuration.

The initial private Hugging Face repository is:

```text
Godlydonuts/shohin-ettr-il-v3
```

Publication uses runtime credentials, private visibility, no token logging,
and an immutable revision receipt.

## 11. Stokes production custody

Generation runs CPU-only on Stokes:

- one commit-pinned clean source snapshot;
- Slurm job arrays split by deterministic candidate bucket;
- no GPU request;
- no model or checkpoint access;
- no overlapping writer for a bucket;
- no-replace partial and final directories;
- resumability only through completed content-addressed shards;
- independent audit jobs with read-only inputs; and
- publication only after the global manifest passes.

The pilot must measure candidates per CPU-second, compressed bytes per core,
peak memory, audit cost, and projected total wall time. Full generation is not
submitted until all three families pass cardinality and quality gates.

## 12. Phase order

1. Freeze this protocol and all executable schemas.
2. Run local unit and exhaustive primitive tests.
3. Run one bounded Stokes pilot for every family/stage combination.
4. Freeze the pilot report and production array plan.
5. Generate candidate shards.
6. Independently replay and audit candidates.
7. Deterministically select quotas and reserves.
8. Materialize training and assessor configurations.
9. Freeze the global manifest and dataset card.
10. Upload private Hugging Face shards.
11. Download into a fresh directory and verify all hashes.
12. Report zero-update data readiness.
13. Request separate authorization before any fitting.
