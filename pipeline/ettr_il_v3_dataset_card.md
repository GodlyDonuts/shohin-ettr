---
license: other
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - shohin
  - synthetic
  - reasoning
  - source-deleted
  - state-transition
pretty_name: Shohin ETTR-IL-v3 Initializer
---

# Shohin ETTR-IL-v3 Initializer

This private dataset initializes Shohin's Endogenous Typed Theory Reactor
(ETTR). It contains formally generated, independently replayed
WORLD/COMMAND/QUERY episodes with typed packet, transaction-trace, terminal
packet, and answer supervision.

## Intended use

The dataset is intended only for the frozen 192,779,435-parameter Shohin ETTR
architecture and matched scientific controls. It is not a general-purpose
instruction dataset.

## Data construction

Every semantic target is produced by a formal primary executor and a
structurally independent replay executor. A row is rejected if the executors
disagree at any intermediate state or if replaying its generic transaction
trace does not reproduce its terminal packet.

The semantic families are:

- typed Horn closure;
- bounded typed local rewriting; and
- guarded resource processes.

The training curriculum covers compiler grounding, atomic transactions,
dependent composition, query/counterfactual grounding, and closed-loop
source-deleted invariance.

## Scale target

The selected training population contains:

- 40,000 unique semantic cores;
- four controlled views per core;
- 2,560,000 expanded rows; and
- 1,351,680,000 charged full-objective positions.

Repeated optimizer exposure is not counted as new data.

## Configurations

- `training`: candidate-visible source segments and architecture targets.
- `audit`: assessor-only latent semantics and independent execution receipts.
- `development`: held-out development rows and receipts.
- `reserve`: frozen unused candidates.
- `confirmation-commitments`: hashes and counts only.

Confirmation payloads are stored separately and are not mounted during
fitting.

## Quality controls

The release manifest binds:

- primary/replay agreement;
- exact trace-to-packet replay;
- packet and trace capacity;
- semantic and surface deduplication;
- graph-isomorphism split ownership;
- train/development/confirmation decontamination;
- operation, law, depth, query, outcome, renderer, topology, and packet-density
  coverage;
- counterfactual sensitivity;
- shortcut diagnostic reports; and
- every source, environment, tokenizer, shard, and manifest SHA-256.

See `R12_ETTR_IL_V3_INITIALIZER_DATA_PROTOCOL.md` in the source repository.

## Limitations

This dataset does not prove that Shohin reasons. It defines a high-quality
initializer and controlled evaluation substrate. Synthetic transfer is
bounded by its formal families, renderers, and query grammar. Natural-language
surfaces are admitted only when they round-trip to the same formal semantics.

No protected Shohin checkpoint or model weight is included.

