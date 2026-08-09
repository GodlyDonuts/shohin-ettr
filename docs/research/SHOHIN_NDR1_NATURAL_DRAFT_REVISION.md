# NDR1: Natural Draft Revision

Status: mechanism and gates frozen before draft generation.

## Hypothesis

CFR1 failed because appended synthetic faults and clean-copy presentations did
not resemble the errors produced by the B1 draft owner. NDR1 changes the data
mechanism, not a CFR hyperparameter: B1 generates one deterministic draft for
each short, fully verified, source-disjoint training problem. The revision
owner then sees the original source plus that exact model-owned draft and must
emit the untouched verified solution. There are no synthetic faults and no
clean draft equals target presentation.

## Matched Control

The control receives another source's same-domain, nearest-length B1 draft.
Source, verified target, row order, target-token multiset, initialization,
optimizer, update budget, trainable parameters, and evaluator are identical.
No source receives its own draft. The comparison isolates information in the
actual draft from generic verified source-to-solution training.

## Data Gate

The source is a fresh Qwen-tokenizer-exact 4M-target-token mix with 1,536-token
source-plus-target admission and 40/10/40/10 math/code/science/procedural
weights. Sixteen deterministic B1 shards generate at most 768 tokens per
source. The merged curricula must retain at least 90% of source rows at 4,096
tokens, have exact target multisets, zero source/donor identity matches, exact
source/draft/checkpoint hashes, and no holdout use. Failed or missing shards
fail closed.

## Capability Gate

Train exactly one aligned and one shuffled arm from immutable B1 update 256,
using identical final-four rank-8 LoRA, 512 updates, batch 1, accumulation 8,
4,096 context, learning rate `2e-5`, and matched seeds. Evaluate once on the
existing 1,289-row source-disjoint IDR1 development board. Promotion requires
all of:

- aligned at least `603/1,289`;
- aligned at least `+10` answers over shuffled;
- math at least `223`, logic/science at least `349`, code at least `17`;
- aligned 768-token exhaustion no more than `400` and no more than shuffled
  plus `25`; and
- complete matched token, parameter, memory, latency, and protected-hash
  receipts.

Any miss closes exact NDR1 without source-size, generation, update, rank,
layer, seed, decoding, parser, or threshold rescue. Holdout remains sealed
until a conjunctive development pass.
