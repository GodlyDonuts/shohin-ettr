# NDR1: Natural Draft Revision

Status: mechanism and gates frozen before draft generation; source rebuild
required after a pre-output overlap audit found that the first prepared mix did
not enforce the stated evaluation boundary.

## Pre-Output Custody Correction

The first prepared source (`e9965343...4389`) was never consumed by a draft
owner or fit. A read-only comparison against the original IDR1 source questions
found 34 exact development overlaps, 26 exact holdout overlaps, and additional
split-unique word-13-gram collisions. Filtering that already-selected file
would retain 10,918/11,276 rows but disproportionately remove science rows, so
it is not an admissible training source.

The corrected source must be rebuilt from the same raw corpora, tokenizer,
seed, token budget, and domain weights. Before tokenization and deterministic
selection, it removes exact word-normalized source questions and every word
13-gram occurring in exactly one row within either bound evaluation split.
Repeated split boilerplate is not an exclusion gram. Selection then backfills
from remaining candidates until every original group token quota is met. The
report must bind both reference paths and SHA-256 values, exclusion-set sizes,
global and per-source drop counters, and the final output hash. Draft generation
remains closed until an independent audit reports zero exact and zero protected
13-gram overlap.

The first hash-bound rebuild (`750210`) met every domain token quota, but its
independent audit (`750214`) failed before GPU release because punctuation-
equivalent source questions remained duplicated under the builder's legacy
whitespace-only identity. The corrected filtered path uses the same word-
normalized identity for exact-overlap filtering and source deduplication. This
is an input-custody repair with unchanged corpora, tokenizer, seed, quotas, and
capability gate; the rejected build is not admissible training data.

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
weights, rebuilt prospectively under the corrected overlap boundary above.
Sixteen deterministic B1 shards generate at most 768 tokens per
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
