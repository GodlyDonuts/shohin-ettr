# SLC1: Structured Ledger Compiler

Status: prospectively frozen before any GPU capability output on 2026-08-10.

## Purpose

The closed DSET-to-BSOT cascade showed that a language model can often repair a
known local error but that free-form natural trajectories do not provide stable
edit addresses. A holdout-blind CVG1 audit then rejected raw text pointer
editing: an unrestricted ordered diff copied only `44.9179%` of development
target characters and required `56.97` copy runs on average.

SLC1 tests the necessary replacement interface: can one model compile a source
problem into a compact, model-owned, addressable operation ledger without an
external solver at inference? SLC1 is an interface qualification, not a claim
of general reasoning and not the final edit architecture.

## Data

The immutable RG-v4 source has SHA-256
`a87e7c8279048d9fbdf19b245aeff194305415ea06d28258902b397bbe96875f`.
An independent exact-arithmetic compiler admitted only five executable
families and rejected every false intermediate step or terminal mismatch.

- train: `75,935` source-disjoint rows, `198,335` exact operations;
- development: `3,917` rows, `10,120` exact operations;
- operations: ADD, SUB, MUL, DIV;
- depths: one through five records;
- holdout: unused and unavailable to SLC1;
- train JSONL SHA-256:
  `6a5876f2b8eed1387c31459062102b9bd007bff99d556ee6f63c75613310f671`;
- development JSONL SHA-256:
  `760044d9b3851197988b361eba021ffbfe013600fab464731ab72de5e617ec87`.

Every target uses one canonical grammar:

```text
<LEDGER_V1>
R0|SUB|8|3|5
R1|ADD|@R0|5|10
COMMIT|@R1|10
</LEDGER_V1>
```

Record references create stable semantic addresses. Canonical materialization
round-trips every record exactly. With the pinned tokenizer, complete
source-plus-target sequences have train/development maxima of `380/355`
tokens; a 384-token context therefore has zero truncation.

## Frozen Fit

- host: pinned Qwen3.5-0.8B revision
  `2fc06364715b967f1860aea9cf38778875588b17`;
- base weights frozen;
- final four layers, rank-8 LoRA over the established full projection scope;
- alpha 16, learning rate `2e-5`, cosine decay;
- `1,024` updates, microbatch 4, accumulation 2;
- maximum 75,935 rows, deterministic data seed `2026081022`;
- model seed `2026081021`;
- 384-token context;
- greedy generation, maximum 320 new tokens, seed `2026081023`.

One two-update mechanics run may verify finite loss/gradient, memory, and
generation plumbing. It cannot change any scientific setting or count as a
capability result.

## Development Evaluation

All 3,917 source-disjoint development rows are evaluated exactly once in the
aligned arm. A matched causal control replaces each source with another source
from the same family and ledger depth whose gold ledger differs, while keeping
the recipient gold fixed. The control changes no model, decode, prompt shape,
or target distribution.

SLC1 passes only if all conditions hold:

1. syntax-valid ledgers `>=99%`;
2. exact record count `>=99%`;
3. exact operation sequence `>=95%`;
4. exact complete record sequence `>=90%`;
5. exact terminal value `>=95%`;
6. exact terminal value `>=90%` in every family;
7. exact terminal value `>=85%` at depth five;
8. aligned terminal accuracy exceeds source-shuffled accuracy by at least 65
   percentage points and source-shuffled recipient accuracy is `<=25%`;
9. zero generation exhaustion and complete hash/identity coverage.

There is no prompt, rank, layer, learning-rate, update-count, seed, or threshold
rescue. A pass qualifies the ledger compiler and opens one separately frozen
record-edit/replay transducer with clean/fault/natural pairs, deterministic
generic edit execution, forced-script intervention, and matched hidden,
shuffled, label-permuted, and full-regeneration controls. A failure closes this
standard-LoRA compiler interface; any successor must change the compiler
architecture rather than repeat this fit.

## Claim Boundary

SLC1 may establish only reliable source-to-ledger compilation and causal source
use. It does not establish autonomous general reasoning, natural-language
coverage, or a benefit from editing. Those claims require the downstream
model-owned record editor, tied recurrent replay, final renderer, and matched
answer-level controls.
