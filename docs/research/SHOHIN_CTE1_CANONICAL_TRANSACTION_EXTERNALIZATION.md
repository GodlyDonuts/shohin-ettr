# CTE1: Canonical Transaction Externalization

Status: data/mechanics pass; frozen fit running; no capability output yet

Date: 2026-08-10

## Hypothesis

DTC1 proves that explicit model-owned arithmetic transactions are causal and
execute reliably, but the ordinary direct owner emits an accepted transaction
on only `257/666` development problems. CTE1 tests whether this is a
post-training target defect rather than an architectural limit.

Train the same pinned Qwen3.5-0.8B backbone to externalize one compact,
canonical arithmetic transaction trace directly from each word problem. A
generic grammar lowers the generated trace into the unchanged typed graph and
frozen learned LAM1 executes it. No learned fixed-slot graph decoder is used.

CTE1 changes the supervised representation, not model scale, data identities,
optimizer budget, or executor. It is distinct from NMC1's result-free register
language: the target uses the familiar GSM annotation form that the pretrained
model already partially emits. It is also distinct from SLC1's broad synthetic
addressable ledger and from DTMC1's nonautoregressive full-graph prediction.

## Canonical Target

An independent training-only builder reads each immutable admitted gold
register program and renders every causal record as:

```text
<<fully_parenthesized_expression=exact_result>>
...
#### exact_final
```

`PUSH` operands use their exact rational surface. `LOAD` operands use the
exact result of the referenced prior record, making the textual alias
recoverable as a causal `STATE` link. Unary and binary operations are fully
parenthesized. The final line is emitted for ordinary deployment readability,
but DTC1 parsing ignores it; only the typed transaction graph reaches LAM1.

The builder must round-trip every target through the already frozen DTC1
parser and exact assessor, reproduce the immutable answer, preserve causal
state ownership, and report all source/state/literal reads. It may not use the
public test or alter source identities. Any row that fails exact mechanics is
excluded and reported before training; admission requires all 6,333 train and
666 development rows.

Training prompt:

```text
Emit a concise arithmetic transaction trace for the word problem. Use
<<expression=result>> for every step, then write #### followed by the final
result. Emit no other text.

PROBLEM:
{source}
```

## Frozen Model And Budget

- exact pinned `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`;
- fresh final-four-layer rank-8 LoRA over all linear projections, alpha 16;
- base weights frozen, BF16, no quantization;
- exactly 1,024 updates, batch 4, accumulation 2;
- AdamW, LR `2e-5`, existing trainer defaults and gradient clipping;
- exact NMC1 model/data seeds `2026081051/2026081052`;
- all 6,333 existing source identities, no selection or curriculum;
- 1,024-token training context with zero retained truncation;
- greedy no-thinking evaluation, maximum 512 new tokens, seed `2026081053`;
- unchanged frozen learned-LAM1 checkpoint.

This exactly matches NMC1 direct/program training examples, optimizer steps,
adapter geometry, learning rate, and seeds. Target token count is reported,
not force-matched, because representation compactness is part of the tested
intervention.

## Development Controls

Evaluate exactly once on the existing 666 source-disjoint rows:

1. aligned source;
2. deterministic same-register-depth source shuffle, scored against the
   untouched target answer;
3. state-read reset to zero on the aligned compiled graph;
4. frozen LAM1 opcode permutation;
5. immutable direct owner `267/666`, NMC1 `0/666`, TMC1 `44/666`, DTMC1
   `45/666`, and DTC1 `108/666` as fixed references.

## Prospective Gate

All conditions are conjunctive:

- all 6,333 train and 666 development canonical targets pass exact CPU
  round-trip mechanics;
- zero retained source/target truncation;
- at least `600/666` generated traces compile and execute normally;
- aligned reaches at least `300/666` exact answers;
- aligned exceeds the immutable direct owner by at least `33` answers;
- source shuffle is at most `67/666`;
- at least 300 aligned rows contain a causal state read;
- state reset loses at least 20 points on aligned-correct linked rows;
- opcode permutation loses at least 30 points from aligned;
- zero normal execution invalidity among compiled rows; and
- no public-test access.

A pass opens one separately frozen public GSM8K evaluation of CTE1 and the
direct owner. A miss closes exact CTE1 without target-format, punctuation,
prompt, rank, layer, update, LR, seed, decoding, parser, or threshold variants.
No output fallback, verifier, selector, host repair, or answer extraction may
be added after scoring.

## Claim Boundary

A pass would establish that compact canonical trace post-training can connect
a small pretrained language owner to causal learned execution and improve its
source-disjoint arithmetic reasoning. It would not prove unrestricted general
reasoning, architecture novelty for transaction notation, or a LAM1 holdout
claim outside this separately defined GSM pathway.

## Data And Mechanics Result

CPU job `750045` admits all `6,333` train and `666` development identities.
The exact canonical corpus contains `20,678 / 2,168` transactions and
`15,654 / 1,629` cross-record register loads. Every target parses and executes
to the immutable terminal answer with no train/development overlap and no
public-test access. Train/development SHA-256 values are
`8fb68943...6625` and `aff46617...eb04`; report SHA-256 is
`abb12785...d556`.

Tokenizer audit `750050` passes with zero truncation. Maximum complete lengths
are `393/1024` train and `335/1024` development tokens. The train target
contains 276,777 charged response tokens; development contains 29,215. Audit
SHA-256 is `6f999a91...43b8`.

Immutable runtime `1e21f38` has manifest SHA-256 `9216c99e...48c1`. Frozen
fit `750074` is running on one H100. No CTE1 capability output exists yet.
