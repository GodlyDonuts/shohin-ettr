# Shohin Phase 2 Scale Decision

Status: capacity-boundary decision complete, 2026-08-08. The protected 4B
product aggregate and frozen 0.8B transfer both closed under their original
conjunctive gates. Under the former scratch contingency the selected trunk
would be `shohin_920m`. The project has since redirected to transferable
reasoning on pretrained hosts; this result does not authorize a scratch
canary or large pretraining launch.

## Evidence Boundary

The original 125.08M Shohin completed 300,000 updates and 157.286B nominal
token presentations over a 57.826B-token mounted corpus. It is a useful
protected baseline, but its public and direct-interaction results do not meet
the project objective.

The strongest causal result is now same-family internal draft/revision:

- pinned Qwen3.5-4B draft owner;
- the same 4B base with a learned revision state;
- one complete internal draft followed by one complete revised answer;
- no external proposal model, tool, verifier, answer router, or host solver at
  inference.

On source-disjoint data, trained revision beats the unchanged second pass by
`+158/1,289` development answers and `+174/1,279` holdout answers. Every math,
logic/science, and executable-code delta is positive. On the protected
568-case product board, treatment scores `320/538 = 59.48%` and `61.391%`
five-domain macro versus unchanged-B1 `272/538 = 50.56%` and `51.053%`.
Science improves `52->90` and code `6->21`, but GSM8K, MATH-500, and BBH logic
regress by `2/1/2` answers. The no-domain-regression product gate therefore
fails despite the `+48` aggregate lift. The 4B artifact is retained as causal
evidence and is not packaged as a qualified product.

## Candidate Trunks

Both scratch candidates use the existing pre-norm GQA/SwiGLU/QK-norm Shohin
trunk, tied embeddings, a 4,096-token training context, and a 49,152-token
vocabulary. Parameter counts are exact and exclude no trainable tensor.

| Candidate | Layers | Width | Heads/KV | FFN | Exact parameters |
|---|---:|---:|---:|---:|---:|
| `shohin_390m` | 30 | 1,024 | 16/4 | 2,816 | 388,563,712 |
| `shohin_920m` | 32 | 1,536 | 24/8 | 4,352 | 918,656,512 |

The training CLI requires `--vocab-size 49152`. The selected vocabulary is the
Apache-2.0 SmolLM2 49,152-token tokenizer, physical SHA-256
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`;
it is not a Qwen tokenizer. A hash-bound 500-document comparison across
FineWeb-Edu, peS2o, Essential-Web, bounded FinePDF, and code measured 759,779
tokens versus 789,331 for Shohin's historical 32K tokenizer, a 3.744% token
reduction overall and 7.19% on code, with zero encode/decode/re-encode
mismatches in either arm. The report SHA-256 is
`777d0fd6a28153be73bf55370e95a784582edc37a6a7173021d4d2bc6e15ea2e`.
Tokenizer identity remains an explicit physical launch field rather than being
hidden in the size preset.

The deployed reasoning architecture is a shared trunk with two model-owned
temporal roles:

1. **Draft role:** generate one complete source-only trajectory.
2. **Revision role:** reread source plus the internal trajectory and generate
   one coherent corrected answer.

The initial implementation may store the two role states as adapters. A later
joint checkpoint may package them together, but packaging cannot change the
measured inference path.

## Scale Selection

The decision is capability-driven rather than parameter-count-driven.

1. Preserve the closed 4B mixed failure without local threshold, prompt,
   decoding, rank, or duration rescue variants.
2. Run one matched 0.8B internal-draft/revision transfer using the already
   pinned Qwen3.5-0.8B B1 state, exact source-disjoint bank, exact 256-update
   revision schedule, and unchanged control.
3. Select `shohin_390m` only if 0.8B preserves at least a five-point revision
   gain with nonnegative math, logic/science, and code deltas on both splits.
4. Otherwise select `shohin_920m`. Do not spend the remaining compute on a
   scratch scale already contradicted by the capacity-boundary result.

All seventeen frozen 0.8B source-only draft shards completed. Complete runtime
r2 replay `746027` reproduced the 8,392-row merged draft bank and built
9,655/1,289/1,279 train/development/holdout rows. The legacy B1 checkpoint
encoded a frozen trunk as `unfreeze_layers=null`; metadata-only migration
`746045` normalized this to numeric zero while proving all 62 LoRA tensors and
optimizer/state bitwise equal. Training `746046` completed all 256 updates in
`13m38s`, charging `365,028` tokens at `455.0 tok/s`; checkpoint/report
SHA-256 values are `540771a3cd6c446c6fb90e225b1b1dc69050152669152761b129d58e1a41b357`
and `8a756b50de52545f552754e169914c320df85119f52afc6e5896b9c7e79b466a`.
Eight trained and eight unchanged-B1 evaluation shards ran under the same
evaluator, followed by deterministic split merges and one conjunctive scale
comparison. All completed cleanly; the comparison report SHA-256 is
`6f42de42dfb78ef77042238308e11d82f1fb748f624ba5babae3216c5c53347f`.

| Split/domain | Unchanged B1 | Trained revision | Delta |
|---|---:|---:|---:|
| Development overall | 236/1,289 | 323/1,289 | +87 (+6.749 points) |
| Development MATH-500 | 44/623 | 64/623 | +20 |
| Development logic/science | 190/637 | 257/637 | +67 |
| Development MBPP | 2/29 | 2/29 | 0 |
| Holdout overall | 242/1,279 | 328/1,279 | +86 (+6.724 points) |
| Holdout MATH-500 | 39/621 | 74/621 | +35 |
| Holdout logic/science | 194/625 | 246/625 | +52 |
| Holdout MBPP | 9/33 | 8/33 | -1 |

Development passes all four frozen conditions. Holdout passes the five-point
overall, math, and logic/science conditions but fails nonnegative code by one
answer. The conjunctive result is therefore FAIL and selects `shohin_920m`.
The 0.8B mechanism shows real, source-disjoint overall improvement, but its
code retention is not robust enough to justify the 389M risk. There is no
nearby 0.8B retry family.

This test determined the smallest plausible scratch trunk. It is not a nearby
variant of the failed QST1 workspace: the changed factor is the already
qualified complete-trajectory revision mechanism.

## Data Requirement

Current optimizer-authorized general data are 62.426B historical tokens from
FineMath, OpenWebMath, and Python code. This stream is too narrow and too small
for the intended scratch run without heavy replay.

The most developed fresh candidates include:

- 8.764B selected FineWeb-Edu tokens;
- 4.350B balanced peS2o technical/science tokens;
- 0.308B Essential-Web reasoning-core tokens;
- bounded FinePDF core and verified math/science/code sources.

They remain candidates until their existing provenance, residualization,
privacy, contamination, license, and utility gates close. The production
corpus must be mostly fresh broad data rather than repeated historical math
and code.

Minimum staged targets:

| Stage | Purpose | Unique admitted tokens |
|---|---|---:|
| transport/optimization canary | prove multi-H100 scaling and resume | 0.25B |
| capability canary | compare 390M/920M learning at equal tokens | 5B |
| first milestone | broad NLL and downstream-skill checkpoint | 50B |
| scale run | competitive base representation | >=300B for 920M or >=700B for 390M |

Repeated presentations are reported separately and never counted as unique
corpus size.

## Training Sequence

1. **Base pretraining:** broad next-token training with source-balanced token
   sampling and immutable evaluation monitors.
2. **Reasoning mid-training:** interleave verified explanatory math, code,
   science, logic, and long-form decomposition without answer-only oversampling.
3. **General instruction alignment:** teach concise direct answers, complete
   solutions, code boundaries, and refusal/uncertainty behavior.
4. **Internal draft/revision training:** train both temporal roles on complete
   trajectories, including initially wrong drafts that can be repaired.
5. **Verified-reward training:** begin only after the model generates a useful
   positive rate on independently checked math/code/science tasks; preserve an
   identical no-RLVR continuation control.
6. **Milestone evaluation:** greedy pass@1 plus separately reported test-time
   compute on GSM8K, MATH-500, AIME, executable code, GPQA, and logic, followed
   by direct interaction and untouched distribution-shifted boards.

## Launch Gate

A large run starts only after all of the following are true:

- the 4B product comparison and 0.8B capacity-boundary decision are recorded;
- the selected configuration instantiates at its exact parameter count;
- 1/2/4/8/16-H100 scaling is measured and the efficient geometry selected;
- the physical data contract proves source weights, unique tokens, tokenizer,
  licenses, contamination scans, and shard hashes;
- exact checkpoint/resume and data-cursor continuation pass;
- a 5B-token matched canary shows stable optimization and no broad regression;
- projected H100-hours are recorded against current quota before launch.

The target is not merely a benchmark-specialized adapter. The target is one
deployable Shohin model that retains broad language and code utility while its
own draft/revision path materially improves difficult reasoning.
